"""One arm of the session-13 experiment.

Every run trains the SAME model on the SAME 50M tokens in the SAME order with
the SAME optimiser and schedule. The only knobs that change between arms are
`--mode` (baseline | euler | midpoint) and `--batch_size`.

Emits results/<run_name>.json containing everything the README claims.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import time

import torch

from data import TokenLoader
from model import GPT, GPTConfig


def get_lr(step, total, warmup, lr, min_lr):
    if step < warmup:
        return lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    r = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * r)) * (lr - min_lr)


def save_ckpt(path, model, opt, step, loader_pos, log, recon, t_tokens, elapsed):
    """Atomic: write to .tmp then rename, so a kill mid-write cannot corrupt the
    checkpoint a 4-hour run depends on."""
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                "loader_pos": loader_pos, "log": log, "recon": recon,
                "tokens": t_tokens, "elapsed": elapsed,
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
               tmp)
    os.replace(tmp, path)


def load_ckpt(path, model, opt):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    torch.set_rng_state(ck["rng"].cpu() if hasattr(ck["rng"], "cpu") else ck["rng"])
    if ck.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ck["cuda_rng"])
    return ck


def build(args, device):
    cfg = GPTConfig(
        vocab_size=args.vocab_size, block_size=args.seq_len, n_layer=args.n_layer,
        n_head=args.n_head, n_embd=args.n_embd, mode=args.mode, h=args.h,
        euler_iters=args.euler_iters, dropout=0.0,
    )
    return GPT(cfg).to(device), cfg


def gpu_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return platform.processor() or "cpu"


def peak_mem_gb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 3
    return float("nan")


@torch.no_grad()
def evaluate(model, loader, steps, ctx):
    model.eval()
    loader.reset()
    tot = 0.0
    for _ in range(steps):
        x, y = loader.next_batch()
        with ctx:
            _, loss = model(x, y)
        tot += loss.item()
    model.train()
    return tot / steps


def run(args):
    device = args.device
    torch.manual_seed(args.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model, cfg = build(args, device)
    amp = args.dtype == "bf16" and device.startswith("cuda")
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if amp \
        else torch.autocast(device_type="cpu", enabled=False)

    # No weight decay anywhere: reversible arms forbid it, so the baseline
    # drops it too, otherwise the loss comparison would not be like-for-like.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.0, eps=1e-8)

    tok_per_step = args.batch_size * args.seq_len * args.grad_accum
    total_steps = args.total_tokens // tok_per_step
    warmup = max(1, int(0.02 * total_steps))

    train = TokenLoader(os.path.join(args.data_dir, "train.bin"),
                        args.batch_size, args.seq_len, device)
    val = TokenLoader(os.path.join(args.data_dir, "val.bin"),
                      args.batch_size, args.seq_len, device)

    log, t_tokens, recon = [], 0, []
    start_step, prior_elapsed = 0, 0.0
    os.makedirs(args.results_dir, exist_ok=True)
    ckpt = args.ckpt or os.path.join(args.results_dir, f"{args.run_name}.ckpt")
    if args.resume and os.path.exists(ckpt):
        ck = load_ckpt(ckpt, model, opt)
        start_step = ck["step"] + 1
        train.pos, log, recon = ck["loader_pos"], ck["log"], ck["recon"]
        t_tokens, prior_elapsed = ck["tokens"], ck["elapsed"]
        print(f"resumed {args.run_name} at step {start_step}/{total_steps} "
              f"({t_tokens:,} tokens already done)", flush=True)

    model.train()
    t0 = time.time() - prior_elapsed
    for step in range(start_step, total_steps):
        lr = get_lr(step, total_steps, warmup, args.lr, args.lr * 0.1)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        acc = 0.0
        check = args.check_recon and (step % args.log_every == 0)
        for _ in range(args.grad_accum):
            x, y = train.next_batch()
            with ctx:
                _, loss = model(x, y, check_recon=check)
            (loss / args.grad_accum).backward()
            acc += loss.item() / args.grad_accum
        if check and cfg.mode != "baseline":
            recon.append([step, model.last_recon_err])
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        t_tokens += tok_per_step

        if step % args.log_every == 0 or step == total_steps - 1:
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            dt = time.time() - t0
            log.append({"step": step, "loss": acc, "lr": lr,
                        "grad_norm": float(gn), "tokens": t_tokens,
                        "elapsed_s": dt, "tok_per_s": t_tokens / dt})
            if not args.quiet:
                print(f"step {step:6d}/{total_steps} loss {acc:.4f} "
                      f"lr {lr:.2e} {t_tokens/dt:,.0f} tok/s "
                      f"peak {peak_mem_gb():.2f}GB", flush=True)

        # checkpoint AFTER logging: otherwise a resumed run restores a log that
        # is missing up to log_every steps, and reports a stale final loss
        if args.ckpt_every and (step % args.ckpt_every == 0 or step == total_steps - 1):
            save_ckpt(ckpt, model, opt, step, train.pos, log, recon,
                      t_tokens, time.time() - t0)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    wall = time.time() - t0
    # snapshot BEFORE eval: peak must describe training, not the eval pass
    train_peak = peak_mem_gb()
    val_loss = evaluate(model, val, args.eval_steps, ctx)

    out = {
        "run_name": args.run_name,
        "mode": args.mode,
        "h": args.h if args.mode != "baseline" else None,
        "euler_iters": args.euler_iters if args.mode == "euler" else None,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "seq_len": args.seq_len,
        "tokens_per_step": tok_per_step,
        "total_steps": total_steps,
        "tokens_trained": t_tokens,
        "final_train_loss": log[-1]["loss"],
        "final_val_loss": val_loss,
        "tok_per_s": t_tokens / wall,
        "wall_s": wall,
        "peak_mem_gb": train_peak,
        "params_total": model.num_params(),
        "params_non_embedding": model.num_params(non_embedding=True),
        "recon_err_log": recon,
        "max_recon_err": max((r[1] for r in recon), default=None),
        "device": gpu_name(),
        "dtype": args.dtype,
        "torch": torch.__version__,
        "seed": args.seed,
        "config": cfg.dict(),
        "log": log,
    }
    os.makedirs(args.results_dir, exist_ok=True)
    path = os.path.join(args.results_dir, f"{args.run_name}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}")
    if args.ckpt_every and os.path.exists(ckpt) and not args.keep_ckpt:
        os.remove(ckpt)          # run finished; the JSON is the artifact
    return out


def find_max_batch(args):
    """Doubling then binary search for the largest batch that survives a real
    train step (fwd + bwd + optimiser), not just a forward pass."""
    if not args.device.startswith("cuda"):
        raise SystemExit(
            "--find_max_batch needs a CUDA device: it searches for the batch that "
            "triggers OOM, and on CPU nothing OOMs -- it would just allocate until "
            "the machine dies. Use scripts/saved_bytes_scan.py for a device-"
            "independent activation-memory measurement instead.")
    device = args.device

    def fits(B):
        m = o = None
        try:
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            m, _ = build(args, device)
            o = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.0)
            amp = args.dtype == "bf16" and device.startswith("cuda")
            c = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if amp \
                else torch.autocast(device_type="cpu", enabled=False)
            for _ in range(3):                       # 3 steps: caches settle
                x = torch.randint(0, args.vocab_size, (B, args.seq_len), device=device)
                with c:
                    _, loss = m(x, x)
                loss.backward()
                o.step()
                o.zero_grad(set_to_none=True)
            return True, peak_mem_gb()
        except torch.cuda.OutOfMemoryError:
            return False, None
        except RuntimeError as e:                 # older torch raises plain RuntimeError
            if "out of memory" in str(e).lower():
                return False, None
            raise
        finally:
            # must drop the refs BEFORE empty_cache, or a failed probe leaves
            # its model resident and poisons the next probe with a false OOM
            del m, o
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    lo, hi, peak_at_lo = 0, 1, None
    while True:
        ok, pk = fits(hi)
        print(f"  batch {hi}: {'ok' if ok else 'OOM'}"
              + (f" peak {pk:.2f}GB" if ok else ""), flush=True)
        if not ok:
            break
        lo, peak_at_lo, hi = hi, pk, hi * 2
        if hi > args.max_batch_cap:
            break
    while hi - lo > 1:
        mid = (lo + hi) // 2
        ok, pk = fits(mid)
        print(f"  batch {mid}: {'ok' if ok else 'OOM'}"
              + (f" peak {pk:.2f}GB" if ok else ""), flush=True)
        if ok:
            lo, peak_at_lo = mid, pk
        else:
            hi = mid
    res = {"mode": args.mode, "seq_len": args.seq_len, "max_batch": lo,
           "peak_mem_gb_at_max": peak_at_lo, "device": gpu_name(),
           "dtype": args.dtype, "h": args.h}
    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, f"maxbatch_{args.mode}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    return res


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="baseline", choices=["baseline", "euler", "midpoint"])
    p.add_argument("--run_name", default=None)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--n_layer", type=int, default=10)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_embd", type=int, default=256)
    p.add_argument("--vocab_size", type=int, default=50257)
    p.add_argument("--h", type=float, default=0.5)
    p.add_argument("--euler_iters", type=int, default=8)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--total_tokens", type=int, default=50_000_000)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_steps", type=int, default=20)
    p.add_argument("--check_recon", action="store_true")
    p.add_argument("--find_max_batch", action="store_true")
    p.add_argument("--max_batch_cap", type=int, default=4096)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--ckpt", default=None, help="checkpoint path (default results/<run>.ckpt)")
    p.add_argument("--ckpt_every", type=int, default=100, help="steps between checkpoints; 0 disables")
    p.add_argument("--resume", action="store_true", help="continue from checkpoint if present")
    p.add_argument("--keep_ckpt", action="store_true")
    a = p.parse_args()
    if a.run_name is None:
        a.run_name = f"{a.mode}_b{a.batch_size}"
    return a


if __name__ == "__main__":
    args = cli()
    if args.find_max_batch:
        find_max_batch(args)
    else:
        run(args)
