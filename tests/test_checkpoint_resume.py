"""A four-hour Colab run must survive a disconnect.

Resuming from a checkpoint has to reproduce the uninterrupted run exactly --
same weights, same optimiser moments, same position in the token stream. If it
does not, a resumed run silently reports numbers from a different experiment.
"""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")


def _train(tmp, name, extra, timeout=600):
    cmd = [sys.executable, os.path.join(SRC, "train.py"), "--mode", "midpoint",
           "--run_name", name, "--total_tokens", "81920", "--seq_len", "64",
           "--n_layer", "4", "--n_embd", "64", "--n_head", "4", "--vocab_size", "512",
           "--device", "cpu", "--dtype", "fp32", "--batch_size", "4", "--log_every", "10",
           "--eval_steps", "2", "--ckpt_every", "10", "--quiet",
           "--data_dir", tmp, "--results_dir", tmp] + extra
    env = dict(os.environ, PYTHONPATH=SRC)
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)


def test_resume_reproduces_uninterrupted_run(tmp_path):
    tmp = str(tmp_path)
    sys.path.insert(0, SRC)
    import data
    data.make_synthetic(tmp, 100_000, vocab=512)

    assert _train(tmp, "full", []).returncode == 0

    # genuine disconnect: kill the process part-way through
    killed = False
    for t in (3, 2, 1):
        try:
            _train(tmp, "part", ["--keep_ckpt"], timeout=t)
        except subprocess.TimeoutExpired:
            # a kill that lands after the final write is not an interruption
            if not os.path.exists(os.path.join(tmp, "part.json")):
                killed = True
                break
            os.remove(os.path.join(tmp, "part.json"))
    if not killed or not os.path.exists(os.path.join(tmp, "part.ckpt")):
        pytest.skip("could not interrupt mid-run on this machine")

    assert _train(tmp, "part", ["--resume"]).returncode == 0

    a = json.load(open(os.path.join(tmp, "full.json")))
    b = json.load(open(os.path.join(tmp, "part.json")))
    assert b["final_train_loss"] == pytest.approx(a["final_train_loss"], abs=1e-9)
    assert b["final_val_loss"] == pytest.approx(a["final_val_loss"], abs=1e-9)
    assert b["tokens_trained"] == a["tokens_trained"]


def test_checkpoint_write_is_atomic(tmp_path):
    """A kill during torch.save must not leave a half-written checkpoint."""
    sys.path.insert(0, SRC)
    import torch
    from train import save_ckpt
    import model as M
    m = M.GPT(M.GPTConfig(vocab_size=64, block_size=8, n_layer=2, n_head=2, n_embd=16))
    o = torch.optim.AdamW(m.parameters(), lr=1e-4)
    p = str(tmp_path / "a.ckpt")
    save_ckpt(p, m, o, 5, 100, [], [], 1000, 1.0)
    assert os.path.exists(p) and not os.path.exists(p + ".tmp")
    torch.load(p, map_location="cpu", weights_only=False)   # loads cleanly
