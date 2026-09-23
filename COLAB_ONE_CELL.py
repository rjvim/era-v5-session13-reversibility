# ============================================================================
# ERA V5 Session 13 -- paste this ENTIRE cell into a blank Colab notebook.
# Runtime -> Change runtime type -> T4 (or A100). Then run it and walk away.
#
# It clones, installs, verifies correctness, builds 50M tokens, finds your max
# batch, runs all four arms, scans depth, regenerates the README and pushes.
#
# If Colab drops the session: re-run this same cell. Every run checkpoints
# every 100 steps and resumes exactly where it stopped (verified bit-identical
# in tests/test_checkpoint_resume.py). Finished runs are skipped.
# ============================================================================
OWNER = "rjvim"
REPO  = "era-v5-session13-reversibility"
SEQ_LEN = 512
TOKENS  = 50_000_000
EULER_ITERS = 8          # drop to 4 if the euler arm is dragging
PUSH = True              # needs a GitHub token; set False to just produce files

import os, json, subprocess, sys, textwrap

def sh(cmd, check=True):
    print(f"\n$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check)

print(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv",
                     shell=True, capture_output=True, text=True).stdout)

if not os.path.isdir(REPO):
    sh(f"git clone https://github.com/{OWNER}/{REPO}.git")
os.chdir(f"/content/{REPO}" if os.path.isdir(f"/content/{REPO}") else REPO)
sh("pip -q install -r requirements.txt")
os.environ["PYTHONPATH"] = "src"

# ---- 1. correctness before any timing number -------------------------------
sh("pytest tests/test_reversibility.py -q")

# ---- 2. data ---------------------------------------------------------------
if not os.path.exists("data/train.bin"):
    sh(f"python src/data.py --out_dir data --tokens {TOKENS}")

COMMON = f"--seq_len {SEQ_LEN} --total_tokens {TOKENS} --resume"

# ---- 3. max batch per mode -------------------------------------------------
for mode in ("baseline", "midpoint", "euler"):
    if not os.path.exists(f"results/maxbatch_{mode}.json"):
        sh(f"python src/train.py --mode {mode} --find_max_batch --seq_len {SEQ_LEN}")

B_FIX = json.load(open("results/maxbatch_baseline.json"))["max_batch"]
B_MAX = json.load(open("results/maxbatch_midpoint.json"))["max_batch"]
print(f"\n>>> fixed batch {B_FIX} | max reversible batch {B_MAX} "
      f"({B_MAX/B_FIX:.2f}x)\n")
if B_FIX < 2:
    sys.exit("batch of 1 -- this GPU is too small for the config; reduce "
             "seq_len or n_layer before continuing")

# ---- 4. the four runs ------------------------------------------------------
runs = [
    ("baseline_fixed", f"--mode baseline --batch_size {B_FIX}"),
    ("midpoint_fixed", f"--mode midpoint --batch_size {B_FIX} --h 0.5 --check_recon"),
    ("euler_fixed",    f"--mode euler --batch_size {B_FIX} --h 0.5 "
                       f"--euler_iters {EULER_ITERS} --check_recon"),
    ("midpoint_max",   f"--mode midpoint --batch_size {B_MAX} --h 0.5 --check_recon"),
]
for name, args in runs:
    if os.path.exists(f"results/{name}.json"):
        print(f"skip {name} (already done)"); continue
    sh(f"python src/train.py {args} --run_name {name} {COMMON}")

# ---- 5. depth scan, report, verification -----------------------------------
if not os.path.exists("results/depth_scaling.json"):
    sh(f"python scripts/depth_scan.py --layers 4 8 16 32 64 --batch_size 8 --seq_len {SEQ_LEN}")
sh("python scripts/saved_bytes_scan.py")
sh("python scripts/make_readme.py")
sh("pytest tests/ -q")          # must show NO skips: a skip = a missing run

# ---- 6. summary ------------------------------------------------------------
print("\n" + "=" * 78)
for name, _ in runs:
    d = json.load(open(f"results/{name}.json"))
    re_ = d["max_recon_err"]
    print(f"{name:16s} batch {d['batch_size']:>5} | loss {d['final_train_loss']:.4f} "
          f"| val {d['final_val_loss']:.4f} | {d['tok_per_s']:>9,.0f} tok/s "
          f"| peak {d['peak_mem_gb']:.2f} GB | recon "
          f"{(f'{re_:.2e}' if re_ else 'n/a')}")
print("=" * 78)

# ---- 7. push ---------------------------------------------------------------
if PUSH:
    from google.colab import userdata
    try:
        tok = userdata.get("GITHUB_TOKEN")   # Colab: key icon -> add secret
        sh(f"git remote set-url origin https://{tok}@github.com/{OWNER}/{REPO}.git")
        sh("git add -A")
        sh("git -c user.email=rajivs.iitkgp@gmail.com -c user.name=rjvim "
           "commit -q -m 'session 13: GPU results + generated README'", check=False)
        sh("git push")
        print(f"\ndone: https://github.com/{OWNER}/{REPO}")
    except Exception as e:
        print(f"push skipped ({e}). Download results/ and commit locally, or "
              f"add a GITHUB_TOKEN secret via the key icon in the left sidebar.")
