#!/usr/bin/env bash
# The four runs the assignment asks for. BATCH_FIX must be the largest batch
# the BASELINE arm can hold -- discovered by step 0, not guessed.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src

COMMON="--seq_len ${SEQ_LEN:-512} --total_tokens ${TOKENS:-50000000} --data_dir data --results_dir results --resume"

echo "== step 0: find max batch per mode =="
for m in baseline euler midpoint; do
  python src/train.py --mode $m --find_max_batch $COMMON
done
BATCH_FIX=$(python -c "import json;print(json.load(open('results/maxbatch_baseline.json'))['max_batch'])")
BATCH_MAX=$(python -c "import json;print(json.load(open('results/maxbatch_midpoint.json'))['max_batch'])")
echo "fixed batch = $BATCH_FIX   max reversible batch = $BATCH_MAX"

echo "== run 1: baseline @ fixed batch =="
python src/train.py --mode baseline --batch_size $BATCH_FIX --run_name baseline_fixed $COMMON

echo "== run 2a: euler @ fixed batch =="
python src/train.py --mode euler    --batch_size $BATCH_FIX --run_name euler_fixed    --check_recon $COMMON

echo "== run 2b: midpoint @ fixed batch =="
python src/train.py --mode midpoint --batch_size $BATCH_FIX --run_name midpoint_fixed --check_recon $COMMON

echo "== run 3: midpoint @ max batch =="
python src/train.py --mode midpoint --batch_size $BATCH_MAX --run_name midpoint_max   --check_recon $COMMON

python scripts/depth_scan.py
python scripts/make_readme.py
pytest tests/ -q
