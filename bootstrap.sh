#!/usr/bin/env bash
# Session 13 bootstrap: unzip -> git repo -> GitHub -> print the Colab links.
#
#   chmod +x bootstrap.sh && ./bootstrap.sh ~/Downloads/era-v5-session13-reversibility.zip
#
# Needs: git, and either the GitHub CLI (`gh`) or an existing empty repo.
# It will NOT run the training -- that needs a GPU session you open yourself.
set -euo pipefail

ZIP="${1:-$HOME/Downloads/era-v5-session13-reversibility.zip}"
OWNER="${OWNER:-rjvim}"
REPO="${REPO:-era-v5-session13-reversibility}"
DIR="${DIR:-$HOME/$REPO}"

[ -f "$ZIP" ] || { echo "zip not found: $ZIP"; exit 1; }

echo "==> unpacking into $DIR"
rm -rf "$DIR"; mkdir -p "$DIR"
unzip -q "$ZIP" -d "$DIR"
cd "$DIR"

echo "==> sanity check (no GPU needed)"
if command -v python3 >/dev/null; then
  python3 -m pytest tests/ -q 2>/dev/null | tail -2 || \
    echo "    (pytest/torch not installed locally -- fine, CI and Colab will run them)"
fi

echo "==> git init"
git init -q
git add -A
git -c user.email="rajivs.iitkgp@gmail.com" -c user.name="$OWNER" \
    commit -q -m "ERA V5 Session 13: reversible LLM training (euler vs midpoint)"
git branch -M main

if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
  echo "==> creating GitHub repo via gh"
  gh repo create "$OWNER/$REPO" --public --source=. --remote=origin --push \
     --description "ERA V5 Session 13 - reversible LLM training: baseline vs euler vs midpoint, 20M params / 50M tokens" \
     || { git remote add origin "https://github.com/$OWNER/$REPO.git" 2>/dev/null || true; git push -u origin main; }
else
  echo "==> gh not available. Create an EMPTY public repo named $REPO, then:"
  echo "    cd $DIR"
  echo "    git remote add origin git@github.com:$OWNER/$REPO.git"
  echo "    git push -u origin main"
  git remote add origin "git@github.com:$OWNER/$REPO.git" 2>/dev/null || true
fi

cat <<EOF

==> pushed (or ready to push): https://github.com/$OWNER/$REPO

Next, the part that needs a GPU. Open Colab, set Runtime -> T4 or A100, and
either run the three notebooks in order:

  https://colab.research.google.com/github/$OWNER/$REPO/blob/main/notebooks/01_setup_and_baseline.ipynb
  https://colab.research.google.com/github/$OWNER/$REPO/blob/main/notebooks/02_reversible_variants.ipynb
  https://colab.research.google.com/github/$OWNER/$REPO/blob/main/notebooks/03_max_batch_and_report.ipynb

...or paste the single cell in COLAB_ONE_CELL.py into a blank Colab notebook
and leave it. Both paths checkpoint every 100 steps, so a dropped session is
resumed by re-running the same cell.
EOF
