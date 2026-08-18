#!/usr/bin/env bash
# Fetch the LongMemEval dataset. Public and ungated; ~15MB for oracle.
set -euo pipefail
cd "$(dirname "$0")"
BASE=https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main
for variant in "${@:-oracle}"; do
  echo "[+] longmemeval_${variant}"
  curl -fsSL -o "${variant}.json" "${BASE}/longmemeval_${variant}"
done
