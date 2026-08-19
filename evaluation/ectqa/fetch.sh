#!/bin/sh
# Fetch ECT-QA (austinmyc/ECT-QA on Hugging Face): 384 earnings call transcripts
# and the question sets. ~27MB, public and ungated, so no token is needed.
#
# The corpus is not committed — it is someone else's dataset, it is large, and
# pinning a copy here would silently diverge from upstream.
set -e
cd "$(dirname "$0")"
BASE="https://huggingface.co/datasets/austinmyc/ECT-QA/resolve/main"

mkdir -p data

echo "fetching question sets..."
for f in local_questions_old local_questions_new global_questions_old global_questions_new; do
    [ -f "$f.json" ] || curl -sL -o "$f.json" "$BASE/questions/$f.json"
done

echo "listing corpus..."
curl -s "https://huggingface.co/api/datasets/austinmyc/ECT-QA" \
  | python3 -c "import json,sys; print('\n'.join(f['rfilename'] for f in json.load(sys.stdin)['siblings'] if f['rfilename'].startswith('data/old/')))" \
  > filelist.txt

total=$(wc -l < filelist.txt | tr -d ' ')
echo "fetching $total transcripts..."
n=0
while read -r f; do
    out="data/$(basename "$f")"
    [ -f "$out" ] || curl -sL -o "$out" "$BASE/$f"
    n=$((n + 1))
    [ $((n % 50)) -eq 0 ] && echo "  $n/$total"
done < filelist.txt

echo "done: $(ls data | wc -l | tr -d ' ') transcripts in data/"
