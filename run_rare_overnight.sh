#!/usr/bin/env bash
# Overnight: finish the 35M rare-class screen (dev-only), pick the winner on dev with a pre-set rule,
# then train the winner on 150M with seeds 13, 1, 2 (test scored once per run) for comparison with the 3-seed baseline.
cd "$(dirname "$0")"
export PATH="/usr/bin:$PATH"; PY=venv/Scripts/python.exe; LOG=logs/rare_overnight.txt
echo "start $(date -Is)" >> $LOG
for seed in 13 1; do
  for v in "A:" "B:--pw_clip 100" "C:--loss focal" "D:--pool attn"; do
    name=${v%%:*}; args=${v#*:}; out=loc/models/screen/${name}_s$seed
    [ -f $out/dev_results.json ] && { echo "SKIP $name seed $seed (done)" >> $LOG; continue; }
    echo "RUN $name seed $seed $(date -Is)" >> $LOG
    $PY -u loc/scripts/train_loc.py --base facebook/esm2_t12_35M_UR50D --epochs 2 --seed $seed --no_test $args --out $out >> $LOG 2>&1
    echo "DONE $name seed $seed exit $? $(date -Is)" >> $LOG
  done
done
W=$($PY loc/scripts/pick_rare.py 2>> $LOG); echo "PICK [$W] $(date -Is)" >> $LOG
if [ "$W" != "NONE" ] && [ -n "$W" ]; then
  for seed in 13 1 2; do
    out=loc/models/rare150/s$seed
    [ -f $out/results.json ] && continue
    echo "RUN150 [$W] seed $seed $(date -Is)" >> $LOG
    $PY -u loc/scripts/train_loc.py --base facebook/esm2_t30_150M_UR50D --epochs 2 --seed $seed $W --out $out >> $LOG 2>&1
    echo "DONE150 seed $seed exit $? $(date -Is)" >> $LOG
  done
fi
echo "RUN_END $(date -Is)" >> $LOG
