#!/usr/bin/env bash
# Rare-class screen on ESM-2 35M (2 epochs). Dev-only (--no_test): test is NOT scored during screening.
# Variants fixed in advance: A baseline (pw cap 10), B pw cap 100, C focal loss, D attention-pooling head.
cd "$(dirname "$0")"
PY=venv/Scripts/python.exe; LOG=logs/rare_screen.txt
echo "start $(date -Is)" > $LOG
for seed in 13 1; do
  for v in "A:" "B:--pw_clip 100" "C:--loss focal" "D:--pool attn"; do
    name=${v%%:*}; args=${v#*:}
    echo "RUN $name seed $seed $(date -Is)" >> $LOG
    $PY -u loc/scripts/train_loc.py --base facebook/esm2_t12_35M_UR50D --epochs 2 --seed $seed --no_test $args \
        --out loc/models/screen/${name}_s$seed >> $LOG 2>&1
    echo "DONE $name seed $seed exit $? $(date -Is)" >> $LOG
  done
done
echo "RUN_END $(date -Is)" >> $LOG
