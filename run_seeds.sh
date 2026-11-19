#!/usr/bin/env bash
# Multi-seed runs for confidence intervals. Seeds fixed in advance; every run is reported.
# Released checkpoints are unchanged (seed 13); these runs only measure seed variance.
cd "$(dirname "$0")"
PY=venv/Scripts/python.exe
LOG=logs/seeds.txt
echo "start $(date -Is)" > $LOG
for s in 1 2 3 4; do
  $PY -u ner/scripts/train2.py --base microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
      --out ner/models/seeds/run2a_s$s --seed $s >> $LOG 2>&1
  echo "NER_SEED_DONE $s $(date -Is)" >> $LOG
done
for s in 1 2; do
  $PY -u loc/scripts/train_loc.py --base facebook/esm2_t30_150M_UR50D --out loc/models/seeds/run1_150m_s$s \
      --epochs 2 --seed $s >> $LOG 2>&1
  echo "LOC_SEED_DONE $s $(date -Is)" >> $LOG
done
echo "RUN_END $(date -Is)" >> $LOG
