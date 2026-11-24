#!/bin/bash
# ESM-2 650M loc run (pre-registered fallback: bottom 16 layers frozen, batch 2 x accum 4). Resumes after a crash, up to 3 times.
cd /c/QuorumBio
echo "START $(date)" > logs/loc650_s1.txt
for attempt in 1 2 3 4; do
  venv/Scripts/python.exe loc/scripts/train_loc.py --base facebook/esm2_t33_650M_UR50D --out loc/models/run650_s1 --seed 1 --bs 2 --accum 4 --freeze_layers 16 --mem_fraction 0.92 --resume >> logs/loc650_s1.txt 2>&1
  grep -q TRAIN_LOC_DONE logs/loc650_s1.txt && break
  echo "ATTEMPT $attempt EXITED $(date); resuming" >> logs/loc650_s1.txt; sleep 30
done
echo "ALL_DONE $(date)" >> logs/loc650_s1.txt
