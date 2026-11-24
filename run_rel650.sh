#!/bin/bash
cd /c/QuorumBio
{ echo "START $(date)"
venv/Scripts/python.exe loc/scripts/release_loc650.py && \
venv/Scripts/python.exe loc/scripts/bootstrap_loc.py loc/release650 loc/models/run650/test_preds.npz train_results_650m.json loc/models/run1_150m/test_preds.npz && \
venv/Scripts/python.exe loc/scripts/card_loc650.py
echo "REL650_EXIT $? $(date)"; } >> logs/rel650.txt 2>&1
