#!/bin/bash
# NER v3 follow-up: pipeline eval, ONNX, bootstrap for v3a; then pre-registered seeds 1-4 (weights deleted after, results kept).
cd /c/QuorumBio
B=microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
echo "START $(date)" > logs/v3b.txt
venv/Scripts/python.exe ner/scripts/eval_v3.py ner/models/v3a >> logs/v3b.txt 2>&1
venv/Scripts/python.exe ner/scripts/onnx_export3.py ner/models/v3a ner/release3 >> logs/v3b.txt 2>&1
venv/Scripts/python.exe ner/scripts/bootstrap_ner3.py ner/models/v3a >> logs/v3b.txt 2>&1
for s in 1 2 3 4; do
  venv/Scripts/python.exe ner/scripts/train3.py --base $B --out ner/models/v3a_s$s --seed $s >> logs/v3b.txt 2>&1
  rm -f ner/models/v3a_s$s/model.safetensors
done
echo "ALL_DONE $(date)" >> logs/v3b.txt
