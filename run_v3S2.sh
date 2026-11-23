#!/bin/bash
cd /c/QuorumBio
echo "START $(date)" > logs/v3S2.txt
venv/Scripts/python.exe ner/scripts/eval_v3.py ner/models/v3S >> logs/v3S2.txt 2>&1
venv/Scripts/python.exe ner/scripts/onnx_export3.py ner/models/v3S ner/release3s >> logs/v3S2.txt 2>&1
venv/Scripts/python.exe ner/scripts/bootstrap_ner3.py ner/models/v3S >> logs/v3S2.txt 2>&1
for s in 1 2 3 4; do
  venv/Scripts/python.exe ner/scripts/train3_distill.py --teacher ner/models/v3a --keep 1,3,5,7,9,11 --out ner/models/v3S_s$s --seed $s >> logs/v3S2.txt 2>&1
  rm -f ner/models/v3S_s$s/model.safetensors
done
echo "ALL_DONE $(date)" >> logs/v3S2.txt
