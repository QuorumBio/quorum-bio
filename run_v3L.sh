#!/bin/bash
cd /c/QuorumBio
echo "START $(date)" > logs/v3L.txt
venv/Scripts/python.exe ner/scripts/train3.py --base microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract --out ner/models/v3L --lr 2e-5 >> logs/v3L.txt 2>&1
echo "ALL_DONE $(date)" >> logs/v3L.txt
