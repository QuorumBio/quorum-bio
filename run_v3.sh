#!/bin/bash
# NER v3: partial-label run (release candidate) then naive-objective comparison. Selection between them on validation only.
cd /c/QuorumBio
B=microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
echo "START $(date)" > logs/v3.txt
venv/Scripts/python.exe ner/scripts/train3.py --base $B --out ner/models/v3a >> logs/v3.txt 2>&1
venv/Scripts/python.exe ner/scripts/train3.py --base $B --out ner/models/v3n --naive >> logs/v3.txt 2>&1
echo "ALL_DONE $(date)" >> logs/v3.txt
