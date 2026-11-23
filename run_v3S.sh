#!/bin/bash
# waits for the large run to finish, then trains the distilled small model
cd /c/QuorumBio
until grep -q "ALL_DONE" logs/v3L.txt; do sleep 10; done
echo "START $(date)" > logs/v3S.txt
venv/Scripts/python.exe ner/scripts/train3_distill.py --teacher ner/models/v3a --keep 1,3,5,7,9,11 --out ner/models/v3S >> logs/v3S.txt 2>&1
echo "ALL_DONE $(date)" >> logs/v3S.txt
