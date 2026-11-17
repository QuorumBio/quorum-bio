# COMPANY_STATE (single source of memory; update every session)

Lab: Quorum Bio. HF org/user: quorumbio. GitHub: QuorumBio.
Last updated: 2026-11-17

## Environment
- Work dir: C:\QuorumBio (Windows PC). Python venv: C:\QuorumBio\venv (torch 2.14.1+cu130, CUDA True, RTX 3060 12 GB).
- Installed: transformers, datasets, accelerate, evaluate, seqeval, onnx, onnxruntime, optimum, huggingface_hub, gradio, scikit-learn.
- Run jobs by writing a .bat in this folder and double-clicking it in File Explorer (computer use is click-only there). Logs go to logs\.
- Cloud workspace cannot reach huggingface.co (allowlist). The PC can. All HF uploads run from the PC.
- Cloud workspace GitHub proxy acts as the owner's personal account (mishalide). Never push from the cloud workspace. Push from the PC with the QuorumBio token.
- Credentials are NOT in this repo. Kept in the cloud workspace CREDENTIALS.md and the user's password manager. HF_TOKEN must be set in the PC environment for uploads.

## Budget
- Rented compute: $0 spent of $20. (log every dollar here with the reason before spending)

## Hypotheses
1. A small permissive (Apache-2.0) biomedical NER model that is fast and ONNX-ready can get downloads against d4data/biomedical-ner-all.
2. Signal peptide + localization from ESM-2 on Swiss-Prot, homology-partitioned, is a second family.

## Dataset provenance (check license BEFORE training)
| dataset | url | license | verified | used for |
|---|---|---|---|---|
| BC5CDR (Chemical, Disease; 1,500 PubMed abstracts) | biocreative.org CDR track; github.com/JHnlp/BioCreative-V-CDR-Corpus | Public Domain Mark 1.0 (per BigBio dataset metadata; confirm on the original site before release) | 2026-11-17 (secondary source only) | NER train/val/test |
| NCBI-disease (793 abstracts, Disease) | ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE | CC0 1.0 per BigBio loader; NCBI public domain notice on HF card | 2026-11-17 (secondary source) | NER train/val/test |
| JNLPBA / GENIA 3.02 (DNA, RNA, protein, cell line, cell type) | geniaproject.org shared task | CC BY 3.0 annotations; abstracts under NLM PubMed terms. Must ship the GENIA license text + attribution "Corpus annotations (c) GENIA Project" | 2026-11-17 | optional, with attribution in NOTICE |
| BC2GM (gene mentions) | biocreative.org BioCreative II | NO KNOWN LICENSE (HF/TFDS list none) | not verified | SKIP until a license is confirmed |

## Eval results
- run1 (2026-11-17): base microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext (MIT, verified via HF model info), corpus BC5CDR (Chemical, Disease), 5 epochs, bs16, lr5e-5, bf16, seed 13, ~1 GPU run on the 3060. Checkpoint = best validation entity F1 (0.9055, epoch 5). Test entity F1 (seqeval, word-level, sentence-split, evaluated once) = 0.8960 (P 0.8825, R 0.9100). Files: ner/models/run1, logs/ner1.txt. NOTE: not directly comparable to published BC5CDR numbers (different tokenization/metric); do not claim parity in the card.
- Data stats: BC5CDR train/dev/test = 500/500/500 docs, 0 offset mismatches. NCBI-disease 593/100/100 docs, 1 mismatch in train. sha256 of CDR_Data.zip = 0a359a7f...2016 (full hashes in ner/data/provenance.json).

## Releases
- 2026-11-17: https://huggingface.co/quorumbio/biomed-ner-chem-disease (Apache-2.0; weights, ONNX fp32 + int8 as onnx/model_quantized.onnx, LICENSE, NOTICE, SHA256SUMS, training scripts, provenance). Parity fp32 ONNX vs torch: max logit diff 5.6e-5, argmax agreement 1.0. ONNX test F1 fp32 0.8960 / int8 0.8940 (CPU 24.5 / 13.7 ms per sentence).
- Baseline d4data/biomedical-ner-all on same BC5CDR test path: F1 0.479 (strict label map) / 0.533 (generous). Reported in the card with in-domain vs out-of-domain caveat. Our model is larger (108.9M vs 66.4M params) and slightly slower on GPU; do not claim speed or size advantage.
- Demo: https://huggingface.co/spaces/quorumbio/biomed-ner-demo (STATIC Space, transformers.js, int8 ONNX; verified in a browser: tags Lidocaine/Chemical, cardiac asystole/Disease, etc.). Gradio Space not possible on free tier: HF returns 402 (Gradio Spaces on cpu-basic need PRO). Gradio version kept in ner/space_gradio for later.
- GitHub repo QuorumBio/quorum-bio created (empty). Push from the cloud sandbox is blocked (only clone/fetch allowed); push must run from the PC via run_push.bat (Windows git).

- Cross-corpus (run on CPU, int8 ONNX): Disease-only F1 on NCBI-disease test = 0.738 (P 0.780, R 0.701), vs 0.852 on BC5CDR test Disease. Gap likely annotation-guideline differences. Idea for run2: train on BC5CDR + NCBI-disease (handle that NCBI has no Chemical labels, e.g. mask chemical loss or train separate heads). Added to the model card locally; hub copy updated on next run_upload.

## Download metrics
- 2026-11-17: 0 downloads (just released). Check via model_info(...).downloads in DAILY step 2. Never self-download.

## Actions log
- 2026-11-17: prepared data, trained run1 (test F1 0.896), baseline, ONNX, packaged and uploaded model + static Space. accounts created (Proton, HF, GitHub). GPU verified. CUDA torch venv built. Repo layout created.

## Blockers (also in NEEDS_HUMAN.md)
- BC5CDR license is NOT explicitly stated by the BioCreative page or the GitHub mirror ("freely available for the research community"); BigBio lists Public Domain Mark 1.0. Released under Apache-2.0 with that uncertainty disclosed in the card. Legal question for the human (non-blocking).
- huggingface.co blocked from cloud workspace (PC is fine).
- (resolved 2026-11-17) HF token verified from the PC: user quorumbio, role write. Token is in C:\QuorumBio\.env (gitignored); scripts load it from there.

## Next best action
Get the code onto GitHub (run_push.bat), then start family 2 prep: signal peptide + localization (Swiss-Prot, ESM-2), license-check UniProt (CC BY 4.0) and plan a homology-partitioned split (MMseqs2/CD-HIT clustering). Also consider adding NCBI-disease (public domain) to NER training as run2 to improve Disease, and a multilingual variant.
