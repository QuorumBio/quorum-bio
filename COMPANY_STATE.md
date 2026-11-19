# COMPANY_STATE (single source of memory; update every session)

Lab: Quorum Bio. HF org/user: quorumbio. GitHub: QuorumBio.
Last updated: 2026-11-18 (evening session)

## Environment
- Work dir: C:\QuorumBio (Windows PC). Python venv: C:\QuorumBio\venv (torch 2.14.1+cu130, CUDA True, RTX 3060 12 GB).
- Installed: transformers, datasets, accelerate, evaluate, seqeval, onnx, onnxruntime, optimum, huggingface_hub, gradio, scikit-learn.
- Run jobs by writing a .bat in this folder and double-clicking it in File Explorer (computer use is click-only there). Logs go to logs\.
- Cloud workspace cannot reach huggingface.co (allowlist). The PC can. All HF uploads run from the PC.
- Cloud workspace GitHub proxy acts as the owner's personal account (mishalide). Never push from the cloud workspace. Push from the PC with the QuorumBio token.
- 2026-11-18 evening: session ran as Claude Code directly on the Windows PC (Git Bash + venv). From there HF API, GitHub and the GPU are reachable directly, so no .bat double-click is needed. Gotcha: a bare `cat > file` with no heredoc waits on stdin and hangs the shell.
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
- NCBI-disease same-path comparison (fp32, Disease only): run1 F1 0.735; d4data 0.377 (strict) / 0.329 (generous). Mapping is ours, annotation conventions differ.
- 2026-11-17: pushed code to GitHub QuorumBio/quorum-bio; HF card updated with cross-corpus section. HF downloads_30d 0, likes 1.
- run2a (BC5CDR+NCBI-disease merged, naive): test F1 BC5CDR 0.8890, NCBI-disease 0.8697 (now in-domain). run1 was 0.8960 on BC5CDR, so BC5CDR is slightly lower (single seed, within likely noise) while NCBI-disease rose from 0.735 (out-of-domain). Checkpoint = best mean val F1 across corpora.
- run2b (same but O labels ignored for NCBI sentences, since NCBI lacks Chemical labels): BC5CDR 0.8750, NCBI 0.4267. Bad: precision collapses without negatives. Do not use. Naive merge wins.
- Next: release run2a as v2 if user/next session agrees (re-export ONNX, re-pack card, upload); then family 2 (signal peptide/localization).
- v2 packaged locally in ner/release (run2a, ONNX int8 BC5CDR F1 0.886 / NCBI 0.879, parity ok). Pipeline path: run2a BC5CDR 0.889, NCBI 0.878; run1 0.896 / 0.735. PUBLISHED to hub 2026-11-17 (v1 tagged v1.0). GitHub still lacks the v2 scripts (eval_v2, train2, onnx_export2, release_pack2, upload_v2); run run_finish.bat or run_push.bat again to push them.
- Family 2 (localization + signal peptide) started 2026-11-17. Written, not yet run: loc/scripts/prepare_uniprot.py (UniProt Swiss-Prot reviewed TSV via REST, CC BY 4.0; locations kept only with ECO:0000269 experimental evidence, mapped to 10 DeepLoc-style classes; SP label = UniProt SIGNAL feature; MMseqs2 cluster split at 30% identity, then dev/test seqs with any >=30% hit to train removed) and loc/scripts/train_loc.py (ESM-2 150M, MIT license; multi-label loc head + SP head; dev-selected, thresholds on dev, test once). Launcher: run_loc_prep.bat (prep + 35M smoke), then a full 150M run. UniProt is blocked from the cloud and PC VM, so everything runs on Windows. Published SignalP/DeepLoc use different partitions, so compare only qualitatively and say so in the card.
- loc dataset built 2026-11-17 (loc/data, gitignored; provenance.json): 37,733 Swiss-Prot entries with experimental locations; 20,269 clusters at 30% id; final split train 30,134 / dev 2,457 / test 2,336 after removing 1,328 dev and 1,478 test seqs with >=30% hits to train; SP positives 5,636/385/450. Smoke train (35M, 400 seqs) ran OK end to end. Windows MMseqs2 notes: easy-cluster/easy-search segfault, use createdb/cluster/createtsv/search/convertalis; needs busybox applets installed (script does it).
- Next: run_loc1.bat (35M 3 epochs then 150M 2 epochs). Then export, card (compare qualitatively to DeepLoc 2.0 / SignalP 6, different partitions), ONNX, Space.
- 2026-11-18: loc run1 finished. 35M (best epoch 2): loc macro AUROC 0.906, macro F1 0.604, SP AUROC 0.994 / MCC 0.956. 150M (2 epochs, 99 min): loc macro AUROC 0.926, macro F1 0.650, SP AUROC 0.994 / MCC 0.956. Rare classes (Golgi, Lysosome/Vacuole, Peroxisome) still weak on F1/AP. Models in loc/models/run1_35m and run1_150m (model.pt, gitignored). Single seed. 150M is the release candidate.
- 2026-11-18: user reported HF shows 34 downloads on the NER model (screenshot). Cannot verify from sandbox; HF counts bots/mirrors. Never self-download. User offered rewards at 100 (new harness) and 1k (PC with GPU slot); declined any count inflation.
- Next: write loc release (export 150M to a loadable HF format or custom code + ONNX, card with per-class F1/AP and honest rare-class warning, LICENSE Apache-2.0, NOTICE crediting ESM-2 MIT and UniProt CC BY 4.0, SHA256SUMS, static or free demo, training scripts); push to GitHub via run_push2.bat; run_upload scripts pattern as for NER.
- 2026-11-18: wrote loc release pipeline (loc/scripts/release_loc.py, upload_loc.py, quorum_loc.py, loc/space static demo) and launched run_loc_release.bat (also pushes code to GitHub). Handoff prompt in NEXT_SESSION_PROMPT.md.
- 2026-11-18 15:54: loc release DONE. Model quorumbio/esm2-localization-signalpeptide and static Space quorumbio/protein-localization-demo uploaded; ONNX fp32 591.7 MB (parity 9e-6) and int8 149.9 MB (decision agreement 99.9% on 150 seqs, max prob diff 0.10); Space verified working in browser. GitHub pushed (b4035ba). Pending: relabel Space example button (local edit done, not uploaded).

- 2026-11-18 evening: verified loc release with metadata only (no file downloads). Every LFS sha256 on the hub matches loc/release/SHA256SUMS. Every non-LFS file matches local by git blob id. GitHub HEAD = b4035ba. Both Spaces RUNNING. Did NOT run in-browser inference again: the Space fetches our model file and that may count as a self-download (the earlier session had already tested it).
- 2026-11-18 evening: fix 3b done. Card usage example now has the comment "Example input only: a serum albumin-like sequence, not a verified UniProt entry" (also in release_loc.py template); SHA256SUMS regenerated; model commit c40a455. Space index.html relabeled ("Example: serum albumin-like sequence"), Space commit 3def0a7; live page checked.
- 2026-11-18 evening: picked the next work with the five criteria. Choice: multi-seed runs for CIs on both released models (improves and documents what we already have, $0 on the local GPU, reuses existing scripts; demand and distribution are neutral). Next after that: the weak loc classes. run_seeds.sh: NER v2 seeds 1-4 (~5 min each) and loc 150M seeds 1-2 (~100 min each), same settings as the released runs. Seeds were fixed in advance and every run is reported. Released checkpoints (seed 13) unchanged. Aggregate with docs/seed_summary.py -> docs/seed_summary.json; log in logs/seeds.txt.

## Download metrics
- 2026-11-17: 0 downloads (just released). Check via model_info(...).downloads in DAILY step 2. Never self-download.
- 2026-11-18 ~20:45 CST (HF API model_info, all-time count as reported by HF): biomed-ner-chem-disease 34 downloads, 1 like; esm2-localization-signalpeptide 0 downloads, 0 likes (released ~6h earlier). Spaces biomed-ner-demo and protein-localization-demo: 0 likes each.

## Actions log
- 2026-11-17: prepared data, trained run1 (test F1 0.896), baseline, ONNX, packaged and uploaded model + static Space. accounts created (Proton, HF, GitHub). GPU verified. CUDA torch venv built. Repo layout created.

## Blockers (also in NEEDS_HUMAN.md)
- BC5CDR license is NOT explicitly stated by the BioCreative page or the GitHub mirror ("freely available for the research community"); BigBio lists Public Domain Mark 1.0. Released under Apache-2.0 with that uncertainty disclosed in the card. Legal question for the human (non-blocking).
- huggingface.co blocked from cloud workspace (PC is fine).
- (resolved 2026-11-17) HF token verified from the PC: user quorumbio, role write. Token is in C:\QuorumBio\.env (gitignored); scripts load it from there.

## Next best action
1. When run_seeds.sh finishes (logs/seeds.txt RUN_END): run `venv/Scripts/python.exe docs/seed_summary.py`, add a "Seed variation" section (mean, sd, range, n seeds) to both cards (ner release_pack2 template + loc release_loc.py template), regenerate SHA256SUMS, upload README + SHA256SUMS only, commit + push to GitHub.
2. Then the weak loc classes (Golgi, Lysosome/Vacuole, Peroxisome): try class-weighted or focal BCE, choose on dev macro AP, compare against the seed spread from step 1. Do not call a change an improvement unless it is bigger than the seed sd.
3. Later: a short paper-style write-up in docs/ covering both models.
