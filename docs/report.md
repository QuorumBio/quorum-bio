# Two small open models and a homology-partitioned dataset for biomedical text and protein annotation

Quorum Bio, technical report, draft of 2026-11-22.
Code: https://github.com/QuorumBio/quorum-bio. Models and data: https://huggingface.co/quorumbio.

## Abstract

We release two fine-tuned models and one dataset, all under permissive licenses, with training scripts, checksums and ONNX exports. (1) `quorumbio/biomedical-ner-chemical-disease` is a BiomedBERT-base token classifier for Chemical and Disease mentions, trained on BC5CDR and NCBI-disease. Over 5 training seeds its test entity F1 is 0.889 ± 0.002 on BC5CDR and 0.872 ± 0.002 on NCBI-disease. (2) `quorumbio/esm2-protein-subcellular-localization-signal-peptide` is an ESM-2 150M model that predicts 10 subcellular locations (multi-label) and the presence of an N-terminal signal peptide. On a test set with no protein at 30% or more identity to any training protein, over 3 seeds, it reaches location macro AUROC 0.927 ± 0.001, macro F1 0.645 ± 0.009, and signal peptide AUROC 0.993 ± 0.001 with MCC 0.956. It is weak on three rare compartments (Golgi, lysosome/vacuole, peroxisome; AP 0.36 to 0.38), and a pre-registered screen of three fixes (stronger class weights, focal loss, attention pooling) did not help. (3) `quorumbio/swissprot-subcellular-localization-signal-peptide` is the exact train/validation/test split used for (2), 34,927 Swiss-Prot proteins with experimentally supported locations, published under CC BY 4.0. Everything was trained on one consumer GPU (RTX 3060, 12 GB) at no compute cost. None of our numbers are directly comparable to published benchmark results, because splits, tokenization and label definitions differ. Section 5 explains this.

## 1. Motivation and scope

Many small biomedical models on the Hugging Face Hub have unclear training data, no held-out evaluation, or licenses that do not allow commercial use. Our aim is narrower than state of the art. We want models that are permissively licensed, cheap to run (ONNX, int8), honestly evaluated (checkpoints chosen on validation data, test sets scored once, seed variation reported), and reproducible from public data with the published scripts.

Rules we followed for both models: public data only, with each dataset license recorded before training; no non-commercial data; model selection and thresholds on validation data only; every run reported, including ones that did worse.

## 2. Biomedical NER: Chemical and Disease

### 2.1 Data
- **BC5CDR** (Li et al., 2016): 1,500 PubMed abstracts (500 train / 500 dev / 500 test) with Chemical and Disease mentions. The distributors give no explicit license text ("freely available for the research community"). BigBio lists it as Public Domain Mark 1.0. We disclose this uncertainty in the model card.
- **NCBI-disease** (Doğan et al., 2014): 793 abstracts (593 / 100 / 100), Disease only, under NCBI's public domain notice.
- We checked JNLPBA/GENIA (CC BY 3.0) and did not use it. We skipped BC2GM because we could not find a license.

Character offsets were converted to word-level BIO tags with a simple regex tokenizer, and abstracts were split into sentences. There were 0 offset mismatches in BC5CDR and 1 in the NCBI-disease training set. Source archives are SHA256-hashed in `ner/data/provenance.json`.

### 2.2 Method
Base model: `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` (MIT, 108.9M parameters). Token classification with 5 labels (B/I for Chemical and Disease, plus O). Settings: 5 epochs, batch size 16, learning rate 5e-5, max length 256, bf16 autocast, AdamW with linear warmup and decay. Each run takes about 5 minutes on the RTX 3060. The checkpoint is the epoch with the best validation entity F1 (for v2, the mean over the two corpora).

We trained three variants:
- **run1 / v1:** BC5CDR only.
- **run2a / v2 (released):** BC5CDR and NCBI-disease merged naively. Chemicals in NCBI sentences are therefore labeled O.
- **run2b:** like run2a, but O labels are ignored in NCBI sentences, so unlabeled chemicals are not penalized.

### 2.3 Results
Scores are entity-level seqeval P/R/F1 on regex-tokenized words, with test sets scored once.

| Model | BC5CDR test F1 | NCBI-disease test F1 (Disease only) |
|---|---|---|
| run1 (BC5CDR only, 1 seed) | 0.896 | 0.735 (out of domain) |
| run2a, released seed 13 | 0.889 | 0.878 |
| run2a, 5 seeds (13, 1, 2, 3, 4), mean ± SD | 0.889 ± 0.002 (range 0.887 to 0.891) | 0.872 ± 0.002 (range 0.870 to 0.875) |
| run2b (ignore O on NCBI) | 0.875 | 0.427 |

The 5-seed numbers come from the training script's evaluation path. The seed-13 row comes from the `transformers` pipeline path. The two differ slightly on NCBI-disease (0.870 vs 0.878 for the same checkpoint). BC5CDR per-type F1 for v2 is 0.922 for Chemical and 0.850 for Disease.

Observations:
- Adding NCBI-disease costs a small but consistent amount on BC5CDR. v1 scored 0.896 on one seed, and v2 averages 0.889 with SD 0.002, so the drop is probably real. In exchange, NCBI-disease F1 rises by about 0.14.
- run2b's precision on NCBI-disease collapses (F1 0.427). Without O labels, the NCBI sentences give the model no negative signal. We discarded it.
- The released seed is typical, not a lucky draw.
- Test-set uncertainty, from 2,000 bootstrap resamples of whole abstracts with 95% intervals: BC5CDR F1 is 0.889 (0.879 to 0.898) and NCBI-disease F1 is 0.878 (0.851 to 0.904). Both are several times wider than the seed SD. We also ran a paired bootstrap of v2 against v1 on the same resampled BC5CDR abstracts. It gives −0.007 F1 (−0.012 to −0.002), and v2 is lower in 99.5% of resamples, so the drop holds up (`ner/scripts/bootstrap_ner.py`).

### 2.4 Comparison with an existing model
We ran `d4data/biomedical-ner-all` (66.4M parameters, 84 clinical entity types) through the same pipeline and scorer, mapping its labels to ours by hand. Strict map: BC5CDR F1 0.479, NCBI-disease 0.377. Generous map (Sign_symptom also counted as Disease): 0.533 and 0.329. This is an in-domain vs out-of-domain comparison. Our model was trained on these corpora, and d4data's was not. It says nothing about general quality. d4data is also smaller, ran faster in our GPU test (13.7 s vs 17.2 s for 4,821 sentences), and covers many more entity types.

### 2.5 Deployment
We provide ONNX fp32 (435.8 MB) and dynamic int8 (109.7 MB) exports. fp32 vs PyTorch: max absolute logit difference 3.4e-5, token argmax agreement 1.0 on 300 sentences. On CPU, int8 runs at 13.6 ms per sentence vs 24.5 ms for fp32, and F1 changes from 0.889 to 0.886 (BC5CDR) and from 0.878 to 0.879 (NCBI-disease). A static Hugging Face Space (`quorumbio/biomed-ner-demo`) runs the int8 model in the browser with transformers.js.

## 3. Subcellular localization and signal peptide

### 3.1 Dataset
Source: UniProtKB/Swiss-Prot reviewed entries, release 2026_04 (576,073 entries), CC BY 4.0. Steps:
1. Keep sequences of length 30 to 4,000 that use only the 20 standard amino acids.
2. Keep only subcellular locations with evidence code ECO:0000269 (experimental), mapped by name prefix to 10 classes: nucleus, cytoplasm, extracellular, mitochondrion, cell membrane, ER, plastid, Golgi, lysosome/vacuole, peroxisome. Drop entries with no mapped experimental location. This leaves 37,733 proteins.
3. Signal peptide label: presence of a UniProt SIGNAL feature, at any evidence level. Most are predicted: 74% of positives carry ECO:0000255 (sequence-model evidence, typically from tools such as SignalP) and 18% are experimental (ECO:0000269).
4. Cluster with MMseqs2 (30% identity, coverage 0.8), giving 20,269 clusters. Assign whole clusters 80/10/10 to train/dev/test.
5. Search dev and test against train at 30% identity and remove every hit: 1,328 dev and 1,478 test proteins removed.

Final split: train 30,134, dev 2,457, test 2,336. Signal peptide positives: 5,636 / 385 / 450. No accession or cluster appears in more than one split. Three classes are rare in test: Golgi has 78 positives, lysosome/vacuole 88 and peroxisome 35 (1.5% of test). We publish the split as `quorumbio/swissprot-subcellular-localization-signal-peptide`, with accession, sequence, organism, multi-hot labels, signal peptide end position and cluster ID, so others can use the same partition or grouped cross-validation.

An important caveat for users: **a missing location is not a negative.** It means there is no experimental annotation for that compartment, not that the protein is absent from it. Our negatives are therefore noisy, most of all for multi-compartment proteins.

### 3.2 Method
The model has three parts:
- Backbone: ESM-2 (Lin et al., 2023; MIT license), either 35M (`esm2_t12_35M_UR50D`) or 150M (`esm2_t30_150M_UR50D`), fully fine-tuned.
- Location head: a 10-way multi-label head on mean-pooled residue embeddings.
- Signal peptide head: sees the overall mean plus the mean of the first 30 residues.

Training uses binary cross-entropy with per-class positive weights (capped at 10), bf16, gradient checkpointing, AdamW with warmup and linear decay, batch size 8, learning rate 5e-5 and 2 epochs, keeping the best dev epoch. Per-class decision thresholds are tuned on dev. Sequences over 1,000 residues keep the first 600 and last 400. One 150M run takes about 100 minutes on the RTX 3060.

### 3.3 Results
On test, the 150M model beats the 35M model on every class, by the most on ER and peroxisome. 35M: location macro AUROC 0.906, macro F1 0.604. 150M: 0.926 and 0.650. Signal peptide is the same for both: AUROC 0.994, MCC 0.956.

Per-class test results for the released 150M checkpoint (seed 13):

| Class | Test positives | AUROC | AP | F1 |
|---|---|---|---|---|
| Nucleus | 716 | 0.929 | 0.874 | 0.785 |
| Cytoplasm | 842 | 0.899 | 0.820 | 0.779 |
| Extracellular | 351 | 0.977 | 0.892 | 0.861 |
| Mitochondrion | 221 | 0.931 | 0.773 | 0.737 |
| Cell membrane | 295 | 0.924 | 0.728 | 0.693 |
| Endoplasmic reticulum | 192 | 0.932 | 0.619 | 0.630 |
| Plastid | 84 | 0.980 | 0.884 | 0.883 |
| Golgi apparatus | 78 | 0.885 | 0.381 | 0.342 |
| Lysosome/Vacuole | 88 | 0.879 | 0.367 | 0.361 |
| Peroxisome | 35 | 0.922 | 0.405 | 0.432 |

Seed variation (150M, seeds 13, 1, 2, test, mean ± SD):

| Metric | Mean ± SD | Range |
|---|---|---|
| Location macro AUROC | 0.927 ± 0.001 | 0.926 to 0.927 |
| Location macro F1 | 0.645 ± 0.009 | 0.635 to 0.650 |
| Signal peptide AUROC | 0.993 ± 0.001 | 0.992 to 0.994 |
| Signal peptide MCC | 0.956 ± 0.000 | 0.956 to 0.956 |
| AP Golgi / Lys-Vac / Peroxisome | 0.375 ± 0.011 / 0.356 ± 0.010 / 0.356 ± 0.054 | |
| F1 Golgi / Lys-Vac / Peroxisome | 0.376 ± 0.030 / 0.354 ± 0.010 / 0.337 ± 0.139 | Peroxisome 0.178 to 0.432 |

**Test-set sampling uncertainty.** We bootstrapped the released checkpoint's test predictions. We resampled the 1,475 MMseqs2 test clusters rather than single proteins, because similar proteins within a cluster are not independent. We used 2,000 resamples with the dev-chosen thresholds held fixed (`loc/scripts/bootstrap_loc.py`, `bootstrap_ci.json`). Re-running inference reproduced the published test numbers to 4 decimal places. The resulting 95% intervals:

| Metric | Point | 95% CI |
|---|---|---|
| Location macro AUROC | 0.926 | 0.914 to 0.935 |
| Location macro AP | 0.674 | 0.642 to 0.708 |
| Location macro F1 | 0.650 | 0.614 to 0.680 |
| Rare-class AP (mean of 3) | 0.384 | 0.299 to 0.477 |
| AP Golgi / Lys-Vac / Peroxisome | 0.381 / 0.367 / 0.405 | 0.24 to 0.51 / 0.22 to 0.52 / 0.24 to 0.59 |
| Signal peptide AUROC | 0.994 | 0.988 to 0.998 |
| Signal peptide MCC | 0.956 | 0.935 to 0.972 |

For the rare classes, the uncertainty from the small test set (about ±0.1 to 0.2 AP) is roughly ten times larger than the seed-to-seed SD (about 0.01 to 0.05). Treat the per-class rare numbers as rough magnitudes. Comparisons between variants should be paired on the same examples, as in the dev screen in 3.4, and not compared across separate intervals.

**Signal peptide labels by evidence.** On the 77 test signal peptides with experimental evidence, recall is 0.948 (95% Wilson interval 0.87 to 0.98) and AUROC is 0.995 against all test negatives. On the 340 sequence-analysis positives the values are 0.950 and 0.995. So the high signal peptide score does not merely reflect agreement with predicted labels, though the experimental subset is small (`loc/scripts/sp_evidence.py`).

Two points matter for reading these tables.
- AUROC looks high for the rare classes (0.88 to 0.92), but AP is the more informative metric there.
- Rare-class F1 depends on a threshold tuned on few dev positives (62 for peroxisome), so it swings widely between seeds. The released checkpoint's peroxisome F1 of 0.432 is the best of the 3 seeds, and the typical value is about 0.34. The model card says so.

### 3.4 Rare-class experiments (null result)
To try to improve the three rare classes, we fixed a screen in advance on the 35M model (2 epochs, dev only, test not scored). It has four variants, each with seeds 13 and 1:
- A: baseline, positive weight capped at 10.
- B: positive weight cap raised to 100.
- C: focal loss.
- D: attention pooling in place of mean pooling.

The selection rule was also fixed before any result was seen. The winner is the variant with the best mean dev rare-class AP, and it advances only if it beats A by at least 0.02 rare AP while losing no more than 0.01 dev macro AP. A winner would be retrained on 150M with seeds 13, 1 and 2 and compared against the 3-seed baseline above. We would call it better only if the rare-class AP gain exceeds the seed SD.

**Result (2026-11-22): null. No variant beat the baseline.** Dev set, ESM-2 35M, 2 epochs, mean of seeds 13 and 1:

| Variant | Dev rare-class AP (mean of Golgi, Lys/Vac, Peroxisome) | Per seed (13, 1) | Dev macro AP |
|---|---|---|---|
| A baseline (pos-weight cap 10) | **0.339** | 0.339, 0.339 | **0.622** |
| B pos-weight cap 100 | 0.312 | 0.316, 0.308 | 0.607 |
| C focal loss | 0.221 | 0.243, 0.199 | 0.590 |
| D attention pooling | 0.255 | 0.305, 0.204 | 0.596 |

All three variants scored lower than the baseline on both rare-class AP and macro AP, so under the pre-set rule nothing advanced to 150M and the test set was not touched. The released model stays as it is. Caveats: we ran each variant at a single setting (for example one focal-loss gamma and one learning rate) on the small backbone for 2 epochs. The screen therefore rules out these variants as drop-in changes to our recipe, not the ideas in general. Attention pooling also varied widely between seeds (0.305 vs 0.204) and might need longer training. Raw results: `loc/models/screen/*/dev_results.json` and `loc/models/screen/pick.json`.

### 3.5 Deployment
- ONNX fp32 (591.7 MB): max probability difference vs PyTorch 9.2e-6 on 150 test sequences, with 100% decision agreement.
- ONNX int8 (149.9 MB): max probability difference 0.10 and 99.9% decision agreement. This was checked only on that sample, not on the full test set.
- A static Space (`quorumbio/protein-localization-demo`) runs the int8 model in the browser.
- `quorum_loc.py` gives a two-line Python loader.

## 4. Release practice
Each repository ships the following:
- `LICENSE`: Apache-2.0 for the models, CC BY 4.0 for the dataset.
- `NOTICE`: credits the base-model and data licenses.
- `SHA256SUMS` for every file.
- `training/`: the scripts, plus `provenance.json` with source URLs and hashes.
- JSON files with every reported number.

We verified the hub copies against local checksums using metadata only. We do not download our own artifacts, because that would inflate download counts. Card claims are limited to what we measured.

## 5. Limitations
- **No benchmark parity.** Our BC5CDR and NCBI-disease scores use our own tokenization and sentence splitting, and our localization split differs from DeepLoc 2.0 (Thumuluri et al., 2022) and SignalP 6.0 (Teufel et al., 2022) in data, partitions and labels. We have not run those tools on our test set, so we make no claim about how we rank against them.
- **Uncertainty is partial.** Seed SDs come from 5 (NER) and 3 (localization) seeds and capture training randomness only. We also give test-set bootstrap intervals: by abstract for NER (2.3) and by sequence cluster for localization (3.3). They are wider than the seed SDs, most of all for NCBI-disease (100 abstracts) and the rare localization classes.
- **Label noise.** NCBI sentences carry unlabeled chemicals (labeled O). Localization negatives are incomplete annotations. Signal peptide labels include sequence-predicted SIGNAL features.
- **Coverage.** NER covers English PubMed abstracts and two entity types, with no normalization to identifiers. Localization covers 10 coarse compartments, with no topology, cleavage site or isoform specificity, and is biased toward well-studied organisms.
- **Licensing.** BC5CDR has no explicit license text from its distributors (see 2.1).
- Neither model is validated for clinical, diagnostic or safety-critical use.

## 6. Reproducibility
All steps run on one RTX 3060 (12 GB) under Windows with PyTorch 2.14 / CUDA 13.0. Compute cost: $0. Training a NER model takes about 5 minutes, and training a 150M localization model takes about 100 minutes. Scripts:
- NER: `ner/scripts/`, including `prepare_data.py` and `train2.py`.
- Localization: `loc/scripts/`, including `prepare_uniprot.py`, `train_loc.py` and `build_dataset.py`.
- Seed aggregation: `docs/seed_summary.py`.

MMseqs2 on Windows: the `easy-*` workflows segfaulted for us. Use `createdb`/`cluster`/`createtsv`/`search`/`convertalis` instead.

## AI use
Data processing, model training, evaluation and the first draft of this report were carried out by an AI agent (Claude, Anthropic) working in Claude Code on the lab founder's computer. The founder set up the environment, approved all compute, made release decisions and reviewed the results. All numbers come from the scripts and result files in the repository.

## References
- Doğan RI, Leaman R, Lu Z. NCBI disease corpus: a resource for disease name recognition and concept normalization. J Biomed Inform 47:1-10 (2014).
- Gu Y, et al. Domain-specific language model pretraining for biomedical natural language processing. ACM Trans Comput Healthcare 3(1):2 (2021).
- Li J, et al. BioCreative V CDR task corpus: a resource for chemical disease relation extraction. Database baw068 (2016).
- Lin TY, et al. Focal loss for dense object detection. ICCV (2017).
- Lin Z, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. Science 379:1123-1130 (2023).
- Steinegger M, Söding J. MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. Nat Biotechnol 35:1026-1028 (2017).
- Teufel F, et al. SignalP 6.0 predicts all five types of signal peptides using protein language models. Nat Biotechnol 40:1023-1025 (2022).
- Thumuluri V, et al. DeepLoc 2.0: multi-label subcellular localization prediction using protein language models. Nucleic Acids Res 50(W1):W228-W234 (2022).
- The UniProt Consortium. UniProt: the Universal Protein Knowledgebase in 2025. Nucleic Acids Res 53(D1):D609-D617 (2025).
