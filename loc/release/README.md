---
license: apache-2.0
library_name: transformers
base_model: facebook/esm2_t30_150M_UR50D
tags: [biology, protein, subcellular-localization, signal-peptide, esm2, uniprot]
---
# Protein subcellular localization and signal peptide prediction (ESM-2 150M)

Takes a protein sequence and returns (1) probabilities for 10 subcellular locations (a protein can have more than one) and (2) the probability that it has an N-terminal signal peptide. Fine-tuned from `facebook/esm2_t30_150M_UR50D` on UniProtKB/Swiss-Prot with a homology-reduced split. 150M parameters.

**Short version.** Signal peptide detection is strong (AUROC 0.994, MCC 0.956). Location prediction is decent for common compartments (nucleus, cytoplasm, extracellular, plastid) and **weak for Golgi, lysosome/vacuole and peroxisome**: do not rely on those three.

## Use
```python
from huggingface_hub import snapshot_download
import sys; folder = snapshot_download("quorumbio/esm2-localization-signalpeptide"); sys.path.insert(0, folder)
from quorum_loc import load, predict
m = load(folder)
# Example input only: a serum albumin-like sequence, not a verified UniProt entry. Use your own sequence.
predict(m, "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVKLVNELTEFAKTCVADESHAGCEKSLHTLFGDELCKVASLRETYGDMADCCEKQEPERNECFLSHKDDSPDLPKLKPDPNTLCDEFKADEKKFWGKYLYEIARRHPYFYAPELLYYANKYNGVFQECCQAEDKGACLLPKIETMREKVLASSARQRLRCASIQKFGERALKAWSVARLSQKFPKAEFVEVTKLVTDLTKVHKECCHGDLLECADDRADLAKYICDNQDTISSKLKECCDKPLLEKSHCIAEVEKDAIPENLPPLTADFAEDKDVCKNYQEAKDAFLGSFLYEYSRRHPEYAVSVLLRLAKEYEATLEECCAKDDPHACYSTVFDKLKHLVDEPQNLIKQNCDQFEKLGEYGFQNALIVRYTRKVPQVSTPTLVEVSRSLGKVGTRCCTKPESERMPCTEDYLSLILNRLCVLHEKTPVSEKVTKCCTESLVNRRPCFSALTPDETYVPKAFDEKLFTFHADICTLPDTEKQIKKQTALVELLKHKPKATEEQLKTVMENFVAFVDKCCAADDKEACFAVEGPKLVVSTQTALA")
```
Each location result has a probability and a yes/no using a per-class threshold picked on the validation set. `predict` handles sequences over 1000 residues by keeping the first 600 and last 400. ONNX files (`onnx/model.onnx` fp32 591.7 MB, `onnx/model_quantized.onnx` int8 149.9 MB) take `input_ids` and `attention_mask` and return `loc_logits` (10, apply sigmoid) and `sp_logit` (apply sigmoid). Checked against PyTorch on 150 test sequences up to 500 residues: max absolute probability difference 9.2e-06 (fp32) and 0.100 (int8); decisions agree with PyTorch on 100.0% (fp32) and 99.9% (int8) of class decisions. The int8 file is smaller and faster but not identical, and its accuracy was only checked on this small sample, not the full test set.

## Results (test set, scored once)
Test set: 2336 proteins from Swiss-Prot, split by sequence similarity (see Data). Single training run, single seed, no confidence intervals. Checkpoint and thresholds were chosen on the validation set only.

**Signal peptide** (450 positives among 2336 test proteins): AUROC 0.994, MCC 0.956 at threshold 0.9.

**Locations** (macro AUROC 0.926, macro F1 0.650). AUROC looks flattering when a class is rare, so average precision (AP) and F1 are the numbers to trust. Last column is the smaller ESM-2 35M version of the same recipe, for reference.

| Class | Test positives | AUROC | AP | F1 | F1 (35M) |
|---|---|---|---|---|---|
| Nucleus | 716 | 0.929 | 0.874 | 0.785 | 0.769 |
| Cytoplasm | 842 | 0.899 | 0.820 | 0.779 | 0.763 |
| Extracellular | 351 | 0.977 | 0.892 | 0.861 | 0.840 |
| Mitochondrion | 221 | 0.931 | 0.773 | 0.737 | 0.701 |
| Cell membrane | 295 | 0.924 | 0.728 | 0.693 | 0.680 |
| Endoplasmic reticulum | 192 | 0.932 | 0.619 | 0.630 | 0.521 |
| Plastid | 84 | 0.980 | 0.884 | 0.883 | 0.859 |
| Golgi apparatus | 78 | 0.885 | 0.381 | 0.342 | 0.311 |
| Lysosome/Vacuole | 88 | 0.879 | 0.367 | 0.361 | 0.299 |
| Peroxisome | 35 | 0.922 | 0.405 | 0.432 | 0.296 |

Rare-class AP is far above random guessing (for example peroxisome is about 1.5% of the test set) but low in absolute terms.

These numbers are not directly comparable to published tools such as SignalP or DeepLoc, which use different datasets, splits and label definitions. We have not run those tools on this test set.

## Data
UniProtKB/Swiss-Prot reviewed entries (CC BY 4.0), retrieved 2026-11-17. Kept proteins of length 30 to 4000 with standard amino acids and at least one subcellular location backed by experimental evidence (ECO:0000269), mapped by name to 10 classes. Signal peptide label: the entry has a UniProt SIGNAL feature of any evidence type, so it can include predicted annotations. Clustering: MMseqs2 at 30% identity (coverage 0.8), clusters assigned 80/10/10 to train/dev/test, then dev and test proteins with any hit at 30% or more identity to training proteins were removed. Final sizes: train 30134, dev 2457, test 2336; signal peptide positives 5636, 385, 450. Multiple isoform-specific locations are merged per protein. The exact split is published as the dataset [quorumbio/swissprot-localization-signalpeptide](https://huggingface.co/datasets/quorumbio/swissprot-localization-signalpeptide).

## Training
`training/` has the scripts and `provenance.json` (checksum of the downloaded data). Mean-pooled ESM-2 features feed a 10-way location head; the signal peptide head also sees the mean of the first 30 residues. Binary cross-entropy with class weights (max 10) for rare locations, bf16, gradient checkpointing, AdamW with warmup and linear decay, batch size 8, lr 5e-05, 2 epochs (best dev epoch kept), one RTX 3060, about 99 minutes.

## Limitations
- Weak on Golgi, lysosome/vacuole and peroxisome; mediocre on ER and cell membrane.
- Labels come from curated annotations, which are incomplete and sometimes inconsistent; proteins with unknown or contested location are not represented.
- Only 10 coarse locations; no sub-compartments, no membrane topology, no signal peptide cleavage site.
- Isoform and condition-specific locations are merged.
- Mostly well-studied proteins, so expect lower accuracy on proteins unlike Swiss-Prot.
- Research use only. One run, one seed, no confidence intervals.

## Files
`SHA256SUMS` lists checksums for every file.
