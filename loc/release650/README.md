---
license: apache-2.0
library_name: transformers
base_model: facebook/esm2_t33_650M_UR50D
tags: [biology, bioinformatics, protein, protein-language-model, subcellular-localization, signal-peptide, esm2, uniprot, onnx]
datasets: [quorumbio/swissprot-subcellular-localization-signal-peptide]
model-index:
- name: esm2-650m-protein-subcellular-localization-signal-peptide
  results:
  - task:
      type: subcellular-localization
      name: Protein subcellular localization (10 classes, multi-label)
    dataset:
      name: Swiss-Prot homology-partitioned split (test)
      type: quorumbio/swissprot-subcellular-localization-signal-peptide
      split: test
    metrics:
    - type: roc_auc
      value: 0.935
      name: Macro AUROC
    - type: f1
      value: 0.655
      name: Macro F1 (validation-chosen thresholds)
  - task:
      type: signal-peptide-prediction
      name: Signal peptide presence
    dataset:
      name: Swiss-Prot homology-partitioned split (test)
      type: quorumbio/swissprot-subcellular-localization-signal-peptide
      split: test
    metrics:
    - type: roc_auc
      value: 0.992
      name: AUROC
    - type: matthews_correlation
      value: 0.946
      name: MCC (validation-chosen threshold)
---
# Protein subcellular localization and signal peptide prediction (ESM-2 650M)

The larger version of [quorumbio/esm2-protein-subcellular-localization-signal-peptide](https://huggingface.co/quorumbio/esm2-protein-subcellular-localization-signal-peptide): same data, same split, same heads, fine-tuned from `facebook/esm2_t33_650M_UR50D` (650M parameters) instead of the 150M ESM-2. It takes a protein sequence and returns probabilities for 10 subcellular locations (multi-label) and the probability of an N-terminal signal peptide.

**Short version.** Compared with the 150M model on the same test proteins, location macro AUROC is +0.010 [+0.001, +0.019] higher (95% interval from a paired cluster bootstrap), larger than the 150M's seed-to-seed SD of 0.001. Location macro AP (+0.020 [-0.008, +0.052]), rare-class AP (+0.033 [-0.048, +0.128]) and macro F1 (-0.001 [-0.043, +0.042]) are not clearly different. Signal peptide prediction is not better: AUROC -0.002 [-0.008, +0.003], MCC -0.007 [-0.019, +0.004]. Golgi, lysosome/vacuole and peroxisome remain weak (AP under 0.5).

**Which one to use.** For most uses, the 150M model: it has about 4x fewer parameters, a browser demo, and was checked over 3 seeds. This 650M model is a single training run (seed 13). Use it if you want the somewhat better location ranking and can afford the size.

## Use
```python
from huggingface_hub import snapshot_download
import sys; folder = snapshot_download("quorumbio/esm2-650m-protein-subcellular-localization-signal-peptide"); sys.path.insert(0, folder)
from quorum_loc import load, predict
m = load(folder)
predict(m, "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVKLVNELTEFAKTCVADESHAGCEKSLHTLFGDELCKVASLRETYGDMADCCEKQEPERNECFLSHKDDSPDLPKLKPDPNTLCDEFKADEKKFWGKYLYEIARRHPYFYAPELLYYANKYNGVFQECCQAEDKGACLLPKIETMREKVLASSARQRLRCASIQKFGERALKAWSVARLSQKFPKAEFVEVTKLVTDLTKVHKECCHGDLLECADDRADLAKYICDNQDTISSKLKECCDKPLLEKSHCIAEVEKDAIPENLPPLTADFAEDKDVCKNYQEAKDAFLGSFLYEYSRRHPEYAVSVLLRLAKEYEATLEECCAKDDPHACYSTVFDKLKHLVDEPQNLIKQNCDQFEKLGEYGFQNALIVRYTRKVPQVSTPTLVEVSRSLGKVGTRCCTKPESERMPCTEDYLSLILNRLCVLHEKTPVSEKVTKCCTESLVNRRPCFSALTPDETYVPKAFDEKLFTFHADICTLPDTEKQIKKQTALVELLKHKPKATEEQLKTVMENFVAFVDKCCAADDKEACFAVEGPKLVVSTQTALA")
```
The example is a serum albumin-like sequence used as input only, not a verified UniProt entry. Each location result has a probability and a yes/no using a per-class threshold picked on the validation set. `predict` handles sequences over 1000 residues by keeping the first 600 and last 400. We have not benchmarked its speed; expect it to be slower than the 150M model.

ONNX files (`onnx/model.onnx` with its weights in `onnx/model.onnx_data`, fp32 2598.7 MB in total; `onnx/model_quantized.onnx` int8 652.7 MB) take `input_ids` and `attention_mask` and return `loc_logits` (10, apply sigmoid) and `sp_logit` (apply sigmoid). Keep `model.onnx` and `model.onnx_data` in the same folder: the fp32 graph is over the 2 GB protobuf limit, so its weights are stored externally. Checked against PyTorch on 150 test sequences up to 500 residues: max absolute probability difference 1.2e-06 (fp32) and 0.140 (int8); decisions agree with PyTorch on 100.0% (fp32) and 99.8% (int8) of class decisions. The int8 file was only checked on this small sample, not the full test set.

## Results (test set, scored once)
Test set: 2336 proteins from Swiss-Prot, split by sequence similarity (see Data). Checkpoint and thresholds were chosen on the validation set only.

**Signal peptide** (450 positives): AUROC 0.992, MCC 0.946 at threshold 0.75 (150M: AUROC 0.994, MCC 0.956).

**Locations**: macro AUROC 0.935, macro F1 0.655 (150M: 0.926, 0.650). AUROC looks flattering for rare classes, so average precision (AP) and F1 are the numbers to trust. The last two columns are the released 150M model.

| Class | Test positives | AUROC | AP | F1 | AP (150M) | F1 (150M) |
|---|---|---|---|---|---|---|
| Nucleus | 716 | 0.931 | 0.887 | 0.777 | 0.874 | 0.785 |
| Cytoplasm | 842 | 0.904 | 0.818 | 0.798 | 0.820 | 0.779 |
| Extracellular | 351 | 0.984 | 0.926 | 0.871 | 0.892 | 0.861 |
| Mitochondrion | 221 | 0.917 | 0.750 | 0.721 | 0.773 | 0.737 |
| Cell membrane | 295 | 0.930 | 0.746 | 0.684 | 0.728 | 0.693 |
| Endoplasmic reticulum | 192 | 0.928 | 0.692 | 0.636 | 0.619 | 0.630 |
| Plastid | 84 | 0.983 | 0.869 | 0.809 | 0.884 | 0.883 |
| Golgi apparatus | 78 | 0.900 | 0.375 | 0.366 | 0.381 | 0.342 |
| Lysosome/Vacuole | 88 | 0.921 | 0.434 | 0.500 | 0.367 | 0.361 |
| Peroxisome | 35 | 0.954 | 0.446 | 0.383 | 0.405 | 0.432 |

### 650M vs 150M on the same test proteins
We bootstrapped both models' test predictions together: 1475 MMseqs2 test clusters resampled 2000 times, with each model's validation-chosen thresholds fixed. "Share" is the fraction of resamples in which the 650M model is better.

| Metric (650M minus 150M) | Difference | 95% interval | Share |
|---|---|---|---|
| Location macro AUROC | +0.010 | +0.001 to +0.019 | 99% |
| Location macro AP | +0.020 | -0.008 to +0.052 | 92% |
| Location macro F1 | -0.001 | -0.043 to +0.042 | 51% |
| Mean AP, Golgi / Lysosome / Peroxisome | +0.033 | -0.048 to +0.128 | 82% |
| Signal peptide AUROC | -0.002 | -0.008 to +0.003 | 28% |
| Signal peptide MCC | -0.007 | -0.019 to +0.004 | 11% |

An interval that includes 0 means this test set cannot tell the two models apart on that metric. These intervals cover test-set sampling only. The 150M seed spread (3 seeds, reported on its card) is a rough guide to training randomness; this 650M model has one seed, so a difference smaller than that spread may not hold up for another run.

### Test-set uncertainty (650M alone)
| Metric | Point | 95% interval |
|---|---|---|
| Location macro AUROC | 0.935 | 0.926 to 0.944 |
| Location macro AP | 0.694 | 0.666 to 0.728 |
| Location macro F1 | 0.655 | 0.622 to 0.684 |
| Mean AP, Golgi / Lysosome / Peroxisome | 0.418 | 0.342 to 0.513 |
| Signal peptide AUROC | 0.992 | 0.985 to 0.998 |
| Signal peptide MCC | 0.946 | 0.923 to 0.966 |

All classes: `bootstrap_ci.json`; script: `training/bootstrap_loc.py`. These numbers are not directly comparable to published tools such as SignalP or DeepLoc, which use different datasets, splits and label definitions. We have not run those tools on this test set.

## Data
Identical to the 150M model: UniProtKB/Swiss-Prot reviewed entries (CC BY 4.0), retrieved 2026-11-17, with locations backed by experimental evidence mapped to 10 classes, and a signal peptide label from UniProt SIGNAL features (often predicted rather than experimental; see the [150M card](https://huggingface.co/quorumbio/esm2-protein-subcellular-localization-signal-peptide) for the evidence breakdown). Split by MMseqs2 clustering at 30% identity with leakage filtering. The exact split is the dataset [quorumbio/swissprot-subcellular-localization-signal-peptide](https://huggingface.co/datasets/quorumbio/swissprot-subcellular-localization-signal-peptide).

## Training
Same heads and loss as the 150M model: mean-pooled ESM-2 features into a 10-way location head, a signal peptide head that also sees the mean of the first 30 residues, binary cross-entropy with class weights (max 10), bf16, AdamW with warmup and linear decay, lr 5e-05.

What differs, and why: full fine-tuning of 650M parameters does not fit in the 12 GB of our single RTX 3060. We froze the embeddings and the bottom 16 of 33 encoder layers and trained the top 17 layers plus the heads, with gradient checkpointing, batch size 2 and gradient accumulation 4 (effective batch 8; the 150M used batch 8 without accumulation), for 3 epochs (the 150M used 2). These choices were fixed before training from memory tests, not tuned on validation. The epoch with the best validation score (mean of location macro AUROC and signal peptide AUROC) was kept. Training took about 7.1 hours.

| Epoch | Dev macro AUROC | Dev SP AUROC | Dev macro AP | Dev rare AP | Selection score |
|---|---|---|---|---|---|
| 1 | 0.901 | 0.996 | 0.673 | 0.418 | 0.9482 |
| 2 | 0.908 | 0.996 | 0.698 | 0.492 | 0.9521 (kept) |
| 3 | 0.902 | 0.996 | 0.700 | 0.499 | 0.9489 |

The 150M model's best selection score was 0.9478. `training/` has the scripts and `provenance.json`.

## Limitations
- One training seed. The 150M model's per-class rare F1 varied a lot across seeds, so expect the same here.
- Still weak on Golgi, lysosome/vacuole and peroxisome; check their AP above before relying on them.
- Labels come from curated annotations, which are incomplete and sometimes inconsistent; only 10 coarse locations; isoform-specific locations are merged; no signal peptide cleavage site.
- Mostly well-studied proteins, so expect lower accuracy on proteins unlike Swiss-Prot.
- Research use only.

## Files
`SHA256SUMS` lists checksums for every file.
