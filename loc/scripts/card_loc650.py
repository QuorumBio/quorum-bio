"""Write loc/release650/README.md and SHA256SUMS for the ESM-2 650M localization model.
Every number comes from JSON written by train_loc.py (train_results_650m.json), release_loc650.py (onnx_results.json)
and bootstrap_loc.py (bootstrap_ci.json, with the paired difference vs the 150M release). Run after those three."""
import json, hashlib
from pathlib import Path
root = Path(__file__).resolve().parents[1]; rel = root / "release650"
REPO = "quorumbio/esm2-650m-protein-subcellular-localization-signal-peptide"; SMALL = "quorumbio/esm2-protein-subcellular-localization-signal-peptide"
DS = "quorumbio/swissprot-subcellular-localization-signal-peptide"
R = json.load(open(rel / "train_results_650m.json")); R150 = json.load(open(rel / "train_results_150m.json"))
B = json.load(open(rel / "bootstrap_ci.json")); O = json.load(open(rel / "onnx_results.json"))
T, T150 = R["test"], R150["test"]; A = R["args"]; H = R["val_history"]
best = max(H, key=lambda m: m["select"]); best150 = max(R150["val_history"], key=lambda m: m["select"])
assert best["select"] > best150["select"], "650M is not better than 150M on validation; record a negative result instead of releasing"
pk = [k for k in B if k.startswith("paired_difference_vs_")]; PD = B[pk[0]] if pk else None
f3 = lambda x: f"{x:.3f}"
SEED150 = json.load(open(root / "release" / "seed_results.json"))
def iv(k): v = PD[k]; return f"{v['point']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]"
better = [k for k in PD if PD[k]["lo"] > 0]; worse = [k for k in PD if PD[k]["hi"] < 0]
short = (f"Compared with the 150M model on the same test proteins, location macro AUROC is {iv('macro_auroc')} higher (95% interval from a paired cluster bootstrap), "
         f"larger than the 150M's seed-to-seed SD of {SEED150['macro_auroc_loc']['std']:.3f}. "
         f"Location macro AP ({iv('macro_ap')}), rare-class AP ({iv('rare_ap')}) and macro F1 ({iv('macro_f1')}) are not clearly different. "
         f"Signal peptide prediction is not better: AUROC {iv('sp_auroc')}, MCC {iv('sp_mcc')}. Golgi, lysosome/vacuole and peroxisome remain weak (AP under 0.5).")
assert better == ["macro_auroc"] and not worse, (better, worse)  # the sentence above is written for this outcome
def ci(k): c = B["ci"][k]; return f"{c['lo']:.3f} to {c['hi']:.3f}"
sp, sp150 = T["signal_peptide"], T150["signal_peptide"]
rows = "\n".join(f"| {c} | {v['n_pos']} | {f3(v['auroc'])} | {f3(v['avg_precision'])} | {f3(v['f1_at_dev_threshold'])} | {f3(T150['per_class'][c]['avg_precision'])} | {f3(T150['per_class'][c]['f1_at_dev_threshold'])} |" for c, v in T["per_class"].items())
hist = "\n".join(f"| {m['epoch']} | {f3(m['dev_macro_auroc_loc'])} | {f3(m['dev_sp_auroc'])} | {f3(m['dev_macro_ap_loc'])} | {f3(m['dev_rare_ap'])} | {m['select']:.4f}{' (kept)' if m is best else ''} |" for m in H)
names = {"macro_auroc": "Location macro AUROC", "macro_ap": "Location macro AP", "macro_f1": "Location macro F1", "rare_ap": "Mean AP, Golgi / Lysosome / Peroxisome", "sp_auroc": "Signal peptide AUROC", "sp_mcc": "Signal peptide MCC"}
pd_rows = "\n".join(f"| {names[k]} | {v['point']:+.3f} | {v['lo']:+.3f} to {v['hi']:+.3f} | {100 * v['share_this_better']:.0f}% |" for k, v in PD.items()) if PD else ""
ci_rows = "\n".join(f"| {names[k]} | {f3(B['ci'][k]['point'])} | {ci(k)} |" for k in names if k in B["ci"])
minutes = R["train_minutes"]; fz = A.get("freeze_layers", 0); eff = A["bs"] * A.get("accum", 1)
onnx_txt = (f"ONNX files (`onnx/model.onnx` with its weights in `onnx/model.onnx_data`, fp32 {O['files_mb']['fp32']} MB in total; `onnx/model_quantized.onnx` int8 {O['files_mb']['int8']} MB) take `input_ids` and `attention_mask` and return `loc_logits` (10, apply sigmoid) and `sp_logit` (apply sigmoid). "
            f"Keep `model.onnx` and `model.onnx_data` in the same folder: the fp32 graph is over the 2 GB protobuf limit, so its weights are stored externally. "
            f"Checked against PyTorch on {O['sequences_checked']} test sequences up to 500 residues: max absolute probability difference {O['max_abs_probability_diff_vs_torch']['fp32']:.1e} (fp32) and {O['max_abs_probability_diff_vs_torch']['int8']:.3f} (int8); "
            f"decisions agree with PyTorch on {100 * O['decision_agreement_vs_torch']['fp32']:.1f}% (fp32) and {100 * O['decision_agreement_vs_torch']['int8']:.1f}% (int8) of class decisions. The int8 file was only checked on this small sample, not the full test set.") if O.get("status") == "ok" else "ONNX export failed for this model, so only PyTorch weights are provided."
card = f"""---
license: apache-2.0
library_name: transformers
base_model: {R['base']}
tags: [biology, bioinformatics, protein, protein-language-model, subcellular-localization, signal-peptide, esm2, uniprot, onnx]
datasets: [{DS}]
model-index:
- name: esm2-650m-protein-subcellular-localization-signal-peptide
  results:
  - task:
      type: subcellular-localization
      name: Protein subcellular localization (10 classes, multi-label)
    dataset:
      name: Swiss-Prot homology-partitioned split (test)
      type: {DS}
      split: test
    metrics:
    - type: roc_auc
      value: {T['macro_auroc_loc']:.3f}
      name: Macro AUROC
    - type: f1
      value: {T['macro_f1_loc']:.3f}
      name: Macro F1 (validation-chosen thresholds)
  - task:
      type: signal-peptide-prediction
      name: Signal peptide presence
    dataset:
      name: Swiss-Prot homology-partitioned split (test)
      type: {DS}
      split: test
    metrics:
    - type: roc_auc
      value: {sp['auroc']:.3f}
      name: AUROC
    - type: matthews_correlation
      value: {sp['mcc_at_dev_threshold']:.3f}
      name: MCC (validation-chosen threshold)
---
# Protein subcellular localization and signal peptide prediction (ESM-2 650M)

The larger version of [{SMALL}](https://huggingface.co/{SMALL}): same data, same split, same heads, fine-tuned from `{R['base']}` (650M parameters) instead of the 150M ESM-2. It takes a protein sequence and returns probabilities for 10 subcellular locations (multi-label) and the probability of an N-terminal signal peptide.

**Short version.** {short}

**Which one to use.** For most uses, the 150M model: it has about 4x fewer parameters, a browser demo, and was checked over 3 seeds. This 650M model is a single training run (seed {A['seed']}). Use it if you want the somewhat better location ranking and can afford the size.

## Use
```python
from huggingface_hub import snapshot_download
import sys; folder = snapshot_download("{REPO}"); sys.path.insert(0, folder)
from quorum_loc import load, predict
m = load(folder)
predict(m, "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVKLVNELTEFAKTCVADESHAGCEKSLHTLFGDELCKVASLRETYGDMADCCEKQEPERNECFLSHKDDSPDLPKLKPDPNTLCDEFKADEKKFWGKYLYEIARRHPYFYAPELLYYANKYNGVFQECCQAEDKGACLLPKIETMREKVLASSARQRLRCASIQKFGERALKAWSVARLSQKFPKAEFVEVTKLVTDLTKVHKECCHGDLLECADDRADLAKYICDNQDTISSKLKECCDKPLLEKSHCIAEVEKDAIPENLPPLTADFAEDKDVCKNYQEAKDAFLGSFLYEYSRRHPEYAVSVLLRLAKEYEATLEECCAKDDPHACYSTVFDKLKHLVDEPQNLIKQNCDQFEKLGEYGFQNALIVRYTRKVPQVSTPTLVEVSRSLGKVGTRCCTKPESERMPCTEDYLSLILNRLCVLHEKTPVSEKVTKCCTESLVNRRPCFSALTPDETYVPKAFDEKLFTFHADICTLPDTEKQIKKQTALVELLKHKPKATEEQLKTVMENFVAFVDKCCAADDKEACFAVEGPKLVVSTQTALA")
```
The example is a serum albumin-like sequence used as input only, not a verified UniProt entry. Each location result has a probability and a yes/no using a per-class threshold picked on the validation set. `predict` handles sequences over 1000 residues by keeping the first 600 and last 400. We have not benchmarked its speed; expect it to be slower than the 150M model.

{onnx_txt}

## Results (test set, scored once)
Test set: {T['n']} proteins from Swiss-Prot, split by sequence similarity (see Data). Checkpoint and thresholds were chosen on the validation set only.

**Signal peptide** ({sp['n_pos']} positives): AUROC {sp['auroc']:.3f}, MCC {sp['mcc_at_dev_threshold']:.3f} at threshold {sp['threshold']:.2g} (150M: AUROC {sp150['auroc']:.3f}, MCC {sp150['mcc_at_dev_threshold']:.3f}).

**Locations**: macro AUROC {T['macro_auroc_loc']:.3f}, macro F1 {T['macro_f1_loc']:.3f} (150M: {T150['macro_auroc_loc']:.3f}, {T150['macro_f1_loc']:.3f}). AUROC looks flattering for rare classes, so average precision (AP) and F1 are the numbers to trust. The last two columns are the released 150M model.

| Class | Test positives | AUROC | AP | F1 | AP (150M) | F1 (150M) |
|---|---|---|---|---|---|---|
{rows}

### 650M vs 150M on the same test proteins
We bootstrapped both models' test predictions together: {B['n_clusters']} MMseqs2 test clusters resampled 2000 times, with each model's validation-chosen thresholds fixed. "Share" is the fraction of resamples in which the 650M model is better.

| Metric (650M minus 150M) | Difference | 95% interval | Share |
|---|---|---|---|
{pd_rows}

An interval that includes 0 means this test set cannot tell the two models apart on that metric. These intervals cover test-set sampling only. The 150M seed spread (3 seeds, reported on its card) is a rough guide to training randomness; this 650M model has one seed, so a difference smaller than that spread may not hold up for another run.

### Test-set uncertainty (650M alone)
| Metric | Point | 95% interval |
|---|---|---|
{ci_rows}

All classes: `bootstrap_ci.json`; script: `training/bootstrap_loc.py`. These numbers are not directly comparable to published tools such as SignalP or DeepLoc, which use different datasets, splits and label definitions. We have not run those tools on this test set.

## Data
Identical to the 150M model: UniProtKB/Swiss-Prot reviewed entries (CC BY 4.0), retrieved 2026-11-17, with locations backed by experimental evidence mapped to 10 classes, and a signal peptide label from UniProt SIGNAL features (often predicted rather than experimental; see the [150M card](https://huggingface.co/{SMALL}) for the evidence breakdown). Split by MMseqs2 clustering at 30% identity with leakage filtering. The exact split is the dataset [{DS}](https://huggingface.co/datasets/{DS}).

## Training
Same heads and loss as the 150M model: mean-pooled ESM-2 features into a 10-way location head, a signal peptide head that also sees the mean of the first 30 residues, binary cross-entropy with class weights (max 10), bf16, AdamW with warmup and linear decay, lr {A['lr']:g}.

What differs, and why: full fine-tuning of 650M parameters does not fit in the 12 GB of our single RTX 3060. We froze the embeddings and the bottom {fz} of 33 encoder layers and trained the top {33 - fz} layers plus the heads, with gradient checkpointing, batch size {A['bs']} and gradient accumulation {A.get('accum', 1)} (effective batch {eff}; the 150M used batch 8 without accumulation), for {A['epochs']} epochs (the 150M used 2). These choices were fixed before training from memory tests, not tuned on validation. The epoch with the best validation score (mean of location macro AUROC and signal peptide AUROC) was kept. Training took about {minutes / 60:.1f} hours.

| Epoch | Dev macro AUROC | Dev SP AUROC | Dev macro AP | Dev rare AP | Selection score |
|---|---|---|---|---|---|
{hist}

The 150M model's best selection score was {best150['select']:.4f}. `training/` has the scripts and `provenance.json`.

## Limitations
- One training seed. The 150M model's per-class rare F1 varied a lot across seeds, so expect the same here.
- Still weak on Golgi, lysosome/vacuole and peroxisome; check their AP above before relying on them.
- Labels come from curated annotations, which are incomplete and sometimes inconsistent; only 10 coarse locations; isoform-specific locations are merged; no signal peptide cleavage site.
- Mostly well-studied proteins, so expect lower accuracy on proteins unlike Swiss-Prot.
- Research use only.

## Files
`SHA256SUMS` lists checksums for every file.
"""
(rel / "README.md").write_text(card, encoding="utf-8")
files = sorted(p.relative_to(rel).as_posix() for p in rel.rglob("*") if p.is_file() and p.name != "SHA256SUMS" and "__pycache__" not in p.parts)
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
(rel / "SHA256SUMS").write_text("".join(f"{sha(rel / f)}  {f}\n" for f in files))
print("CARD650_DONE", len(files), "files")
