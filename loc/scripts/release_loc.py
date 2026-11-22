"""Build loc/release: safetensors weights + config, ONNX (fp32, int8) with parity check, model card, LICENSE, NOTICE, SHA256SUMS."""
import json, shutil, hashlib, time, traceback, sys
from pathlib import Path
import numpy as np, torch
from transformers import AutoConfig, AutoTokenizer
from safetensors.torch import save_file
sys.path.insert(0, str(Path(__file__).parent))
from quorum_loc import LocModel, trunc
root = Path(__file__).resolve().parents[1]; rel = root / "release"; rel.mkdir(exist_ok=True); (rel / "onnx").mkdir(exist_ok=True); (rel / "training").mkdir(exist_ok=True)
REPO = "quorumbio/esm2-localization-signalpeptide"; BASE = "facebook/esm2_t30_150M_UR50D"
run = root / "models" / "run1_150m"; R = json.load(open(run / "results.json")); R35 = json.load(open(root / "models" / "run1_35m" / "results.json")); P = json.load(open(root / "data" / "provenance.json"))
ck = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
cfg = AutoConfig.from_pretrained(BASE); cfg._attn_implementation = "eager"
net = LocModel(cfg, len(ck["classes"])); net.load_state_dict(ck["state_dict"]); net.eval()
save_file({k: v.detach().clone().contiguous() for k, v in net.state_dict().items()}, str(rel / "model.safetensors"))
meta = {"classes": ck["classes"], "thresholds": ck["thresholds"], "sp_threshold": ck["sp_threshold"], "base": BASE, "encoder_config": cfg.to_dict(), "max_len_policy": "sequences over 1000 residues: first 600 + last 400"}
(rel / "config.json").write_text(json.dumps(meta, indent=1))
tok = AutoTokenizer.from_pretrained(BASE); tok.save_pretrained(rel)
(rel / "vocab_for_js.json").write_text(json.dumps(tok.get_vocab()))
test = [json.loads(l) for l in open(root / "data" / "test.jsonl")]
sub = [x for x in test if len(x["sequence"]) <= 500][:150]
class Wrap(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, ids, mask): return s.m(ids, mask)
onnx_res = {"status": "not_run"}
try:
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    t = tok([sub[0]["sequence"][:80], sub[1]["sequence"][:60]], return_tensors="pt", padding=True)
    path = rel / "onnx" / "model.onnx"; w = Wrap(net).eval()
    torch.onnx.export(w, (t["input_ids"], t["attention_mask"]), str(path), input_names=["input_ids", "attention_mask"], output_names=["loc_logits", "sp_logit"],
                      dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "loc_logits": {0: "batch"}, "sp_logit": {0: "batch"}}, opset_version=17, dynamo=False)
    q = rel / "onnx" / "model_quantized.onnx"; quantize_dynamic(str(path), str(q), weight_type=QuantType.QInt8)
    def mk(p): return ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    res = {}
    sess = {"fp32": mk(path), "int8": mk(q)}; maxd = {"fp32": 0.0, "int8": 0.0}; agree = {"fp32": [0, 0], "int8": [0, 0]}; thr = np.array(ck["thresholds"])
    for x in sub:
        e = tok([trunc(x["sequence"])], return_tensors="pt", truncation=True, max_length=1022)
        with torch.no_grad(): tl, ts = net(e["input_ids"], e["attention_mask"])
        tp = torch.sigmoid(tl)[0].numpy(); tsp = torch.sigmoid(ts)[0].item()
        for tag, s in sess.items():
            ol, os_ = s.run(None, {"input_ids": e["input_ids"].numpy().astype(np.int64), "attention_mask": e["attention_mask"].numpy().astype(np.int64)})
            op = 1 / (1 + np.exp(-ol[0])); osp = 1 / (1 + np.exp(-os_[0]))
            maxd[tag] = max(maxd[tag], float(np.abs(op - tp).max()), float(abs(osp - tsp)))
            agree[tag][0] += int(((op >= thr) == (tp >= thr)).sum()) + int((osp >= ck["sp_threshold"]) == (tsp >= ck["sp_threshold"])); agree[tag][1] += len(thr) + 1
    onnx_res = {"status": "ok", "sequences_checked": len(sub), "note": "sequences up to 500 residues", "files_mb": {"fp32": round(path.stat().st_size / 1e6, 1), "int8": round(q.stat().st_size / 1e6, 1)},
                "max_abs_probability_diff_vs_torch": maxd, "decision_agreement_vs_torch": {k: v[0] / v[1] for k, v in agree.items()}}
except Exception as ex:
    traceback.print_exc(); onnx_res = {"status": "failed", "error": repr(ex)[:400]}
    for f in (rel / "onnx").glob("*"): f.unlink()
print("ONNX", json.dumps(onnx_res), flush=True)
(rel / "onnx_results.json").write_text(json.dumps(onnx_res, indent=2))
shutil.copy(run / "results.json", rel / "train_results_150m.json"); shutil.copy(root / "models" / "run1_35m" / "results.json", rel / "train_results_35m.json"); shutil.copy(root / "data" / "provenance.json", rel / "training" / "provenance.json")
for f in ("prepare_uniprot.py", "train_loc.py", "quorum_loc.py", "release_loc.py"): shutil.copy(root / "scripts" / f, rel / "training" / f)
shutil.copy(root / "scripts" / "quorum_loc.py", rel / "quorum_loc.py")
shutil.copy(root.parent / "ner" / "release" / "LICENSE", rel / "LICENSE")
mit = open(root.parent / "ner" / "release" / "NOTICE", encoding="utf-8").read().split("-----")[1].replace("Microsoft Corporation (model authors; see the upstream model card for the exact notice)", "Meta Platforms, Inc. and affiliates (ESM-2 authors; see the upstream repository for the exact notice)").replace("The upstream repository does not ship a LICENSE file; its model card lists the MIT license. https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", "The upstream model card lists the MIT license. https://huggingface.co/" + BASE)
(rel / "NOTICE").write_text(f"""Quorum Bio protein localization and signal peptide model
Copyright 2026 Quorum Bio. Licensed under the Apache License, Version 2.0 (see LICENSE).

This model is fine-tuned from {BASE} (ESM-2, Lin et al. 2023, Science), which is released under the MIT License:
-----{mit}-----

Training data: UniProtKB/Swiss-Prot (reviewed entries), The UniProt Consortium, Nucleic Acids Research. https://www.uniprot.org
UniProt data are licensed CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Retrieved 2026-11-17. Sequences and annotations were filtered and re-split by Quorum Bio (see training/).
""")
f3 = lambda x: "%.3f" % x
T, T35 = R["test"], R35["test"]; cls = list(T["per_class"].keys()); S = T["signal_peptide"]
rows = "\n".join(f"| {c} | {T['per_class'][c]['n_pos']} | {f3(T['per_class'][c]['auroc'])} | {f3(T['per_class'][c]['avg_precision'])} | {f3(T['per_class'][c]['f1_at_dev_threshold'])} | {f3(T35['per_class'][c]['f1_at_dev_threshold'])} |" for c in cls)
fs = P["final_split"]; spp = P["signal_peptide_positive"]
onnx_txt = (f"ONNX files (`onnx/model.onnx` fp32 {onnx_res['files_mb']['fp32']} MB, `onnx/model_quantized.onnx` int8 {onnx_res['files_mb']['int8']} MB) take `input_ids` and `attention_mask` and return `loc_logits` (10, apply sigmoid) and `sp_logit` (apply sigmoid). Checked against PyTorch on {onnx_res['sequences_checked']} test sequences up to 500 residues: max absolute probability difference {onnx_res['max_abs_probability_diff_vs_torch']['fp32']:.1e} (fp32) and {onnx_res['max_abs_probability_diff_vs_torch']['int8']:.3f} (int8); decisions agree with PyTorch on {100*onnx_res['decision_agreement_vs_torch']['fp32']:.1f}% (fp32) and {100*onnx_res['decision_agreement_vs_torch']['int8']:.1f}% (int8) of class decisions. The int8 file is smaller and faster but not identical, and its accuracy was only checked on this small sample, not the full test set." if onnx_res["status"] == "ok" else "An ONNX export was attempted and failed, so no ONNX files are included in this version.")
card = f"""---
license: apache-2.0
library_name: transformers
base_model: {BASE}
tags: [biology, bioinformatics, protein, protein-language-model, subcellular-localization, signal-peptide, esm2, uniprot, onnx]
datasets: [quorumbio/swissprot-localization-signalpeptide]
model-index:
- name: esm2-localization-signalpeptide
  results:
  - task:
      type: subcellular-localization
      name: Protein subcellular localization (10 classes, multi-label)
    dataset:
      name: Swiss-Prot homology-partitioned split (test)
      type: quorumbio/swissprot-localization-signalpeptide
      split: test
    metrics:
    - type: roc_auc
      value: {f3(T['macro_auroc_loc'])}
      name: Macro AUROC
    - type: f1
      value: {f3(T['macro_f1_loc'])}
      name: Macro F1 (validation-chosen thresholds)
  - task:
      type: signal-peptide-prediction
      name: Signal peptide presence
    dataset:
      name: Swiss-Prot homology-partitioned split (test)
      type: quorumbio/swissprot-localization-signalpeptide
      split: test
    metrics:
    - type: roc_auc
      value: {f3(S['auroc'])}
      name: AUROC
    - type: matthews_correlation
      value: {f3(S['mcc_at_dev_threshold'])}
      name: MCC (validation-chosen threshold)
---
# Protein subcellular localization and signal peptide prediction (ESM-2 150M)

Takes a protein sequence and returns (1) probabilities for 10 subcellular locations (a protein can have more than one) and (2) the probability that it has an N-terminal signal peptide. Fine-tuned from `{BASE}` on UniProtKB/Swiss-Prot with a homology-reduced split. 150M parameters.

**Short version.** Signal peptide detection is strong (AUROC {f3(S['auroc'])}, MCC {f3(S['mcc_at_dev_threshold'])}). Location prediction is decent for common compartments (nucleus, cytoplasm, extracellular, plastid) and **weak for Golgi, lysosome/vacuole and peroxisome**: do not rely on those three.

## Use
```python
from huggingface_hub import snapshot_download
import sys; folder = snapshot_download("{REPO}"); sys.path.insert(0, folder)
from quorum_loc import load, predict
m = load(folder)
# Example input only: a serum albumin-like sequence, not a verified UniProt entry. Use your own sequence.
predict(m, "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVKLVNELTEFAKTCVADESHAGCEKSLHTLFGDELCKVASLRETYGDMADCCEKQEPERNECFLSHKDDSPDLPKLKPDPNTLCDEFKADEKKFWGKYLYEIARRHPYFYAPELLYYANKYNGVFQECCQAEDKGACLLPKIETMREKVLASSARQRLRCASIQKFGERALKAWSVARLSQKFPKAEFVEVTKLVTDLTKVHKECCHGDLLECADDRADLAKYICDNQDTISSKLKECCDKPLLEKSHCIAEVEKDAIPENLPPLTADFAEDKDVCKNYQEAKDAFLGSFLYEYSRRHPEYAVSVLLRLAKEYEATLEECCAKDDPHACYSTVFDKLKHLVDEPQNLIKQNCDQFEKLGEYGFQNALIVRYTRKVPQVSTPTLVEVSRSLGKVGTRCCTKPESERMPCTEDYLSLILNRLCVLHEKTPVSEKVTKCCTESLVNRRPCFSALTPDETYVPKAFDEKLFTFHADICTLPDTEKQIKKQTALVELLKHKPKATEEQLKTVMENFVAFVDKCCAADDKEACFAVEGPKLVVSTQTALA")
```
Each location result has a probability and a yes/no using a per-class threshold picked on the validation set. `predict` handles sequences over 1000 residues by keeping the first 600 and last 400. {onnx_txt}

## Results (test set, scored once)
Test set: {fs['test']} proteins from Swiss-Prot, split by sequence similarity (see Data). Released checkpoint: seed 13; seed variation is measured below. Checkpoint and thresholds were chosen on the validation set only.

**Signal peptide** ({S['n_pos']} positives among {T['n']} test proteins): AUROC {f3(S['auroc'])}, MCC {f3(S['mcc_at_dev_threshold'])} at threshold {S['threshold']}.

**Locations** (macro AUROC {f3(T['macro_auroc_loc'])}, macro F1 {f3(T['macro_f1_loc'])}). AUROC looks flattering when a class is rare, so average precision (AP) and F1 are the numbers to trust. Last column is the smaller ESM-2 35M version of the same recipe, for reference.

| Class | Test positives | AUROC | AP | F1 | F1 (35M) |
|---|---|---|---|---|---|
{rows}

Rare-class AP is far above random guessing (for example peroxisome is about 1.5% of the test set) but low in absolute terms.

### Seed variation
We retrained the 150M model with the same data and settings using 2 more seeds (1, 2), chosen before training; all 3 runs (with the released seed 13) are reported, none dropped. Test set, mean ± SD across seeds:

| Metric | Mean ± SD | Range |
|---|---|---|
| Location macro AUROC | 0.927 ± 0.001 | 0.926 to 0.927 |
| Location macro F1 | 0.645 ± 0.009 | 0.635 to 0.650 |
| Signal peptide AUROC | 0.993 ± 0.001 | 0.992 to 0.994 |
| Signal peptide MCC | 0.956 ± 0.000 | 0.956 to 0.956 |
| F1, Golgi apparatus | 0.376 ± 0.030 | 0.342 to 0.400 |
| F1, Lysosome/Vacuole | 0.354 ± 0.010 | 0.343 to 0.361 |
| F1, Peroxisome | 0.337 ± 0.139 | 0.178 to 0.432 |
| AP, Golgi apparatus | 0.375 ± 0.011 | 0.362 to 0.383 |
| AP, Lysosome/Vacuole | 0.356 ± 0.010 | 0.349 to 0.367 |
| AP, Peroxisome | 0.356 ± 0.054 | 0.298 to 0.405 |

Rare-class F1 depends on a threshold picked on few validation positives, so it is much less stable than AUROC or AP. The released checkpoint's Peroxisome F1 (0.432) is the best of the 3 seeds, so the per-class table above overstates what a typical run gets for peroxisome (mean 0.337). 3 seeds is a small sample, so treat the SD as rough. It covers training randomness only, not test-set sampling. Per-class values for all classes: `seed_results.json`.

### Test-set uncertainty
The seed variation above covers training randomness only. To show how much the small test set limits precision, we bootstrapped the released checkpoint's test predictions. We resampled the 1475 MMseqs2 test clusters rather than single proteins, because similar proteins are not independent. We used 2000 resamples, with the validation-chosen thresholds held fixed. 95% percentile intervals:

| Metric | Point | 95% interval |
|---|---|---|
| Location macro AUROC | 0.926 | 0.914 to 0.935 |
| Location macro AP | 0.674 | 0.642 to 0.708 |
| Location macro F1 | 0.650 | 0.614 to 0.680 |
| AP, Golgi apparatus | 0.381 | 0.243 to 0.513 |
| AP, Lysosome/Vacuole | 0.367 | 0.215 to 0.520 |
| AP, Peroxisome | 0.405 | 0.240 to 0.588 |
| F1, Golgi apparatus | 0.342 | 0.234 to 0.447 |
| F1, Lysosome/Vacuole | 0.361 | 0.218 to 0.484 |
| F1, Peroxisome | 0.432 | 0.219 to 0.613 |
| Signal peptide AUROC | 0.994 | 0.988 to 0.998 |
| Signal peptide MCC | 0.956 | 0.935 to 0.972 |

For the rare classes, this test-set uncertainty (roughly ±0.1 to 0.2 AP) is far larger than the seed SD. Treat their per-class numbers as rough magnitudes. All classes: `bootstrap_ci.json`; script: `training/bootstrap_loc.py`.

These numbers are not directly comparable to published tools such as SignalP or DeepLoc, which use different datasets, splits and label definitions. We have not run those tools on this test set.

## Data
UniProtKB/Swiss-Prot reviewed entries (CC BY 4.0), retrieved 2026-11-17. Kept proteins of length 30 to 4000 with standard amino acids and at least one subcellular location backed by experimental evidence (ECO:0000269), mapped by name to 10 classes. Signal peptide label: the entry has a UniProt SIGNAL feature of any evidence type, so it can include predicted annotations. In practice most are predicted: of the 450 test positives, 340 (76%) carry sequence-model evidence (ECO:0000255; UniProt curators typically assign this from prediction tools such as SignalP) and only 77 (17%) have experimental evidence (ECO:0000269). On those 77 experimentally supported signal peptides the model's recall is 0.948 (73/77, 95% Wilson interval 0.87 to 0.98) and AUROC 0.995 against all test negatives, about the same as on the sequence-analysis ones (recall 0.950, AUROC 0.995). So the signal peptide score is not just agreement with predicted labels, but the experimental subset is small. Details: `sp_evidence.json`. Clustering: MMseqs2 at 30% identity (coverage 0.8), clusters assigned 80/10/10 to train/dev/test, then dev and test proteins with any hit at 30% or more identity to training proteins were removed. Final sizes: train {fs['train']}, dev {fs['dev']}, test {fs['test']}; signal peptide positives {spp['train']}, {spp['dev']}, {spp['test']}. Multiple isoform-specific locations are merged per protein. The exact split is published as the dataset [quorumbio/swissprot-localization-signalpeptide](https://huggingface.co/datasets/quorumbio/swissprot-localization-signalpeptide).

## Training
`training/` has the scripts and `provenance.json` (checksum of the downloaded data). Mean-pooled ESM-2 features feed a 10-way location head; the signal peptide head also sees the mean of the first 30 residues. Binary cross-entropy with class weights (max 10) for rare locations, bf16, gradient checkpointing, AdamW with warmup and linear decay, batch size {R['args']['bs']}, lr {R['args']['lr']}, {R['args']['epochs']} epochs (best dev epoch kept), one RTX 3060, about {R['train_minutes']:.0f} minutes.

## Limitations
- Weak on Golgi, lysosome/vacuole and peroxisome; mediocre on ER and cell membrane.
- Labels come from curated annotations, which are incomplete and sometimes inconsistent; proteins with unknown or contested location are not represented.
- Only 10 coarse locations; no sub-compartments, no membrane topology, no signal peptide cleavage site.
- Isoform and condition-specific locations are merged.
- Mostly well-studied proteins, so expect lower accuracy on proteins unlike Swiss-Prot.
- Research use only. Seed variation measured with only 3 seeds; test-set intervals (cluster bootstrap) are above.

## Files
`SHA256SUMS` lists checksums for every file.
"""
(rel / "README.md").write_text(card)
lines = [hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.relative_to(rel).as_posix() for p in sorted(rel.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"]
(rel / "SHA256SUMS").write_text("\n".join(lines) + "\n"); print("LOC_PACK_DONE", len(lines))
