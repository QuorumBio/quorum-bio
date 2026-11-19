"""Assemble ner/release for v2 (run2a: BC5CDR + NCBI-disease). Keeps LICENSE; rewrites NOTICE, README, results files, SHA256SUMS. Leaves ner/space (static) untouched."""
import json, shutil, hashlib
from pathlib import Path
root = Path(__file__).resolve().parents[1]
REPO = "quorumbio/biomed-ner-chem-disease"
BASE = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
rel = root / "release"; run = root / "models" / "run2a"
R = json.load(open(run / "results.json")); E = json.load(open(root / "results" / "eval_v2.json"))
O = json.load(open(root / "results" / "onnx_v2.json")); B = json.load(open(root / "results" / "baseline.json")); N = json.load(open(root / "results" / "ncbi_compare.json"))
for f in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"): shutil.copy(run / f, rel / f)
shutil.copy(run / "results.json", rel / "train_results.json"); shutil.copy(root / "results" / "eval_v2.json", rel / "eval_results.json")
shutil.copy(root / "results" / "onnx_v2.json", rel / "onnx_results.json"); shutil.copy(root / "results" / "baseline.json", rel / "baseline_results.json")
shutil.copy(root / "results" / "ncbi_compare.json", rel / "ncbi_compare_results.json")
(rel / "crosscorpus_results.json").unlink(missing_ok=True)
(rel / "training").mkdir(exist_ok=True)
for f in ("prepare_data.py", "train.py", "train2.py", "baseline_eval.py", "ncbi_compare.py", "eval_v2.py", "onnx_export2.py"): shutil.copy(root / "scripts" / f, rel / "training" / f)
(rel / "training" / "provenance.json").write_text((root / "data" / "provenance.json").read_text())
notice = (rel / "NOTICE").read_text()
up = notice.split("-----")[1]
(rel / "NOTICE").write_text(f"""Quorum Bio biomedical NER (Chemical, Disease)
Copyright 2026 Quorum Bio. Licensed under the Apache License, Version 2.0 (see LICENSE).

This model is fine-tuned from {BASE}, which is released under the MIT License:
-----{up}-----

Training data:
- BioCreative V CDR corpus (BC5CDR), Li et al., 2016, Database (Oxford) baw068. https://doi.org/10.1093/database/baw068
  PubMed abstracts and annotations. The distributors do not state an explicit license; see the model card (Data and licensing).
- NCBI Disease corpus, Dogan et al., 2014, J Biomed Inform 47:1-10. https://doi.org/10.1016/j.jbi.2013.12.006
  PubMed abstracts and annotations, distributed by NCBI under its public domain notice (see the model card).
""")
f3 = lambda x: "%.3f" % x
tr = lambda d: "%s / %s / %s" % (f3(d["precision"]), f3(d["recall"]), f3(d["f1"]))
pt = lambda d, t: "%s / %s / %s" % (f3(d["per_type"][t]["precision"]), f3(d["per_type"][t]["recall"]), f3(d["per_type"][t]["f1-score"]))
a, v1 = E["run2a"], E["run1"]; ab, an = a["bc5cdr_test"], a["ncbi_disease_test_disease_only"]
v1b, v1n = v1["bc5cdr_test"], v1["ncbi_disease_test_disease_only"]
ds, dg = B["d4data_strict_mapping"], B["d4data_generous_mapping"]; ns, ng = N["d4data_strict_mapping"], N["d4data_generous_mapping"]
ep = max(R["val_history"], key=lambda h: h["f1"])["epoch"]
card = f"""---
license: apache-2.0
language: en
library_name: transformers
base_model: {BASE}
pipeline_tag: token-classification
tags: [biomedical, ner, chemical, disease, bc5cdr, ncbi-disease, onnx]
---
# Biomedical NER: Chemical and Disease (BiomedBERT-base, Apache-2.0)

Token-classification model that tags **Chemical** and **Disease** mentions in English biomedical text. {ab['params_millions']}M parameters, fine-tuned from `{BASE}` on the BC5CDR and NCBI-disease training sets. ONNX (fp32 and int8) files are included.

**Version note.** This is v2 (trained on BC5CDR + NCBI-disease). v1 (BC5CDR only) is kept as the git revision `v1.0` of this repo.

## Use
```python
from transformers import pipeline
ner = pipeline("token-classification", model="{REPO}", aggregation_strategy="simple")
ner("Naloxone reverses the antihypertensive effect of clonidine.")
```
ONNX: `onnx/model.onnx` (fp32, {O['files']['fp32']['mb']} MB) and `onnx/model_quantized.onnx` (dynamic int8, {O['files']['int8']['mb']} MB), inputs `input_ids`, `attention_mask`, `token_type_ids`, output `logits`. Tokenize with the tokenizer in this repo.

## Results
Entity-level precision / recall / F1 (seqeval) on words from a simple regex tokenizer, sentence-split input, test splits scored once. The checkpoint was chosen on the **validation** sets (best mean validation entity F1 across the two corpora, epoch {ep}). The released checkpoint is seed {R['args']['seed']}; seed variation is measured below.

| Test set | Entity types scored | P / R / F1 | v1 (BC5CDR only) F1 |
|---|---|---|---|
| BC5CDR (500 abstracts) | Chemical, Disease | {tr(ab)} | {f3(v1b['f1'])} |
| NCBI-disease (100 abstracts) | Disease | {tr(an)} | {f3(v1n['f1'])} (never trained on this corpus) |

BC5CDR per type (P / R / F1): Chemical {pt(ab,'Chemical')}; Disease {pt(ab,'Disease')}.

NCBI-disease annotates Disease only, so on that test set predicted Chemical mentions are dropped before scoring. Adding NCBI-disease to training lowered BC5CDR F1 slightly (v1, a single seed: 0.896; v2 over 5 seeds: 0.889 ± 0.002, so the drop looks real but small) and raised NCBI-disease F1 a lot, which is expected since it is now in-domain. We also tried ignoring O labels for NCBI sentences (to avoid penalizing unlabeled chemicals); that variant collapsed precision on NCBI-disease (F1 0.427) and was discarded.

### Seed variation
We retrained v2 with the same data and settings using 4 more seeds (1, 2, 3, 4), chosen before training. All 5 runs (with the released seed 13) are reported, none dropped. Numbers are test entity F1 from the training script's own evaluation: same seqeval metric, but a different input path from the pipeline-based table above, which is why seed 13 shows 0.870 on NCBI-disease here and 0.878 there.

| Test set | Mean ± SD | Range |
|---|---|---|
| BC5CDR | 0.889 ± 0.002 | 0.887 to 0.891 |
| NCBI-disease | 0.872 ± 0.002 | 0.870 to 0.875 |

The released checkpoint is typical, not a lucky seed. These intervals cover training randomness only, not uncertainty from the small test sets (NCBI-disease test has 100 abstracts). Per-seed values: `seed_results.json`.

| Runtime check (ONNX via onnxruntime, CPU) | BC5CDR F1 | NCBI-disease F1 | ms/sentence | Size |
|---|---|---|---|---|
| fp32 | {f3(O['fp32']['f1'])} | {f3(O['fp32']['ncbi_disease_disease_only']['f1'])} | {O['fp32']['ms_per_sentence']} | {O['files']['fp32']['mb']} MB |
| int8 (dynamic quantization) | {f3(O['int8']['f1'])} | {f3(O['int8']['ncbi_disease_disease_only']['f1'])} | {O['int8']['ms_per_sentence']} | {O['files']['int8']['mb']} MB |

fp32 ONNX vs PyTorch: max absolute logit difference {O['parity_fp32_vs_torch']['max_abs_logit_diff']:.2e}, token argmax agreement {O['parity_fp32_vs_torch']['token_argmax_agreement']:.4f} on {O['parity_fp32_vs_torch']['sentences_checked']} BC5CDR sentences.

### Comparison with `d4data/biomedical-ner-all`
We ran `d4data/biomedical-ner-all` through the same pipeline and scoring (the v1 numbers below are from that run; v2 is in the table above). It is a general clinical model with 84 entity types that was **not trained on either corpus**, so its labels do not line up and we mapped them ourselves, which affects its score. Strict mapping (Disease_disorder to Disease, Medication to Chemical): BC5CDR F1 {f3(ds['f1'])} (P {f3(ds['precision'])}, R {f3(ds['recall'])}); NCBI-disease Disease F1 {f3(ns['f1'])} (P {f3(ns['precision'])}, R {f3(ns['recall'])}). Generous mapping (also Sign_symptom to Disease): BC5CDR F1 {f3(dg['f1'])}; NCBI-disease F1 {f3(ng['f1'])}. Our model was trained on these corpora's training splits, so this is an in-domain versus out-of-domain comparison, not evidence that our model is better in general. `d4data/biomedical-ner-all` ({ds['params_millions']}M parameters) is smaller and ran faster than this model in our GPU test ({ds['gpu_seconds']} s vs {v1b['gpu_seconds']} s for {v1b['sentences']} BC5CDR sentences). Its coverage of other entity types (symptoms, procedures, doses and more) is far broader than ours.

These scores are not directly comparable to published BC5CDR or NCBI-disease numbers, which use different tokenization and scoring.

## Training
`training/` has the scripts and `provenance.json` (source URLs and SHA256). 5 epochs, batch size {R['args']['bs']}, learning rate {R['args']['lr']}, max length {R['args']['max_len']}, bf16 autocast, AdamW with linear warmup/decay, one NVIDIA RTX 3060 (about {R['train_minutes']:.0f} minutes). Reproduce: `python training/prepare_data.py` then `python training/train2.py --base {BASE} --out run2a` (v1: `train.py --corpus bc5cdr`).

## Limitations
- English only, two entity types only (Chemical, Disease). No normalization to MeSH identifiers.
- NCBI-disease has no Chemical labels, so chemicals in NCBI-disease sentences are labeled O during training. This adds some label noise for the Chemical class.
- Trained on PubMed abstracts; expect lower accuracy on clinical notes, full text, patents, or social media.
- Boundary conventions follow BC5CDR and NCBI-disease, which differ in places (for example disease classes and modifiers); composite and discontinuous mentions are handled imperfectly.
- Not validated for clinical or diagnostic use. Do not use it to make medical decisions.
- Seed variation measured with 5 seeds (above); v1 was a single seed, and no test-set bootstrap intervals are given.

## Data and licensing
Model weights: Apache-2.0. Base model: MIT (see `NOTICE`). Training data: (1) BioCreative V CDR corpus (Li et al., 2016, Database baw068): the distributors do not state an explicit license on the BioCreative page or the public GitHub mirror ("freely available for the research community"); BigBio lists it as Public Domain Mark 1.0. (2) NCBI Disease corpus (Dogan et al., 2014): distributed by NCBI under its public domain notice. If you have strict compliance needs, review the corpus terms yourself.

## Files
`SHA256SUMS` lists checksums for every file in this repo.
"""
(rel / "README.md").write_text(card)
lines = []
for p in sorted(rel.rglob("*")):
    if p.is_file() and p.name != "SHA256SUMS": lines.append(hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.relative_to(rel).as_posix())
(rel / "SHA256SUMS").write_text("\n".join(lines) + "\n")
print("PACK2_DONE", len(lines), "files")
