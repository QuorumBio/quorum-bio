"""Assemble ner/release (model repo) and ner/space (Gradio Space) from run1 + result JSONs. Writes SHA256SUMS."""
import json, shutil, hashlib, glob
from pathlib import Path
root = Path(__file__).resolve().parents[1]
REPO = "quorumbio/biomedical-ner-chemical-disease"
BASE = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
MIT_TEXT = 'MIT License\n\nCopyright (c) Microsoft Corporation (model authors; see the upstream model card for the exact notice)\n\nPermission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.\n\n(The upstream repository does not ship a LICENSE file; its model card lists the MIT license. https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext)'
rel = root / "release"; sp = root / "space"; run = root / "models" / "run1"
R = json.load(open(run / "results.json")); B = json.load(open(root / "results" / "baseline.json")); O = json.load(open(root / "results" / "onnx.json"))
for f in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"):
    shutil.copy(run / f, rel / f)
for f in glob.glob(str(run / "vocab.txt")) + glob.glob(str(run / "special_tokens_map.json")): shutil.copy(f, rel / Path(f).name)
shutil.copy(run / "results.json", rel / "train_results.json"); shutil.copy(root / "results" / "baseline.json", rel / "baseline_results.json"); shutil.copy(root / "results" / "onnx.json", rel / "onnx_results.json")
(rel / "training").mkdir(exist_ok=True)
for f in ("prepare_data.py", "train.py", "baseline_eval.py", "onnx_export.py"): shutil.copy(root / "scripts" / f, rel / "training" / f)
(rel / "training" / "provenance.json").write_text((root / "data" / "provenance.json").read_text())
# LICENSE (Apache-2.0 text from an installed Apache-2.0 package)
lic = None
for p in glob.glob(str(root.parents[0] / "venv" / "Lib" / "site-packages" / "*.dist-info" / "**" / "LICENSE*"), recursive=True):
    if not Path(p).is_file(): continue
    t = open(p, encoding="utf-8", errors="ignore").read()
    if "Apache License" in t and "Version 2.0, January 2004" in t and "APPENDIX" in t: lic = t; break
assert lic, "no Apache-2.0 text found"
(rel / "LICENSE").write_text(lic)
upstream = ""
try:
    from huggingface_hub import hf_hub_download
    upstream = open(hf_hub_download(BASE, "LICENSE"), encoding="utf-8").read()
except Exception as e:
    upstream = MIT_TEXT
(rel / "NOTICE").write_text(f"""Quorum Bio biomedical NER (Chemical, Disease)
Copyright 2026 Quorum Bio. Licensed under the Apache License, Version 2.0 (see LICENSE).

This model is fine-tuned from {BASE}, which is released under the MIT License:
-----
{upstream}
-----

Training data: BioCreative V CDR corpus (BC5CDR), Li et al., 2016, Database (Oxford) baw068. https://doi.org/10.1093/database/baw068
The corpus consists of PubMed abstracts and annotations. The distributors do not state an explicit license; see the model card (Data and licensing).
""")
f1 = lambda d: "%.3f" % d
r1 = B["run1"]; ds = B["d4data_strict_mapping"]; dg = B["d4data_generous_mapping"]
pt = lambda k, t: "%.3f / %.3f / %.3f" % (B[k]["per_type"][t]["precision"], B[k]["per_type"][t]["recall"], B[k]["per_type"][t]["f1-score"])
card = f"""---
license: apache-2.0
language: en
library_name: transformers
base_model: {BASE}
pipeline_tag: token-classification
tags: [biomedical, ner, chemical, disease, bc5cdr, onnx]
---
# Biomedical NER: Chemical and Disease (BiomedBERT-base, Apache-2.0)

Token-classification model that tags **Chemical** and **Disease** mentions in English biomedical text. It is {B['run1']['params_millions']}M parameters, fine-tuned from `{BASE}` on the BC5CDR training set. ONNX (fp32 and int8) files are included.

## Use
```python
from transformers import pipeline
ner = pipeline("token-classification", model="{REPO}", aggregation_strategy="simple")
ner("Naloxone reverses the antihypertensive effect of clonidine.")
```
ONNX: `onnx/model.onnx` (fp32, {O['files']['fp32']['mb']} MB) and `onnx/model_quantized.onnx` (dynamic int8, {O['files']['int8']['mb']} MB), inputs `input_ids`, `attention_mask`, `token_type_ids`, output `logits`. Tokenize with the tokenizer in this repo.

## Results (BC5CDR test set, 500 abstracts)
Entity-level precision / recall / F1 (seqeval) on words from a simple regex tokenizer, sentence-split input. Checkpoint chosen on the **validation** set (best validation entity F1, epoch {max(R['val_history'], key=lambda h: h['f1'])['epoch']}); the test set was scored once after selection. Single training run, single seed ({R['args']['seed']}), no confidence intervals.

| Model | Chemical (P/R/F1) | Disease (P/R/F1) | Overall F1 |
|---|---|---|---|
| This model | {pt('run1','Chemical')} | {pt('run1','Disease')} | {f1(r1['f1'])} |

| Runtime check (ONNX via onnxruntime, CPU) | Test F1 | ms/sentence | Size |
|---|---|---|---|
| fp32 | {f1(O['fp32']['f1'])} | {O['fp32']['ms_per_sentence']} | {O['files']['fp32']['mb']} MB |
| int8 (dynamic quantization) | {f1(O['int8']['f1'])} | {O['int8']['ms_per_sentence']} | {O['files']['int8']['mb']} MB |

fp32 ONNX vs PyTorch: max absolute logit difference {O['parity_fp32_vs_torch']['max_abs_logit_diff']:.2e}, token argmax agreement {O['parity_fp32_vs_torch']['token_argmax_agreement']:.4f} on {O['parity_fp32_vs_torch']['sentences_checked']} sentences.

### Comparison with `d4data/biomedical-ner-all`
We ran `d4data/biomedical-ner-all` through the same pipeline and scoring. Read this carefully: it is a general clinical model with 84 entity types that was **not trained on BC5CDR**, so its labels do not line up with BC5CDR's. We mapped its labels ourselves, which affects its score. Strict mapping (Disease_disorder to Disease, Medication to Chemical): F1 {f1(ds['f1'])} (P {f1(ds['precision'])}, R {f1(ds['recall'])}). Generous mapping (also Sign_symptom to Disease): F1 {f1(dg['f1'])} (P {f1(dg['precision'])}, R {f1(dg['recall'])}). Our model was trained on this corpus's training split, so this is an in-domain versus out-of-domain comparison, not evidence that our model is better in general. `d4data/biomedical-ner-all` ({ds['params_millions']}M parameters) is smaller than this model ({r1['params_millions']}M) and ran faster in our GPU test ({ds['gpu_seconds']} s vs {r1['gpu_seconds']} s for {r1['sentences']} sentences). Its coverage of other entity types (symptoms, procedures, doses and more) is far broader than ours.

These scores are not directly comparable to published BC5CDR numbers, which use different tokenization and scoring.

## Training
`training/` has the scripts and `provenance.json` (source URLs and SHA256). 5 epochs, batch size {R['args']['bs']}, learning rate {R['args']['lr']}, max length {R['args']['max_len']}, bf16 autocast, AdamW with linear warmup/decay, one NVIDIA RTX 3060 (about {R['train_minutes']:.0f} minutes). Reproduce: `python training/prepare_data.py` then `python training/train.py --corpus bc5cdr --base {BASE} --out run1`.

## Limitations
- English only, two entity types only (Chemical, Disease). No normalization to MeSH identifiers.
- Trained on PubMed abstracts; expect lower accuracy on clinical notes, full text, patents, or social media.
- Boundary conventions follow BC5CDR; composite and discontinuous mentions are handled imperfectly.
- Not validated for clinical or diagnostic use. Do not use it to make medical decisions.
- One run, one seed, no confidence intervals; small differences between models are not meaningful.

## Data and licensing
Model weights: Apache-2.0. Base model: MIT (see `NOTICE`). Training data: BioCreative V CDR corpus (Li et al., 2016, Database baw068), PubMed abstracts with chemical and disease annotations. The corpus distributors do not state an explicit license on the BioCreative page or the public GitHub mirror ("freely available for the research community"); BigBio lists it as Public Domain Mark 1.0, and the NCBI-disease corpus (not used here) carries an NCBI public domain notice. If you have strict compliance needs, review the corpus terms yourself.

## Files
`SHA256SUMS` lists checksums for every file in this repo.
"""
(rel / "README.md").write_text(card)
# Space
sp.mkdir(exist_ok=True)
(sp / "README.md").write_text(f"""---
title: Biomedical NER demo
sdk: gradio
app_file: app.py
license: apache-2.0
short_description: Tag chemicals and diseases in biomedical text
---
Demo for [{REPO}](https://huggingface.co/{REPO}). Research use only, not for clinical decisions.
""")
(sp / "requirements.txt").write_text("transformers\ntorch\n")
(sp / "app.py").write_text(f'''import gradio as gr
from transformers import pipeline

ner = pipeline("token-classification", model="{REPO}", aggregation_strategy="simple")

def run(text):
    ents = ner(text) if text.strip() else []
    return {{"text": text, "entities": [{{"entity": e["entity_group"], "start": e["start"], "end": e["end"]}} for e in ents]}}

demo = gr.Interface(
    run, gr.Textbox(lines=6, label="Biomedical text (English)"), gr.HighlightedText(label="Chemical and Disease mentions"),
    examples=[["Naloxone reverses the antihypertensive effect of clonidine."],
              ["Lidocaine-induced cardiac asystole developed in a patient with chronic renal failure."]],
    title="Quorum Bio: biomedical NER (Chemical, Disease)",
    description="Research demo. English only. Not for clinical decisions.")
demo.launch()
''')
# SHA256SUMS
lines = []
for p in sorted(rel.rglob("*")):
    if p.is_file() and p.name != "SHA256SUMS":
        lines.append(hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.relative_to(rel).as_posix())
(rel / "SHA256SUMS").write_text("\n".join(lines) + "\n")
print("PACK_DONE", len(lines), "files")
