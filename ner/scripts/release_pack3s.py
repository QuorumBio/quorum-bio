"""Assemble ner/release3s for NER v3 small (v3S: 6-layer student distilled from v3a, 6 types, seed 13). Every number in the card is read from result files.
Inputs: ner/models/v3S/{results,eval_pipeline,bootstrap_ci}.json, ner/release3s/onnx_results.json (+ onnx/), ner/models/v3S_s{1..4}/results.json,
ner/models/v3a/{results,eval_pipeline}.json, ner/release3/onnx_results.json (full-model comparison), ner/data3/meta.json, ner/release/{LICENSE,NOTICE} (for the base-model MIT text)."""
import hashlib, json, shutil, statistics as st
from pathlib import Path
root = Path(__file__).resolve().parents[1]
REPO = "quorumbio/biomedical-ner-gene-chemical-disease-species-small"
FULL = "quorumbio/biomedical-ner-gene-chemical-disease-species"
BASE = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
rel = root / "release3s"; run = root / "models" / "v3S"; FR = json.load(open(root / "models/v3a/results.json", encoding="utf-8")); rel.mkdir(exist_ok=True)
J = lambda p: json.load(open(p, encoding="utf-8"))
R, E, BS, O = J(run / "results.json"), J(run / "eval_pipeline.json"), J(run / "bootstrap_ci.json"), J(rel / "onnx_results.json")
META = J(root / "data3/meta.json"); FE = J(root / "models/v3a/eval_pipeline.json")["v3"]; FO = J(root / "release3/onnx_results.json")
SEEDS = {13: R["test"]} | {s: J(root / f"models/v3S_s{s}/results.json")["test"] for s in (1, 2, 3, 4)}
CORP = ["bc5cdr", "ncbi_disease", "nlmgene", "biored"]
NAME = {"bc5cdr": "BC5CDR", "ncbi_disease": "NCBI-disease", "nlmgene": "NLM-Gene", "biored": "BioRED"}
HUB = {"bc5cdr": "bigbio/bc5cdr", "ncbi_disease": "ncbi/ncbi_disease", "nlmgene": "bigbio/nlm_gene", "biored": "bigbio/biored"}
TYPES = META["types"]; SPL = META["splits"]
import math
from safetensors import safe_open
with safe_open(str(run / "model.safetensors"), "pt") as sf: PARAMS = sum(math.prod(sf.get_slice(k).get_shape()) for k in sf.keys()) / 1e6

for f in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"): shutil.copy(run / f, rel / f)
for f in ("special_tokens_map.json", "vocab.txt"):
    if (run / f).exists(): shutil.copy(run / f, rel / f)
shutil.copy(root / "release" / "LICENSE", rel / "LICENSE")
shutil.copy(run / "results.json", rel / "train_results.json"); shutil.copy(run / "eval_pipeline.json", rel / "eval_results.json")
shutil.copy(run / "bootstrap_ci.json", rel / "bootstrap_ci.json")
(rel / "seed_results.json").write_text(json.dumps({"note": "test entity F1 from train3.py's own evaluation; seed 13 is the released checkpoint; seeds 1-4 pre-registered", "per_seed": {str(k): {c: v[c]["f1"] for c in CORP} for k, v in SEEDS.items()}}, indent=2))
(rel / "training").mkdir(exist_ok=True)
for f in ("prepare_data.py", "prepare_data3.py", "train3.py", "eval_v3.py", "onnx_export3.py", "bootstrap_ner3.py", "train3_distill.py", "release_pack3s.py"): shutil.copy(root / "scripts" / f, rel / "training" / f)
(rel / "training" / "data3_meta.json").write_text(json.dumps(META, indent=2)); shutil.copy(root / "data" / "provenance.json", rel / "training" / "provenance_bc5cdr_ncbi.json")

mit = (root / "release" / "NOTICE").read_text(encoding="utf-8").split("-----")[1]
(rel / "NOTICE").write_text(f"""Quorum Bio biomedical NER, small (Gene, Chemical, Disease, Species, Variant, CellLine)
Copyright 2026 Quorum Bio. Licensed under the Apache License, Version 2.0 (see LICENSE).

This model is distilled from {FULL} (Apache-2.0), which is fine-tuned from {BASE}, released under the MIT License:
-----{mit}-----

Training data:
- BioCreative V CDR corpus (BC5CDR), Li et al., 2016, Database (Oxford) baw068. https://doi.org/10.1093/database/baw068
  The distributors do not state an explicit license; see the model card (Data and licensing).
- NCBI Disease corpus, Dogan et al., 2014, J Biomed Inform 47:1-10. https://doi.org/10.1016/j.jbi.2013.12.006
  Distributed by NCBI under its public domain notice.
- NLM-Gene corpus, Islamaj et al., 2021, J Biomed Inform 118:103779. https://doi.org/10.1016/j.jbi.2021.103779
  CC0 1.0 (Zenodo record 5089049).
- BioRED corpus, Luo et al., 2022, Briefings in Bioinformatics 23(5):bbac282. https://doi.org/10.1093/bib/bbac282
  United States Government Work (NCBI public domain notice); the distributors ask that the authors be cited.
""", encoding="utf-8")

f3 = lambda x: "%.3f" % x
ev = E["v3"]; v2 = E["v2_released"]
def ci(c, k="all"): x = BS[c]["ci"][k]["f1"]; return f"{f3(x['lo'])} to {f3(x['hi'])}"
def mi(c):
    d = ev[c]; return f"""  - task:
      type: token-classification
      name: Named entity recognition
    dataset:
      name: {NAME[c]} ({', '.join(TYPES[c])})
      type: {HUB[c]}
      split: test
    metrics:
    - type: f1
      value: {f3(d['f1'])}
      name: Entity F1 (seqeval, own word tokenization, only this corpus's types scored; not comparable to published numbers)
    - type: precision
      value: {f3(d['precision'])}
      name: Entity precision
    - type: recall
      value: {f3(d['recall'])}
      name: Entity recall"""
rows = "\n".join(f"| {NAME[c]} ({SPL[f'{c}/test']['docs_kept']} abstracts) | {', '.join(TYPES[c])} | {f3(ev[c]['precision'])} / {f3(ev[c]['recall'])} / {f3(ev[c]['f1'])} | {ci(c)} |" for c in CORP)
order = ["Gene", "Chemical", "Disease", "Species", "Variant", "CellLine"]
brow = "\n".join(f"| {t} | {int(ev['biored']['per_type'][t]['support'])} | {f3(ev['biored']['per_type'][t]['precision'])} / {f3(ev['biored']['per_type'][t]['recall'])} / {f3(ev['biored']['per_type'][t]['f1-score'])} | {ci('biored', t)} |" for t in order)
srow = "\n".join(f"| {NAME[c]} | {f3(st.mean([SEEDS[s][c]['f1'] for s in SEEDS]))} ± {f3(st.stdev([SEEDS[s][c]['f1'] for s in SEEDS]))} | {f3(min(SEEDS[s][c]['f1'] for s in SEEDS))} to {f3(max(SEEDS[s][c]['f1'] for s in SEEDS))} |" for c in CORP)
nrow = "\n".join(f"| {NAME[c]} | {f3(ev[c]['f1'])} | {f3(FE[c]['f1'])} | {ev[c]['f1'] - FE[c]['f1']:+.3f} |" for c in CORP)
pv = BS["paired_v3_minus_v2"]
drow = "\n".join(f"| {NAME[c]} | {', '.join(TYPES[c])} | {f3(ev[c]['f1'])} | {f3(v2[c]['f1'])} | {pv[c]['point']:+.3f} ({pv[c]['lo']:+.3f} to {pv[c]['hi']:+.3f}) |" for c in ("bc5cdr", "ncbi_disease"))
rm = META["removed_for_cross_corpus_overlap"]
drows = "\n".join(f"| {NAME[c]} | {', '.join(TYPES[c])} | {SPL[f'{c}/train']['docs_kept']} | {SPL[f'{c}/dev']['docs_kept']} | {SPL[f'{c}/test']['docs_kept']} | {rm.get(f'{c}/train', 0)} / {rm.get(f'{c}/dev', 0)} |" for c in CORP)
ep = max(R["val_history"], key=lambda h: h["f1"])["epoch"]
par = O["parity_fp32_vs_torch"]

card = f"""---
license: apache-2.0
language: en
library_name: transformers
base_model: {FULL}
base_model_relation: finetune
pipeline_tag: token-classification
tags: [biomedical, biomedical-ner, ner, distillation, gene, protein, chemical, disease, species, variant, cell-line, bc5cdr, ncbi-disease, nlm-gene, biored, pubmed, onnx, transformers.js]
datasets: [bigbio/bc5cdr, ncbi/ncbi_disease, bigbio/nlm_gene, bigbio/biored]
model-index:
- name: biomedical-ner-gene-chemical-disease-species-small
  results:
{chr(10).join(mi(c) for c in CORP)}
---
# Biomedical NER, small: Gene, Chemical, Disease, Species, Variant, Cell line (6 layers, Apache-2.0)

A smaller, faster version of [{FULL}](https://huggingface.co/{FULL}) that tags the same six entity types (**Gene**, **Chemical**, **Disease**, **Species**, **Variant**, **CellLine**) in English biomedical text. {PARAMS:.1f}M parameters (6 transformer layers instead of 12). The int8 ONNX file is {O['files']['int8']['mb']} MB (full model: {FO['files']['int8']['mb']} MB) and ran at {O['int8']['ms_per_sentence']} ms per sentence on CPU in our test (full model: {FO['int8']['ms_per_sentence']} ms). Mean validation F1 is {FR['best_val_f1_mean_over_corpora'] - R['best_val_f1_mean_over_corpora']:.3f} lower. Pick this one for browsers, CPUs and high volume; pick the full model when accuracy matters more.

## Use
```python
from transformers import pipeline
ner = pipeline("token-classification", model="{REPO}", aggregation_strategy="simple")
ner("The BRCA1 c.68_69delAG mutation was found in HeLa cells and in mice with breast cancer that were treated with tamoxifen.")
```
ONNX: `onnx/model.onnx` (fp32, {O['files']['fp32']['mb']} MB) and `onnx/model_quantized.onnx` (dynamic int8, {O['files']['int8']['mb']} MB); inputs `input_ids`, `attention_mask`, `token_type_ids`, output `logits` (13 labels: O and B-/I- for each type).

## How it was trained
Student: the full model with every other transformer layer kept (layers 2, 4, 6, 8, 10 and 12 of 12), so it starts from trained weights. It was then trained on the same data with the full model's partial-label loss (an O word in a corpus means "none of this corpus's types"), plus distillation from the full model: loss = 0.5 x partial-label loss + 0.5 x T^2 x KL(teacher || student), temperature T = 2, on the first subword of each word. Checkpoint chosen on validation (mean entity F1 over the four validation sets: {R['best_val_f1_mean_over_corpora']:.3f}; full model {FR['best_val_f1_mean_over_corpora']:.3f}). Before training we set the rule that a small model would be released only if it stayed within 0.02 of the full model on that validation mean.

Test F1, same pipeline and scoring:

| Test set | This model | Full model | Difference |
|---|---|---|---|
{nrow}

## Results
Entity-level precision / recall / F1 (seqeval, exact span and type) on words from a simple regex tokenizer, sentence-split input, through the `transformers` pipeline (`aggregation_strategy="simple"`). Each test set is scored only on the types its corpus annotates (other predicted types are dropped first), and test sets were scored once after the checkpoint was chosen on validation (epoch {ep}). Intervals are 95% bootstrap intervals over test abstracts (2000 resamples).

| Test set | Types scored | P / R / F1 | F1 95% interval |
|---|---|---|---|
{rows}

BioRED test by type:

| Type | Mentions | P / R / F1 | F1 95% interval |
|---|---|---|---|
{brow}

CellLine has few test mentions, so its interval is wide. These scores are not comparable to published BC5CDR, NCBI-disease, NLM-Gene or BioRED numbers, which use different tokenization, scoring and (for NLM-Gene and BioRED) sometimes normalization to identifiers.

### Compared with our Chemical/Disease model
Same pipeline and scoring; paired bootstrap over the same resampled abstracts:

| Test set | Types | This model F1 | Chemical/Disease model F1 | Difference (95% interval) |
|---|---|---|---|---|
{drow}

### Seed variation
We repeated the distillation with 4 more seeds (1, 2, 3, 4; same teacher), chosen before training. All 5 runs (with the released seed 13) are reported. Numbers are test entity F1 from the training script's own evaluation (same metric, different input path from the pipeline table, so values differ slightly).

| Test set | Mean ± SD | Range |
|---|---|---|
{srow}

Per-seed values: `seed_results.json`.

### Runtime check (ONNX via onnxruntime, CPU)
| File | BC5CDR F1 | NCBI-disease F1 | NLM-Gene F1 | BioRED F1 | ms/sentence | Size |
|---|---|---|---|---|---|---|
| fp32 | {f3(O['fp32']['bc5cdr']['f1'])} | {f3(O['fp32']['ncbi_disease']['f1'])} | {f3(O['fp32']['nlmgene']['f1'])} | {f3(O['fp32']['biored']['f1'])} | {O['fp32']['ms_per_sentence']} | {O['files']['fp32']['mb']} MB |
| int8 (dynamic quantization) | {f3(O['int8']['bc5cdr']['f1'])} | {f3(O['int8']['ncbi_disease']['f1'])} | {f3(O['int8']['nlmgene']['f1'])} | {f3(O['int8']['biored']['f1'])} | {O['int8']['ms_per_sentence']} | {O['files']['int8']['mb']} MB |

fp32 ONNX vs PyTorch: max absolute logit difference {par['max_abs_logit_diff']:.2e}, token argmax agreement {par['token_argmax_agreement']:.4f} on {par['sentences_checked']} BioRED test sentences. (These use the training script's input path, so they differ slightly from the pipeline table.)

## Training data
| Corpus | Types annotated | Train abstracts | Validation | Test | Removed from train / validation |
|---|---|---|---|---|---|
{drows}

**No test abstract is used for training.** The corpora share many PubMed abstracts (for example, {rm['biored/train']} BioRED training abstracts are also in other corpora's test or validation sets). We kept every test set exactly as published and removed from training and validation any abstract that appears in any test set, and from training any abstract in any validation set. NLM-Gene has no validation split, so 50 of its training abstracts were drawn at random (seed 13) before training. Label mapping: BioRED ChemicalEntity, DiseaseOrPhenotypicFeature, GeneOrGeneProduct, OrganismTaxon, SequenceVariant and CellLine map to the six types; NLM-Gene's Gene, GENERIF and STARGENE mentions map to Gene, and its Domain and Other mentions (protein domains, gene products used as drugs or tools, which its guidelines say not to annotate as genes) are ignored in training and scoring.

5 epochs, batch size {R['args']['bs']}, learning rate {R['args']['lr']}, max length {R['args']['max_len']}, bf16 autocast, AdamW with linear warmup/decay, one NVIDIA RTX 3060 (about {R['train_minutes']:.0f} minutes). Reproduce: `python training/prepare_data.py`, `python training/prepare_data3.py` (after downloading BIORED.zip and NLM-Gene-Corpus.zip; URLs and hashes in `training/data3_meta.json`), then train or download the full model and run `python training/train3_distill.py --teacher <full model dir> --keep 1,3,5,7,9,11 --out v3S`.

## Limitations
- About 0.01 to 0.02 F1 below the full model on most test sets (table above).
- English only, six types only, no normalization to database identifiers (NCBI Gene, MeSH, NCBI Taxonomy).
- Species, Variant and CellLine are learned from BioRED alone ({SPL['biored/train']['docs_kept']} training abstracts after de-duplication), so coverage of rare organisms, variant notations and cell line names is limited.
- Type definitions follow the source corpora, which differ: BioRED's Disease includes phenotypic features and BC5CDR's does not; NLM-Gene labels gene family and complex mentions as genes. Expect boundary and type disagreements on such mentions.
- Trained on PubMed abstracts; expect lower accuracy on full text, clinical notes, patents or social media.
- If you need only one type, dedicated single-type models (for example the OpenMed NER series) may suit you better. We have not benchmarked against them.
- Not validated for clinical or diagnostic use.

## Data and licensing
Model weights: Apache-2.0. Distilled from our Apache-2.0 full model, which is fine-tuned from an MIT base model (see `NOTICE`). Training data: NLM-Gene is CC0 1.0. BioRED is a US Government work with an NCBI public domain notice. NCBI-disease is distributed by NCBI under its public domain notice. BC5CDR: the distributors do not state an explicit license ("freely available for the research community"); BigBio lists it as Public Domain Mark 1.0. If you have strict compliance needs, review the corpus terms yourself. Please cite the corpora (references in `NOTICE`).

## Files
`SHA256SUMS` lists checksums for every file in this repo.
"""
(rel / "README.md").write_text(card, encoding="utf-8")
lines = [hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.relative_to(rel).as_posix() for p in sorted(rel.rglob("*")) if p.is_file() and p.name != "SHA256SUMS" and "__pycache__" not in p.parts]
(rel / "SHA256SUMS").write_text("\n".join(lines) + "\n"); print("PACK3_DONE", len(lines), "files")
