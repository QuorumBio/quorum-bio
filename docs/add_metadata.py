"""Discoverability metadata for both model cards (README front matter) and their templates:
- `datasets:` (NER: the hub copies of the two corpora we trained on; loc: our own split dataset), so the models
  appear on those dataset pages;
- `model-index:` with our test numbers (metric names say they are not comparable to published benchmarks);
- a few accurate extra tags. Regenerates SHA256SUMS. Card text below the front matter is unchanged."""
import hashlib, json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
f3 = lambda x: "%.3f" % x

def ner_yaml(ab, an, lit):
    v = (lambda e: "{" + e + "}") if lit else (lambda e: f3(eval(e, {"f3": f3, "ab": ab, "an": an})))
    def res(dname, dtype, d, note):
        return f"""  - task:
      type: token-classification
      name: Named entity recognition
    dataset:
      name: {dname}
      type: {dtype}
      split: test
    metrics:
    - type: f1
      value: {v(f"f3({d}['f1'])" if lit else f"{d}['f1']")}
      name: Entity F1 ({note})
    - type: precision
      value: {v(f"f3({d}['precision'])" if lit else f"{d}['precision']")}
      name: Entity precision
    - type: recall
      value: {v(f"f3({d}['recall'])" if lit else f"{d}['recall']")}
      name: Entity recall"""
    return f"""datasets: [bigbio/bc5cdr, ncbi/ncbi_disease]
model-index:
- name: biomedical-ner-chemical-disease
  results:
{res("BC5CDR (Chemical + Disease)", "bigbio/bc5cdr", "ab", "seqeval, own word tokenization; not comparable to published BC5CDR numbers")}
{res("NCBI-disease (Disease)", "ncbi/ncbi_disease", "an", "seqeval, own word tokenization; not comparable to published numbers")}
"""

def loc_yaml(T, lit):
    S = T["signal_peptide"]
    vals = {"T['macro_auroc_loc']": T["macro_auroc_loc"], "T['macro_f1_loc']": T["macro_f1_loc"], "S['auroc']": S["auroc"], "S['mcc_at_dev_threshold']": S["mcc_at_dev_threshold"]}
    v = (lambda e: "{f3(" + e + ")}") if lit else (lambda e: f3(vals[e]))
    def m(t, e, n): return f"""    - type: {t}
      value: {v(e)}
      name: {n}"""
    ds = """    dataset:
      name: Swiss-Prot homology-partitioned split (test)
      type: quorumbio/swissprot-subcellular-localization-signal-peptide
      split: test"""
    return f"""datasets: [quorumbio/swissprot-subcellular-localization-signal-peptide]
model-index:
- name: esm2-protein-subcellular-localization-signal-peptide
  results:
  - task:
      type: subcellular-localization
      name: Protein subcellular localization (10 classes, multi-label)
{ds}
    metrics:
{m("roc_auc", "T['macro_auroc_loc']", "Macro AUROC")}
{m("f1", "T['macro_f1_loc']", "Macro F1 (validation-chosen thresholds)")}
  - task:
      type: signal-peptide-prediction
      name: Signal peptide presence
{ds}
    metrics:
{m("roc_auc", "S['auroc']", "AUROC")}
{m("matthews_correlation", "S['mcc_at_dev_threshold']", "MCC (validation-chosen threshold)")}
"""

E = json.load(open(root / "ner/release/eval_results.json"))["run2a"]; ab, an = E["bc5cdr_test"], E["ncbi_disease_test_disease_only"]
T = json.load(open(root / "loc/release/train_results_150m.json"))["test"]
JOBS = {
    "ner": {"tags_old": "tags: [biomedical, ner, chemical, disease, bc5cdr, ncbi-disease, onnx]",
            "tags_new": "tags: [biomedical, biomedical-ner, ner, chemical, disease, bc5cdr, ncbi-disease, pubmed, onnx, transformers.js]",
            "card": (root / "ner/release/README.md", ner_yaml(ab, an, False)), "tpl": (root / "ner/scripts/release_pack2.py", ner_yaml(ab, an, True))},
    "loc": {"tags_old": "tags: [biology, protein, subcellular-localization, signal-peptide, esm2, uniprot]",
            "tags_new": "tags: [biology, bioinformatics, protein, protein-language-model, subcellular-localization, signal-peptide, esm2, uniprot, onnx]",
            "card": (root / "loc/release/README.md", loc_yaml(T, False)), "tpl": (root / "loc/scripts/release_loc.py", loc_yaml(T, True))}}
for fam, J in JOBS.items():
    for key in ("card", "tpl"):
        p, y = J[key]; t = p.read_text(encoding="utf-8")
        assert t.count(J["tags_old"]) == 1 and "model-index" not in t, (p, "already patched or anchor missing")
        t = t.replace(J["tags_old"], J["tags_new"] + "\n" + y.rstrip("\n"))
        p.write_text(t, encoding="utf-8", newline="\n")
    rel = root / fam / "release"
    lines = [hashlib.sha256(q.read_bytes()).hexdigest() + "  " + q.relative_to(rel).as_posix() for q in sorted(rel.rglob("*")) if q.is_file() and q.name != "SHA256SUMS" and "__pycache__" not in q.parts]
    (rel / "SHA256SUMS").write_text("\n".join(lines) + "\n"); print(fam, "ok", len(lines))
