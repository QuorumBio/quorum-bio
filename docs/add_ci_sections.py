"""Add test-set bootstrap intervals to both model cards (release READMEs) and their templates; copy the bootstrap
scripts into release/training/; regenerate SHA256SUMS. Run loc/scripts/bootstrap_loc.py and ner/scripts/bootstrap_ner.py first."""
import hashlib, json, shutil
from pathlib import Path
root = Path(__file__).resolve().parents[1]
f = lambda x: f"{x:.3f}"
def ci(d): return f"{f(d['point'])} | {f(d['lo'])} to {f(d['hi'])}"

L = json.loads((root / "loc/release/bootstrap_ci.json").read_text())["ci"]
RARE = ("Golgi apparatus", "Lysosome/Vacuole", "Peroxisome")
rows = [("Location macro AUROC", L["macro_auroc"]), ("Location macro AP", L["macro_ap"]), ("Location macro F1", L["macro_f1"])]
rows += [(f"AP, {c}", L[f"{c}.ap"]) for c in RARE] + [(f"F1, {c}", L[f"{c}.f1"]) for c in RARE]
rows += [("Signal peptide AUROC", L["sp_auroc"]), ("Signal peptide MCC", L["sp_mcc"])]
LOC_SEC = f"""

### Test-set uncertainty
The seed variation above covers training randomness only. To show how much the small test set limits precision, we bootstrapped the released checkpoint's test predictions. We resampled the {json.loads((root / "loc/release/bootstrap_ci.json").read_text())["n_clusters"]} MMseqs2 test clusters rather than single proteins, because similar proteins are not independent. We used 2000 resamples, with the validation-chosen thresholds held fixed. 95% percentile intervals:

| Metric | Point | 95% interval |
|---|---|---|
{chr(10).join(f"| {n} | {ci(d)} |" for n, d in rows)}

For the rare classes, this test-set uncertainty (roughly ±0.1 to 0.2 AP) is far larger than the seed SD. Treat their per-class numbers as rough magnitudes. All classes: `bootstrap_ci.json`; script: `training/bootstrap_loc.py`."""

N = json.loads((root / "ner/release/bootstrap_ci.json").read_text())
b, n, pd_ = N["bc5cdr_test"], N["ncbi_disease_test_disease_only"], N["paired_v2_minus_v1_bc5cdr_f1"]
nrows = [(f"BC5CDR ({b['n_abstracts']} abstracts), all types", b["ci"]["all"]["f1"]), ("BC5CDR, Chemical", b["ci"]["Chemical"]["f1"]),
         ("BC5CDR, Disease", b["ci"]["Disease"]["f1"]), (f"NCBI-disease ({n['n_abstracts']} abstracts), Disease", n["ci"]["all"]["f1"])]
NER_SEC = f"""

### Test-set uncertainty
The seed variation above covers training randomness only. To show how much the test sets limit precision, we bootstrapped whole test abstracts, with 2000 resamples, for the released checkpoint on the pipeline path of the main table. 95% percentile intervals for entity F1:

| Test set | F1 | 95% interval |
|---|---|---|
{chr(10).join(f"| {k} | {ci(d)} |" for k, d in nrows)}

v1 vs v2 on BC5CDR, paired on the same resampled abstracts: v2 minus v1 F1 = {pd_['point']:.4f} (95% interval {pd_['lo']:.4f} to {pd_['hi']:.4f}). v2 scored lower in {pd_['share_of_resamples_v2_worse'] * 100:.1f}% of resamples, so the drop is small but consistent. Details: `bootstrap_ci.json`; script: `training/bootstrap_ner.py`."""

PATCHES = {
    "loc": [("Per-class values for all classes: `seed_results.json`.", "Per-class values for all classes: `seed_results.json`." + LOC_SEC),
            ("- Research use only. Seed variation measured with only 3 seeds; no test-set bootstrap intervals.",
             "- Research use only. Seed variation measured with only 3 seeds; test-set intervals (cluster bootstrap) are above.")],
    "ner": [("Per-seed values: `seed_results.json`.", "Per-seed values: `seed_results.json`." + NER_SEC),
            ("- Seed variation measured with 5 seeds (above); v1 was a single seed, and no test-set bootstrap intervals are given.",
             "- Seed variation measured with 5 seeds (above); v1 was a single seed. Test-set bootstrap intervals are above; the NCBI-disease test set is small (100 abstracts).")]}
FILES = {"loc": ("loc/release/README.md", "loc/scripts/release_loc.py", "loc/scripts/bootstrap_loc.py"),
         "ner": ("ner/release/README.md", "ner/scripts/release_pack2.py", "ner/scripts/bootstrap_ner.py")}
for fam, (card, tpl, script) in FILES.items():
    for path, is_tpl in ((card, False), (tpl, True)):
        p = root / path; t = p.read_text(encoding="utf-8")
        for old, new in PATCHES[fam]:
            assert t.count(old) == 1, (path, old[:60])
            t = t.replace(old, new.replace("{", "{{").replace("}", "}}") if is_tpl else new)
        p.write_text(t, encoding="utf-8", newline="\n")
    rel = root / fam / "release"; shutil.copy(root / script, rel / "training" / Path(script).name)
    lines = [hashlib.sha256(q.read_bytes()).hexdigest() + "  " + q.relative_to(rel).as_posix() for q in sorted(rel.rglob("*")) if q.is_file() and q.name != "SHA256SUMS"]
    (rel / "SHA256SUMS").write_text("\n".join(lines) + "\n"); print(fam, "ok", len(lines))
