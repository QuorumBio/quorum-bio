"""Aggregate multi-seed test results (mean, sample std, min, max) for NER v2 and the localization model.
Seed 13 is the released checkpoint; others were trained with identical settings. All runs are included."""
import json, statistics as st
from pathlib import Path
root = Path(__file__).resolve().parents[1]

def agg(vals):
    return {"n": len(vals), "mean": st.mean(vals), "std": st.stdev(vals) if len(vals) > 1 else 0.0, "min": min(vals), "max": max(vals), "values": vals}

out = {}
ner = [root / "ner/models/run2a/results.json"] + sorted((root / "ner/models/seeds").glob("run2a_s*/results.json"))
ner = [json.loads(p.read_text()) for p in ner if p.exists()]
out["ner_v2"] = {"seeds": [r["args"]["seed"] for r in ner],
                 **{f"test_f1_{c}": agg([r["test"][c]["f1"] for r in ner]) for c in ner[0]["test"]}}
loc = [root / "loc/models/run1_150m/results.json"] + sorted((root / "loc/models/seeds").glob("run1_150m_s*/results.json"))
loc = [json.loads(p.read_text()) for p in loc if p.exists()]
t0 = loc[0]["test"]
out["loc_150m"] = {"seeds": [r["args"]["seed"] for r in loc],
                   "macro_auroc_loc": agg([r["test"]["macro_auroc_loc"] for r in loc]),
                   "macro_f1_loc": agg([r["test"]["macro_f1_loc"] for r in loc]),
                   "sp_auroc": agg([r["test"]["signal_peptide"]["auroc"] for r in loc]),
                   "sp_mcc": agg([r["test"]["signal_peptide"]["mcc_at_dev_threshold"] for r in loc]),
                   "per_class_f1": {c: agg([r["test"]["per_class"][c]["f1_at_dev_threshold"] for r in loc]) for c in t0["per_class"]},
                   "per_class_ap": {c: agg([r["test"]["per_class"][c]["avg_precision"] for r in loc]) for c in t0["per_class"]}}
(root / "docs/seed_summary.json").write_text(json.dumps(out, indent=2))
def line(name, a): print(f"{name:32s} n={a['n']} mean={a['mean']:.4f} sd={a['std']:.4f} range=[{a['min']:.4f}, {a['max']:.4f}]")
for k, v in out.items():
    print(k, "seeds", v["seeds"])
    for kk, a in v.items():
        if isinstance(a, dict) and "mean" in a: line(kk, a)
        elif isinstance(a, dict) and kk != "seeds":
            for c, b in a.items(): line(f"{kk}.{c}", b)
