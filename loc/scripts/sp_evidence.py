"""Signal peptide labels by UniProt evidence code, and released-model test performance per evidence group.
Needs loc/data/raw/sprot.tsv.gz and loc/models/run1_150m/test_preds.npz (from bootstrap_loc.py). Writes loc/release/sp_evidence.json.
Groups: exp = ECO:0000269 (experimental), seq = ECO:0000255 (sequence-model evidence, typically prediction tools), other = any other code or none."""
import gzip, json, math, re
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
root = Path(__file__).resolve().parents[1]
ev = {}
with gzip.open(root / "data/raw/sprot.tsv.gz", "rt", encoding="utf-8") as fh:
    h = fh.readline().rstrip("\n").split("\t"); si = h.index("Signal peptide")
    for line in fh:
        x = line.rstrip("\n").split("\t")
        if x[si]:
            c = set(re.findall(r"ECO:\d+", x[si])); ev[x[0]] = "exp" if "ECO:0000269" in c else ("seq" if "ECO:0000255" in c else "other")
out = {"definition": __doc__, "label_counts": {}}
for s in ("train", "dev", "test"):
    d = [json.loads(l) for l in open(root / f"data/{s}.jsonl")]
    out["label_counts"][s] = dict(Counter(ev[x["id"]] for x in d if x["signal_peptide"]))
te = [json.loads(l) for l in open(root / "data/test.jsonl")]; z = np.load(root / "models/run1_150m/test_preds.npz"); S = z["S"]
assert list(z["ids"]) == [x["id"] for x in te]
thr = json.load(open(root / "release/config.json"))["sp_threshold"]
y = np.array([x["signal_peptide"] for x in te]).astype(int); g = np.array([ev[x["id"]] if x["signal_peptide"] else "neg" for x in te]); pred = S > thr
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return c - w, c + w
out["test_by_group"] = {}
for grp in ("exp", "seq", "other"):
    m = g == grp; sub = (g == "neg") | m; k, n = int(pred[m].sum()), int(m.sum()); lo, hi = wilson(k, n)
    out["test_by_group"][grp] = {"n_pos": n, "recall": k / n, "recall_95ci_wilson": [lo, hi], "auroc_vs_all_test_negatives": float(roc_auc_score(y[sub], S[sub]))}
out["test_false_positive_rate"] = float(pred[y == 0].mean()); out["threshold"] = thr
(root / "release/sp_evidence.json").write_text(json.dumps(out, indent=2)); print(json.dumps({k: v for k, v in out.items() if k != "definition"}, indent=1))
