"""Test-set bootstrap CIs for the released NER model (ner/release, local files; no hub download).
Same pipeline path as eval_v2.py (the main results table in the card). Resamples whole abstracts (PMIDs) with
replacement, 2000 times; entity-level micro P/R/F1 from per-abstract counts (seqeval-style entity extraction),
95% percentile intervals. Point estimates are checked against seqeval. Writes ner/release/bootstrap_ci.json."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from transformers import pipeline
from seqeval.metrics import f1_score
from seqeval.metrics.sequence_labeling import get_entities
root = Path(__file__).resolve().parents[1]; rel = root / "release"; B, SEED = 2000, 13
pipe = pipeline("token-classification", model=str(rel), aggregation_strategy="simple", device=0)
def run(corpus, keep=None, pipe=pipe, raw=False):
    test = [json.loads(l) for l in open(root / "data" / corpus / "test.jsonl", encoding="utf-8")]
    out = pipe([e["text"] for e in test], batch_size=32)
    f = (lambda tags: [t if t == "O" or t[2:] in keep else "O" for t in tags]) if keep else (lambda tags: tags)
    P, G, docs = [], [], []
    for e, ents in zip(test, out):
        tags = ["O"] * len(e["tokens"])
        for g in ents:
            first = True
            for i, (s, t) in enumerate(e["offsets"]):
                if t > g["start"] and s < g["end"]: tags[i] = ("B-" if first else "I-") + g["entity_group"]; first = False
        P.append(f(tags)); G.append(f(e["tags"])); docs.append(e["pmid"])
    types = sorted({t for t, _, _ in get_entities(G)}) ; keys = ["all"] + types
    cnt = defaultdict(lambda: {k: np.zeros(3) for k in keys})  # per doc: tp, n_pred, n_gold
    for d, p, g in zip(docs, P, G):
        pe, ge = set(get_entities(p)), set(get_entities(g))
        for k in keys:
            sel = (lambda s: s) if k == "all" else (lambda s: {x for x in s if x[0] == k})
            cnt[d][k] += [len(sel(pe) & sel(ge)), len(sel(pe)), len(sel(ge))]
    ids = list(cnt); M = {k: np.array([cnt[d][k] for d in ids]) for k in keys}
    def prf(c): tp, npred, ngold = c; p = tp / npred if npred else 0.0; r = tp / ngold if ngold else 0.0; return p, r, (2 * p * r / (p + r) if p + r else 0.0)
    if raw: return ids, M
    point = {k: prf(M[k].sum(0)) for k in keys}
    assert abs(point["all"][2] - f1_score(G, P)) < 1e-9, (point["all"][2], f1_score(G, P))
    rng = np.random.default_rng(SEED); draws = {k: [] for k in keys}
    for _ in range(B):
        ii = rng.integers(0, len(ids), len(ids))
        for k in keys: draws[k].append(prf(M[k][ii].sum(0)))
    res = {}
    for k in keys:
        D = np.array(draws[k]); res[k] = {m: {"point": float(point[k][j]), "lo": float(np.percentile(D[:, j], 2.5)), "hi": float(np.percentile(D[:, j], 97.5))} for j, m in enumerate(("precision", "recall", "f1"))}
    return {"n_abstracts": len(ids), "n_sentences": len(test), "ci": res}
out = {"method": f"bootstrap over test abstracts (PMIDs), {B} resamples, seed {SEED}, 95% percentile interval; entity-level micro P/R/F1 (seqeval-style, exact span+type); pipeline path as eval_v2.py",
       "bc5cdr_test": run("bc5cdr"), "ncbi_disease_test_disease_only": run("ncbi_disease", {"Disease"})}
# Paired v1 (run1, BC5CDR only) vs v2 (released) on BC5CDR test: same resampled abstracts for both models.
ids2, M2 = run("bc5cdr", raw=True)
ids1, M1 = run("bc5cdr", pipe=pipeline("token-classification", model=str(root / "models" / "run1"), aggregation_strategy="simple", device=0), raw=True)
assert ids1 == ids2
def f1c(c): tp, npred, ngold = c; return 2 * tp / (npred + ngold)
rng = np.random.default_rng(SEED); dd = []
for _ in range(B):
    ii = rng.integers(0, len(ids1), len(ids1)); dd.append(f1c(M2["all"][ii].sum(0)) - f1c(M1["all"][ii].sum(0)))
dd = np.array(dd); d0 = f1c(M2["all"].sum(0)) - f1c(M1["all"].sum(0))
out["paired_v2_minus_v1_bc5cdr_f1"] = {"point": float(d0), "lo": float(np.percentile(dd, 2.5)), "hi": float(np.percentile(dd, 97.5)), "share_of_resamples_v2_worse": float((dd < 0).mean()),
    "note": "v1 = run1 (BC5CDR only, seed 13), v2 = released run2a seed 13; test-sampling uncertainty only, single seed each"}
print("paired v2-v1 BC5CDR F1", out["paired_v2_minus_v1_bc5cdr_f1"])
(rel / "bootstrap_ci.json").write_text(json.dumps(out, indent=2))
for c in ("bc5cdr_test", "ncbi_disease_test_disease_only"):
    for k, v in out[c]["ci"].items(): print(c, k, " ".join(f"{m} {x['point']:.3f} [{x['lo']:.3f}, {x['hi']:.3f}]" for m, x in v.items()))
