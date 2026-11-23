"""Test-set bootstrap CIs for NER v3 (local model dir; no hub download), pipeline path as eval_v3.py.
Resamples whole abstracts (PMIDs) with replacement, 2000 times, seed 13; entity-level micro P/R/F1 from per-abstract counts;
95% percentile intervals; point estimates checked against seqeval. Per corpus only its own types; X words set to O.
Also: paired v3 minus released v2 on BC5CDR (Chemical+Disease) and NCBI-disease (Disease), same resampled abstracts.
Usage: python bootstrap_ner3.py ner/models/v3a  -> writes <model>/bootstrap_ci.json"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from transformers import pipeline
from seqeval.metrics import f1_score
from seqeval.metrics.sequence_labeling import get_entities
root = Path(__file__).resolve().parents[1]; D = root / "data3"; B, SEED = 2000, 13
TYPES = json.loads((D / "meta.json").read_text())["types"]
mdir = Path(sys.argv[1])
def P_(m): return pipeline("token-classification", model=str(m), aggregation_strategy="simple", device=0)

def counts(pipe, corpus, keep):
    test = [json.loads(l) for l in open(D / corpus / "test.jsonl", encoding="utf-8")]
    out = pipe([e["text"] for e in test], batch_size=32); P, G, docs = [], [], []
    for e, ents in zip(test, out):
        tags = ["O"] * len(e["tokens"])
        for g in ents:
            first = True
            for i, (s, t) in enumerate(e["offsets"]):
                if t > g["start"] and s < g["end"]: tags[i] = ("B-" if first else "I-") + g["entity_group"]; first = False
        gold = list(e["tags"])
        for i in range(len(gold)):
            if gold[i] == "X": gold[i] = "O"; tags[i] = "O"
            if tags[i] != "O" and tags[i][2:] not in keep: tags[i] = "O"
            if gold[i] != "O" and gold[i][2:] not in keep: gold[i] = "O"
        P.append(tags); G.append(gold); docs.append(e["pmid"])
    keys = ["all"] + sorted(keep)
    cnt = defaultdict(lambda: {k: np.zeros(3) for k in keys})
    for d, p, g in zip(docs, P, G):
        pe, ge = set(get_entities(p)), set(get_entities(g))
        for k in keys:
            sel = (lambda s: s) if k == "all" else (lambda s, k=k: {x for x in s if x[0] == k})
            cnt[d][k] += [len(sel(pe) & sel(ge)), len(sel(pe)), len(sel(ge))]
    ids = list(cnt); M = {k: np.array([cnt[d][k] for d in ids]) for k in keys}
    return ids, M, f1_score(G, P), len(test)

def prf(c):
    tp, npred, ngold = c; p = tp / npred if npred else 0.0; r = tp / ngold if ngold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)

def ci(ids, M, f1_check, n_sent):
    point = {k: prf(M[k].sum(0)) for k in M}
    assert abs(point["all"][2] - f1_check) < 1e-9, (point["all"][2], f1_check)
    rng = np.random.default_rng(SEED); draws = {k: [] for k in M}
    for _ in range(B):
        ii = rng.integers(0, len(ids), len(ids))
        for k in M: draws[k].append(prf(M[k][ii].sum(0)))
    res = {}
    for k in M:
        A = np.array(draws[k]); res[k] = {m: {"point": float(point[k][j]), "lo": float(np.percentile(A[:, j], 2.5)), "hi": float(np.percentile(A[:, j], 97.5))} for j, m in enumerate(("precision", "recall", "f1"))}
    return {"n_abstracts": len(ids), "n_sentences": n_sent, "ci": res}

p3 = P_(mdir); out = {"method": f"bootstrap over test abstracts (PMIDs), {B} resamples, seed {SEED}, 95% percentile interval; entity-level micro P/R/F1; pipeline path as eval_v3.py"}
raw3 = {}
for c in TYPES:
    raw3[c] = counts(p3, c, set(TYPES[c])); out[c] = ci(*raw3[c])
    for k, v in out[c]["ci"].items(): print(c, k, "F1 %.3f [%.3f, %.3f]" % (v["f1"]["point"], v["f1"]["lo"], v["f1"]["hi"]), flush=True)
p2 = P_(root / "release")
f1c = lambda c: 2 * c[0] / (c[1] + c[2])
out["paired_v3_minus_v2"] = {}
for c in ("bc5cdr", "ncbi_disease"):
    ids3, M3, _, _ = raw3[c]; ids2, M2, _, _ = counts(p2, c, set(TYPES[c])); assert ids2 == ids3
    rng = np.random.default_rng(SEED); dd = []
    for _ in range(B):
        ii = rng.integers(0, len(ids3), len(ids3)); dd.append(f1c(M3["all"][ii].sum(0)) - f1c(M2["all"][ii].sum(0)))
    dd = np.array(dd)
    out["paired_v3_minus_v2"][c] = {"point": float(f1c(M3["all"].sum(0)) - f1c(M2["all"].sum(0))), "lo": float(np.percentile(dd, 2.5)), "hi": float(np.percentile(dd, 97.5)),
                                    "share_of_resamples_v3_worse": float((dd < 0).mean())}
    print("paired v3-v2", c, out["paired_v3_minus_v2"][c], flush=True)
(mdir / "bootstrap_ci.json").write_text(json.dumps(out, indent=2)); print("BOOT3_DONE")
