"""Test-set bootstrap CIs for the released localization model (loc/release, local files; no hub download).
1) Re-predict the test split with the released weights (bf16 autocast on GPU, same as train_loc.py).
2) Check that the point estimates reproduce loc/release/train_results_150m.json.
3) Cluster bootstrap: resample MMseqs2 test clusters (30% identity) with replacement, 2000 times, and report
   95% percentile intervals. Thresholds stay fixed at the dev-chosen values (no re-tuning).
Writes loc/models/run1_150m/test_preds.npz and loc/release/bootstrap_ci.json.
Optional args (defaults reproduce the 150M release): <release dir> <preds .npz> <train results json in release dir> [<other model preds .npz> for a paired difference]."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
root = Path(__file__).resolve().parents[1]; A = sys.argv[1:]
rel = Path(A[0]) if A else root / "release"
sys.dont_write_bytecode = True; sys.path.insert(0, str(rel)); from quorum_loc import load, trunc
B, SEED = 2000, 13
m = load(rel, device="cuda"); C = m.meta["classes"]; thr = np.array(m.meta["thresholds"]); spt = m.meta["sp_threshold"]
te = [json.loads(l) for l in open(root / "data" / "test.jsonl")]
clu = pd.read_parquet(root / "dataset" / "data" / "test.parquet").set_index("accession")["cluster_id"]
cid = np.array([clu[x["id"]] for x in te])
Y = np.array([[c in x["locations"] for c in C] for x in te]).astype(int); YS = np.array([x["signal_peptide"] for x in te]).astype(int)
pf = Path(A[1]) if len(A) > 1 else root / "models" / "run1_150m" / "test_preds.npz"
if pf.exists(): z = np.load(pf); P, S = z["P"], z["S"]
else:
    P, S = np.zeros((len(te), len(C))), np.zeros(len(te))
    idx = sorted(range(len(te)), key=lambda i: len(te[i]["sequence"]))
    with torch.no_grad():
        for k in range(0, len(idx), 8):
            ch = idx[k:k + 8]; t = m.tok([trunc(te[i]["sequence"]) for i in ch], return_tensors="pt", padding=True, truncation=True, max_length=1022)
            with torch.autocast("cuda", dtype=torch.bfloat16): lg, sg = m(t["input_ids"].cuda(), t["attention_mask"].cuda())
            P[ch] = torch.sigmoid(lg.float()).cpu().numpy(); S[ch] = torch.sigmoid(sg.float()).cpu().numpy()
    np.savez(pf, P=P, S=S, ids=np.array([x["id"] for x in te]))

def metrics(Y, P, YS, S):
    out = {}
    for j, c in enumerate(C):
        if 0 < Y[:, j].sum() < len(Y):
            out[c] = {"auroc": roc_auc_score(Y[:, j], P[:, j]), "ap": average_precision_score(Y[:, j], P[:, j]), "f1": f1_score(Y[:, j], P[:, j] > thr[j], zero_division=0)}
    out["macro_auroc"] = np.mean([out[c]["auroc"] for c in C if c in out]); out["macro_ap"] = np.mean([out[c]["ap"] for c in C if c in out])
    out["macro_f1"] = np.mean([out[c]["f1"] for c in C if c in out])
    out["rare_ap"] = np.mean([out[c]["ap"] for c in ("Golgi apparatus", "Lysosome/Vacuole", "Peroxisome") if c in out])
    out["sp_auroc"] = roc_auc_score(YS, S); out["sp_mcc"] = matthews_corrcoef(YS, S > spt)
    return out
def flat(d): return {f"{k}.{kk}": v for k, x in d.items() if isinstance(x, dict) for kk, v in x.items()} | {k: x for k, x in d.items() if not isinstance(x, dict)}
point = flat(metrics(Y, P, YS, S))
ref = json.load(open(rel / (A[2] if len(A) > 2 else "train_results_150m.json")))["test"]
check = {"macro_auroc": (point["macro_auroc"], ref["macro_auroc_loc"]), "macro_f1": (point["macro_f1"], ref["macro_f1_loc"]),
         "sp_auroc": (point["sp_auroc"], ref["signal_peptide"]["auroc"]), "sp_mcc": (point["sp_mcc"], ref["signal_peptide"]["mcc_at_dev_threshold"])}
print("reproduce (ours, published):", {k: (round(a, 4), round(b, 4)) for k, (a, b) in check.items()})
uc = np.unique(cid); members = {c: np.where(cid == c)[0] for c in uc}; rng = np.random.default_rng(SEED); draws = {k: [] for k in point}
for b in range(B):
    ii = np.concatenate([members[c] for c in rng.choice(uc, len(uc), replace=True)])
    try: r = flat(metrics(Y[ii], P[ii], YS[ii], S[ii]))
    except ValueError: continue
    for k in point: draws[k].append(r.get(k, np.nan))
ci = {k: {"point": float(point[k]), "lo": float(np.nanpercentile(v, 2.5)), "hi": float(np.nanpercentile(v, 97.5))} for k, v in draws.items()}
res = {"method": f"cluster bootstrap over {len(uc)} MMseqs2 test clusters (30% id), {B} resamples, seed {SEED}, 95% percentile interval; dev-chosen thresholds fixed",
       "n_test": len(te), "n_clusters": int(len(uc)), "reproduce_check": {k: {"ours": a, "published": b} for k, (a, b) in check.items()}, "ci": ci}
if len(A) > 3:  # paired difference (this model minus the other) on the same resampled clusters
    zo = np.load(A[3]); assert list(zo["ids"]) == [x["id"] for x in te]; Po, So = zo["P"], zo["S"]
    keys = ["macro_auroc", "macro_ap", "macro_f1", "rare_ap", "sp_auroc", "sp_mcc"]
    d0 = {k: point[k] - flat(metrics(Y, Po, YS, So))[k] for k in keys}; rng = np.random.default_rng(SEED); dd = {k: [] for k in keys}
    for b in range(B):
        ii = np.concatenate([members[c] for c in rng.choice(uc, len(uc), replace=True)])
        try: r1, r2 = flat(metrics(Y[ii], P[ii], YS[ii], S[ii])), flat(metrics(Y[ii], Po[ii], YS[ii], So[ii]))
        except ValueError: continue
        for k in keys: dd[k].append(r1[k] - r2[k])
    res["paired_difference_vs_" + Path(A[3]).parent.name] = {k: {"point": float(d0[k]), "lo": float(np.percentile(dd[k], 2.5)), "hi": float(np.percentile(dd[k], 97.5)), "share_this_better": float(np.mean(np.array(dd[k]) > 0))} for k in keys}
    print("paired:", json.dumps(res["paired_difference_vs_" + Path(A[3]).parent.name]))
(rel / "bootstrap_ci.json").write_text(json.dumps(res, indent=2))
for k, v in ci.items(): print(f"{k:40s} {v['point']:.3f}  [{v['lo']:.3f}, {v['hi']:.3f}]")
