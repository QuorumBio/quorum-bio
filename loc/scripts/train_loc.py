"""Fine-tune ESM-2 for multi-label subcellular localization (10 classes) + signal-peptide presence. Homology-partitioned splits from prepare_uniprot.py.
Selection on dev (mean of macro-AUROC over locations and SP AUROC); per-class thresholds chosen on dev; test scored once.
Usage: python train_loc.py --base facebook/esm2_t30_150M_UR50D --out loc/models/run1 [--smoke]"""
import argparse, json, random, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, average_precision_score
ap = argparse.ArgumentParser()
ap.add_argument("--base", default="facebook/esm2_t30_150M_UR50D"); ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--seed", type=int, default=13); ap.add_argument("--smoke", action="store_true")
a = ap.parse_args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
root = Path(__file__).resolve().parents[1]
CLASSES = ["Nucleus", "Cytoplasm", "Extracellular", "Mitochondrion", "Cell membrane", "Endoplasmic reticulum", "Plastid", "Golgi apparatus", "Lysosome/Vacuole", "Peroxisome"]
load = lambda s: [json.loads(l) for l in open(root / "data" / f"{s}.jsonl")]
tr, dv, te = load("train"), load("dev"), load("test")
if a.smoke: tr, dv, te = tr[:400], dv[:100], te[:100]
tok = AutoTokenizer.from_pretrained(a.base); enc = AutoModel.from_pretrained(a.base); enc.gradient_checkpointing_enable()
H = enc.config.hidden_size
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.enc = enc; s.drop = nn.Dropout(0.1); s.loc = nn.Linear(H, len(CLASSES)); s.sp = nn.Linear(2 * H, 1)
    def forward(s, ids, mask):
        h = s.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype); pooled = (h * m).sum(1) / m.sum(1)
        pooled = s.drop(pooled)
        nterm = s.drop(h[:, 1:31].mean(1))  # signal peptides sit at the N-terminus
        return s.loc(pooled), s.sp(torch.cat([pooled, nterm], -1))[:, 0]
dev = "cuda"; net = Net().to(dev)
def trunc(seq): return seq if len(seq) <= 1000 else seq[:600] + seq[-400:]
def batch(items):
    t = tok([trunc(x["sequence"]) for x in items], return_tensors="pt", padding=True, truncation=True, max_length=1022)
    y = torch.tensor([[float(c in x["locations"]) for c in CLASSES] for x in items]); s = torch.tensor([float(x["signal_peptide"]) for x in items])
    return t["input_ids"].to(dev), t["attention_mask"].to(dev), y.to(dev), s.to(dev)
def order(data, shuffle):
    idx = sorted(range(len(data)), key=lambda i: len(data[i]["sequence"]))  # length-bucketed
    chunks = [idx[i:i + a.bs] for i in range(0, len(idx), a.bs)]
    if shuffle: random.shuffle(chunks)
    return chunks
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
steps = a.epochs * len(order(tr, False)); sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
pos = np.array([[c in x["locations"] for c in CLASSES] for x in tr]).mean(0); pw = torch.tensor(np.clip((1 - pos) / np.maximum(pos, 1e-3), 1, 10), dtype=torch.float, device=dev)
bl, bs_ = nn.BCEWithLogitsLoss(pos_weight=pw), nn.BCEWithLogitsLoss()
@torch.no_grad()
def predict(data):
    net.eval(); P, S = np.zeros((len(data), len(CLASSES))), np.zeros(len(data))
    for ch in order(data, False):
        ids, mask, _, _ = batch([data[i] for i in ch])
        with torch.autocast("cuda", dtype=torch.bfloat16): lg, sg = net(ids, mask)
        P[ch] = torch.sigmoid(lg.float()).cpu().numpy(); S[ch] = torch.sigmoid(sg.float()).cpu().numpy()
    net.train(); return P, S
def auroc(y, p):
    ok = [j for j in range(y.shape[1]) if 0 < y[:, j].sum() < len(y)]
    return {CLASSES[j]: float(roc_auc_score(y[:, j], p[:, j])) for j in ok}
def ys(data): return np.array([[c in x["locations"] for c in CLASSES] for x in data]).astype(int), np.array([x["signal_peptide"] for x in data]).astype(int)
def score(data, P, S): 
    Y, YS = ys(data); au = auroc(Y, P); sp_auc = float(roc_auc_score(YS, S)) if 0 < YS.sum() < len(YS) else float("nan")
    return au, sp_auc
best, best_state, hist = -1, None, []; t0 = time.time(); net.train()
for ep in range(a.epochs):
    tot = 0
    for ch in order(tr, True):
        ids, mask, y, s = batch([tr[i] for i in ch])
        with torch.autocast("cuda", dtype=torch.bfloat16): lg, sg = net(ids, mask)
        loss = bl(lg.float(), y) + bs_(sg.float(), s)
        loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(); tot += loss.item()
    P, S = predict(dv); au, sp_auc = score(dv, P, S); sel = float(np.nanmean([np.mean(list(au.values())), sp_auc]))
    m = {"epoch": ep + 1, "train_loss": tot, "dev_macro_auroc_loc": float(np.mean(list(au.values()))), "dev_sp_auroc": sp_auc, "select": sel, "minutes": (time.time() - t0) / 60}
    hist.append(m); print(json.dumps(m), flush=True)
    if sel > best: best = sel; best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
net.load_state_dict(best_state)
Pd, Sd = predict(dv); Yd, YSd = ys(dv)
thr = []
for j in range(len(CLASSES)):
    grid = np.linspace(0.05, 0.95, 19); f = [f1_score(Yd[:, j], Pd[:, j] > g, zero_division=0) for g in grid]; thr.append(float(grid[int(np.argmax(f))]))
sp_thr = float(max(np.linspace(0.05, 0.95, 19), key=lambda g: matthews_corrcoef(YSd, Sd > g)))
Pt, St = predict(te); Yt, YSt = ys(te); au, sp_auc = score(te, Pt, St)
per = {}
for j, c in enumerate(CLASSES):
    if 0 < Yt[:, j].sum() < len(Yt):
        per[c] = {"auroc": float(roc_auc_score(Yt[:, j], Pt[:, j])), "avg_precision": float(average_precision_score(Yt[:, j], Pt[:, j])), "f1_at_dev_threshold": float(f1_score(Yt[:, j], Pt[:, j] > thr[j], zero_division=0)), "threshold": thr[j], "n_pos": int(Yt[:, j].sum())}
res = {"base": a.base, "args": vars(a), "val_history": hist, "selection": "best dev mean(macro AUROC locations, SP AUROC); thresholds on dev; test scored once", "train_minutes": (time.time() - t0) / 60,
       "test": {"n": len(te), "per_class": per, "macro_auroc_loc": float(np.mean([v["auroc"] for v in per.values()])), "macro_f1_loc": float(np.mean([v["f1_at_dev_threshold"] for v in per.values()])),
                "signal_peptide": {"auroc": sp_auc, "mcc_at_dev_threshold": float(matthews_corrcoef(YSt, St > sp_thr)), "threshold": sp_thr, "n_pos": int(YSt.sum())}}}
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
torch.save({"state_dict": net.state_dict(), "classes": CLASSES, "thresholds": thr, "sp_threshold": sp_thr, "base": a.base}, out / "model.pt"); tok.save_pretrained(out)
(out / "results.json").write_text(json.dumps(res, indent=2)); print("FINAL", json.dumps(res["test"]["signal_peptide"]), res["test"]["macro_auroc_loc"], flush=True); print("TRAIN_LOC_DONE")
