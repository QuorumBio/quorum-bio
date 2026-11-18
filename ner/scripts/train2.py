"""Fine-tune a token-classification model. Checkpoint chosen on VALIDATION entity F1 (seqeval), test reported once at the end.
Usage: python train.py --corpus bc5cdr --base microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext --out ner/models/run1"""
import argparse, json, random, time
from pathlib import Path
import torch
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, get_linear_schedule_with_warmup
from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--corpora", default="bc5cdr,ncbi_disease"); ap.add_argument("--mask_o_in", default="", help="comma list of corpora whose O labels are ignored in the loss (they lack some entity types)"); ap.add_argument("--base", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--seed", type=int, default=13); ap.add_argument("--max_len", type=int, default=256)
a = ap.parse_args()
random.seed(a.seed); torch.manual_seed(a.seed)
root = Path(__file__).resolve().parents[1]
CORP = a.corpora.split(","); MASK = set(filter(None, a.mask_o_in.split(",")))
def load(c, s):
    d = [json.loads(l) for l in open(root / "data" / c / f"{s}.jsonl", encoding="utf-8")]
    for x in d: x["corpus"] = c
    return d
tr = [x for c in CORP for x in load(c, "train")]
dv = {c: load(c, "dev") for c in CORP}; te = {c: load(c, "test") for c in CORP}
labels = sorted({t for e in tr for t in e["tags"]}, key=lambda x: (x != "O", x))
l2i = {l: i for i, l in enumerate(labels)}; i2l = {i: l for l, i in l2i.items()}
try:
    info = HfApi().model_info(a.base); print("BASE MODEL LICENSE:", getattr(info.card_data, "license", None), flush=True)
except Exception as ex: print("license lookup failed", ex)
tok = AutoTokenizer.from_pretrained(a.base)

def enc(ex):
    e = tok(ex["tokens"], is_split_into_words=True, truncation=True, max_length=a.max_len)
    wid, prev, lab = e.word_ids(), None, []
    for w in wid:
        t = None if w is None or w == prev else ex["tags"][w]
        lab.append(-100 if t is None or (t == "O" and ex["corpus"] in MASK) else l2i[t]); prev = w
    e["labels"] = lab; e["word_ids"] = wid
    return e
def prep(data): return [enc(x) for x in data]
tr_e = prep(tr); dv_e = {c: prep(v) for c, v in dv.items()}; te_e = {c: prep(v) for c, v in te.items()}
coll = DataCollatorForTokenClassification(tok)
def batches(data, bs, shuffle):
    idx = list(range(len(data)));
    if shuffle: random.shuffle(idx)
    for i in range(0, len(idx), bs):
        yield [{k: v for k, v in data[j].items() if k != "word_ids"} for j in idx[i:i+bs]], idx[i:i+bs]

dev = "cuda"
model = AutoModelForTokenClassification.from_pretrained(a.base, num_labels=len(labels), id2label=i2l, label2id=l2i).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
steps = a.epochs * ((len(tr_e) + a.bs - 1) // a.bs)
sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

@torch.no_grad()
def evaluate(data_e, data_raw):
    model.eval(); preds, golds = [], []
    for b, idx in batches(data_e, 64, False):
        x = {k: v.to(dev) for k, v in coll(b).items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(**{k: v for k, v in x.items() if k != "labels"}).logits
        p = lg.argmax(-1).cpu().tolist()
        for row, j in zip(p, idx):
            wid, ex = data_e[j]["word_ids"], data_raw[j]
            pr = ["O"] * len(ex["tokens"]); seen = set()
            for pos, w in enumerate(wid):
                if w is not None and w not in seen: pr[w] = i2l[row[pos]]; seen.add(w)
            preds.append(pr); golds.append(ex["tags"])
    model.train()
    return {"f1": f1_score(golds, preds), "precision": precision_score(golds, preds), "recall": recall_score(golds, preds)}

best, best_state, hist = -1, None, []
model.train(); t0 = time.time()
for ep in range(a.epochs):
    tot = 0
    for b, _ in batches(tr_e, a.bs, True):
        x = {k: v.to(dev) for k, v in coll(b).items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**x).loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(); tot += loss.item()
    per = {c: evaluate(dv_e[c], dv[c]) for c in CORP}
    m = {"epoch": ep + 1, "train_loss": tot, "per_corpus_val": per, "f1": sum(v["f1"] for v in per.values()) / len(per)}; hist.append(m)
    print(json.dumps(m), flush=True)
    if m["f1"] > best: best = m["f1"]; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
model.load_state_dict(best_state)
res = {"corpora": CORP, "mask_o_in": sorted(MASK), "base": a.base, "args": vars(a), "val_history": hist, "best_val_f1_mean_over_corpora": best, "test": {c: evaluate(te_e[c], te[c]) for c in CORP},
       "checkpoint_selection": "best mean validation entity-level F1 across corpora (seqeval); test evaluated once after selection", "train_minutes": (time.time() - t0) / 60}
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
model.save_pretrained(out); tok.save_pretrained(out); (out / "results.json").write_text(json.dumps(res, indent=2))
print("FINAL", json.dumps(res["test"]), flush=True); print("TRAIN_DONE", flush=True)
