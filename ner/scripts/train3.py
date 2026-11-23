"""NER v3: one standard token-classification head (O + B/I for 6 types = 13 labels), trained on corpora that each annotate
only some types, with a partial-label loss:
  - a word tagged B-T / I-T must be exactly that label;
  - a word tagged O in corpus c means "none of c's types", so any label of a type c does NOT annotate is still allowed
    (e.g. an O word in NLM-Gene, which only labels genes, may be O or a Chemical/Disease/Species/... label);
  - a word tagged X (NLM-Gene Domain/Other) is ignored.
  loss = -log(sum of softmax probabilities over the allowed labels), first subword of each word only.
  --naive treats every O as plain O (the v2-style objective), for comparison.
The output is a standard AutoModelForTokenClassification (works with transformers pipelines, ONNX, transformers.js).

Evaluation (seqeval entity-level, word level, same tokenization as v2): for each corpus, predictions of types the corpus does not
annotate are set to O, and X words are set to O in both gold and prediction. Checkpoint = best mean VALIDATION F1 over the
four corpora; each test set is scored once after selection.
Usage: python train3.py --base microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext --out ner/models/v3a"""
import argparse, json, random, time
from pathlib import Path
import torch
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--seed", type=int, default=13); ap.add_argument("--max_len", type=int, default=256)
ap.add_argument("--naive", action="store_true"); ap.add_argument("--max_steps", type=int, default=0, help="smoke test only")
a = ap.parse_args()
random.seed(a.seed); torch.manual_seed(a.seed)
root = Path(__file__).resolve().parents[1]; D = root / "data3"
meta = json.loads((D / "meta.json").read_text()); TYPES = meta["types"]; CORP = list(TYPES)
ALL = ["Chemical", "Disease", "Gene", "Species", "Variant", "CellLine"]
labels = ["O"] + [f"{p}-{t}" for t in ALL for p in "BI"]; l2i = {l: i for i, l in enumerate(labels)}; i2l = dict(enumerate(labels))
def load(c, s): return [json.loads(l) for l in open(D / c / f"{s}.jsonl", encoding="utf-8")]
tr = [x for c in CORP for x in load(c, "train")]; dv = {c: load(c, "dev") for c in CORP}; te = {c: load(c, "test") for c in CORP}
tok = AutoTokenizer.from_pretrained(a.base)
O_ALLOWED = {c: [0] + [l2i[f"{p}-{t}"] for t in ALL if t not in TYPES[c] for p in "BI"] for c in CORP}

def enc(ex):
    e = tok(ex["tokens"], is_split_into_words=True, truncation=True, max_length=a.max_len)
    wid, prev, allow = e.word_ids(), None, []
    for w in wid:
        row = [0] * len(labels)
        if w is not None and w != prev:
            t = ex["tags"][w]
            if t == "O": idx = [0] if a.naive else O_ALLOWED[ex["corpus"]]
            elif t == "X": idx = []
            else: idx = [l2i[t]]
            for i in idx: row[i] = 1
        allow.append(row); prev = w
    return {"input_ids": e["input_ids"], "attention_mask": e["attention_mask"], "token_type_ids": e.get("token_type_ids", [0] * len(wid)), "allow": allow, "word_ids": wid}
tr_e = [enc(x) for x in tr]; dv_e = {c: [enc(x) for x in v] for c, v in dv.items()}; te_e = {c: [enc(x) for x in v] for c, v in te.items()}
print("train sentences", len(tr_e), {c: sum(x["corpus"] == c for x in tr) for c in CORP}, flush=True)

def collate(b):
    L = max(len(x["input_ids"]) for x in b); pad = tok.pad_token_id
    f = lambda k, v: torch.tensor([x[k] + [v] * (L - len(x[k])) for x in b])
    return {"input_ids": f("input_ids", pad), "attention_mask": f("attention_mask", 0), "token_type_ids": f("token_type_ids", 0),
            "allow": torch.tensor([x["allow"] + [[0] * len(labels)] * (L - len(x["allow"])) for x in b], dtype=torch.bool)}
def batches(data, bs, shuffle):
    idx = list(range(len(data)))
    if shuffle: random.shuffle(idx)
    for i in range(0, len(idx), bs): yield collate([data[j] for j in idx[i:i + bs]]), idx[i:i + bs]

dev = "cuda"
model = AutoModelForTokenClassification.from_pretrained(a.base, num_labels=len(labels), id2label=i2l, label2id=l2i).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
steps = a.epochs * ((len(tr_e) + a.bs - 1) // a.bs)
sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

def partial_loss(logits, allow):
    lp = logits.float().log_softmax(-1); m = allow.any(-1)
    return -(lp.masked_fill(~allow, -1e4).logsumexp(-1)[m]).mean()

@torch.no_grad()
def evaluate(c, data_e, data_raw, report=False):
    model.eval(); preds, golds = [], []
    for b, idx in batches(data_e, 64, False):
        x = {k: v.to(dev) for k, v in b.items() if k != "allow"}
        with torch.autocast("cuda", dtype=torch.bfloat16): p = model(**x).logits.argmax(-1).cpu().tolist()
        for row, j in zip(p, idx):
            wid, ex = data_e[j]["word_ids"], data_raw[j]
            pr = ["O"] * len(ex["tokens"]); seen = set()
            for pos, w in enumerate(wid):
                if w is not None and w not in seen: pr[w] = i2l[row[pos]]; seen.add(w)
            g = list(ex["tags"])
            for i, t in enumerate(g):
                if t == "X": g[i] = "O"; pr[i] = "O"
                elif pr[i] != "O" and pr[i][2:] not in TYPES[c]: pr[i] = "O"
            preds.append(pr); golds.append(g)
    model.train()
    r = {"f1": f1_score(golds, preds), "precision": precision_score(golds, preds), "recall": recall_score(golds, preds)}
    if report: r["per_type"] = {k: {m: float(v[m]) for m in ("precision", "recall", "f1-score", "support")} for k, v in classification_report(golds, preds, output_dict=True).items() if k in ALL}
    return r

best, best_state, hist, step = -1, None, [], 0
model.train(); t0 = time.time()
for ep in range(a.epochs):
    tot = 0
    for b, _ in batches(tr_e, a.bs, True):
        allow = b.pop("allow").to(dev); x = {k: v.to(dev) for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16): lg = model(**x).logits
        loss = partial_loss(lg, allow)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(); tot += loss.item(); step += 1
        if a.max_steps and step >= a.max_steps: break
    per = {c: evaluate(c, dv_e[c], dv[c]) for c in CORP}
    m = {"epoch": ep + 1, "train_loss": tot, "per_corpus_val": per, "f1": sum(v["f1"] for v in per.values()) / len(per), "minutes": (time.time() - t0) / 60}; hist.append(m)
    print(json.dumps(m), flush=True)
    if m["f1"] > best: best = m["f1"]; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if a.max_steps and step >= a.max_steps: break
model.load_state_dict(best_state)
res = {"objective": "naive (O = O everywhere)" if a.naive else "partial-label", "types": TYPES, "base": a.base, "args": vars(a), "val_history": hist,
       "best_val_f1_mean_over_corpora": best, "test": {c: evaluate(c, te_e[c], te[c], report=True) for c in CORP},
       "checkpoint_selection": "best mean validation entity F1 over the four corpora (seqeval); each test set scored once after selection",
       "train_minutes": (time.time() - t0) / 60}
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
if not a.max_steps: model.save_pretrained(out); tok.save_pretrained(out)
(out / "results.json").write_text(json.dumps(res, indent=2))
print("FINAL", json.dumps({c: round(v["f1"], 4) for c, v in res["test"].items()}), flush=True); print("TRAIN_DONE", flush=True)
