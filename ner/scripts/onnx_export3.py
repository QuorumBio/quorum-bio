"""Export a v3 model to ONNX (fp32 + dynamic int8), check parity with PyTorch, and score each v3 test set through onnxruntime (CPU).
Scoring as in train3.py: per corpus only its own types; X words set to O.
Usage: python onnx_export3.py ner/models/v3a ner/release3   -> writes <out>/onnx/*.onnx and <out>/onnx_results.json"""
import json, sys, time
from pathlib import Path
import numpy as np, torch, onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForTokenClassification
from onnxruntime.quantization import quantize_dynamic, QuantType
from seqeval.metrics import f1_score, precision_score, recall_score

root = Path(__file__).resolve().parents[1]; D = root / "data3"; TYPES = json.loads((D / "meta.json").read_text())["types"]
mdir, outdir = Path(sys.argv[1]), Path(sys.argv[2]); out = outdir / "onnx"; out.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained(mdir)
model = AutoModelForTokenClassification.from_pretrained(mdir).eval().float()
i2l = {int(k): v for k, v in model.config.id2label.items()}
sample = tok(["Mutations in BRCA1 increase breast cancer risk in mice treated with tamoxifen."], return_tensors="pt")
names = [n for n in ("input_ids", "attention_mask", "token_type_ids") if n in sample]
axes = {n: {0: "batch", 1: "seq"} for n in names}; axes["logits"] = {0: "batch", 1: "seq"}
path = out / "model.onnx"
torch.onnx.export(model, tuple(sample[n] for n in names), str(path), input_names=names, output_names=["logits"], dynamic_axes=axes, opset_version=17, dynamo=False)
q = out / "model_quantized.onnx"; quantize_dynamic(str(path), str(q), weight_type=QuantType.QInt8)

def L(c): return [json.loads(l) for l in open(D / c / "test.jsonl", encoding="utf-8")]
def score(c, data, preds):
    G, P = [], []
    for ex, pr in zip(data, preds):
        g, p = list(ex["tags"]), list(pr)
        for i in range(len(g)):
            if g[i] == "X": g[i] = "O"; p[i] = "O"
            elif p[i] != "O" and p[i][2:] not in TYPES[c]: p[i] = "O"
        G.append(g); P.append(p)
    return {"precision": precision_score(G, P), "recall": recall_score(G, P), "f1": f1_score(G, P)}
def predict(sess, data, collect=None):
    preds = []
    for ex in data:
        e = tok(ex["tokens"], is_split_into_words=True, truncation=True, max_length=256, return_tensors="np")
        lg = sess.run(None, {n: e[n].astype(np.int64) for n in names})[0][0]
        if collect is not None: collect.append(lg)
        pr = ["O"] * len(ex["tokens"]); seen = set()
        for pos, w in enumerate(e.word_ids()):
            if w is not None and w not in seen: pr[w] = i2l[int(lg[pos].argmax())]; seen.add(w)
        preds.append(pr)
    return preds
tests = {c: L(c) for c in TYPES}
res = {"files": {}, "scoring": "per corpus only its own types; X words set to O (as train3.py)"}
for tag, p in (("fp32", path), ("int8", q)):
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]); res[tag] = {}; n_sent = 0; t_all = 0.0
    for c, data in tests.items():
        logits = [] if (tag == "fp32" and c == "biored") else None
        t0 = time.time(); preds = predict(sess, data, logits); dt = time.time() - t0; t_all += dt; n_sent += len(data)
        res[tag][c] = score(c, data, preds)
        if logits is not None:
            md, agree, n = 0.0, 0, 0
            with torch.no_grad():
                for ex, lg in list(zip(data, logits))[:300]:
                    e = tok(ex["tokens"], is_split_into_words=True, truncation=True, max_length=256, return_tensors="pt")
                    tl = model(**{k: e[k] for k in names}).logits[0].numpy()
                    md = max(md, float(np.abs(tl - lg).max())); agree += int((tl.argmax(-1) == lg.argmax(-1)).sum()); n += tl.shape[0]
            res["parity_fp32_vs_torch"] = {"sentences_checked": 300, "corpus": "biored test", "max_abs_logit_diff": md, "token_argmax_agreement": agree / n}
    res[tag]["ms_per_sentence"] = round(1000 * t_all / n_sent, 2)
    res["files"][tag] = {"path": p.name, "mb": round(p.stat().st_size / 1e6, 1)}
    print(tag, json.dumps({k: (round(v["f1"], 4) if isinstance(v, dict) else v) for k, v in res[tag].items()}), flush=True)
print("parity", json.dumps(res["parity_fp32_vs_torch"]), flush=True)
(outdir / "onnx_results.json").write_text(json.dumps(res, indent=2)); print("ONNX3_DONE")
