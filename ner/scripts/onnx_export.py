"""Export run1 to ONNX (fp32 + dynamic int8), check parity with PyTorch and report test F1 for each through onnxruntime."""
import json, time
from pathlib import Path
import numpy as np, torch, onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForTokenClassification
from onnxruntime.quantization import quantize_dynamic, QuantType
from seqeval.metrics import f1_score, precision_score, recall_score

root = Path(__file__).resolve().parents[1]
mdir = root / "models" / "run1"; out = root / "release" / "onnx"; out.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained(mdir)
model = AutoModelForTokenClassification.from_pretrained(mdir).eval().float()
i2l = {int(k): v for k, v in model.config.id2label.items()}
test = [json.loads(l) for l in open(root / "data" / "bc5cdr" / "test.jsonl", encoding="utf-8")]
sample = tok(["Naloxone reverses the antihypertensive effect of clonidine."], return_tensors="pt")
names = [n for n in ("input_ids", "attention_mask", "token_type_ids") if n in sample]
args = tuple(sample[n] for n in names)
axes = {n: {0: "batch", 1: "seq"} for n in names}; axes["logits"] = {0: "batch", 1: "seq"}
path = out / "model.onnx"
try:
    torch.onnx.export(model, args, str(path), input_names=names, output_names=["logits"], dynamic_axes=axes, opset_version=17, dynamo=False)
except Exception as e:
    print("legacy export failed:", repr(e)[:300], flush=True)
    torch.onnx.export(model, args, str(path), input_names=names, output_names=["logits"], dynamic_axes=axes, opset_version=17)
q = out / "model_quantized.onnx"
quantize_dynamic(str(path), str(q), weight_type=QuantType.QInt8)

def make(p): return ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
def predict(sess, data, collect=None):
    preds = []
    for k, ex in enumerate(data):
        e = tok(ex["tokens"], is_split_into_words=True, truncation=True, max_length=256, return_tensors="np")
        lg = sess.run(None, {n: e[n].astype(np.int64) for n in names})[0][0]
        if collect is not None: collect.append(lg)
        wid = e.word_ids(); pr = ["O"] * len(ex["tokens"]); seen = set()
        for pos, w in enumerate(wid):
            if w is not None and w not in seen: pr[w] = i2l[int(lg[pos].argmax())]; seen.add(w)
        preds.append(pr)
    return preds
golds = [e["tags"] for e in test]
res = {"files": {}}
for tag, p in (("fp32", path), ("int8", q)):
    sess = make(p); logits = []
    t0 = time.time(); preds = predict(sess, test, logits); dt = time.time() - t0
    res[tag] = {"precision": precision_score(golds, preds), "recall": recall_score(golds, preds), "f1": f1_score(golds, preds),
                "cpu_seconds_total": round(dt, 1), "ms_per_sentence": round(1000 * dt / len(test), 2)}
    res["files"][tag] = {"path": str(p.name), "mb": round(p.stat().st_size / 1e6, 1)}
    if tag == "fp32":
        md, agree, n = 0.0, 0, 0
        with torch.no_grad():
            for ex, lg in list(zip(test, logits))[:300]:
                e = tok(ex["tokens"], is_split_into_words=True, truncation=True, max_length=256, return_tensors="pt")
                tl = model(**{n_: e[n_] for n_ in names}).logits[0].numpy()
                md = max(md, float(np.abs(tl - lg).max())); agree += int((tl.argmax(-1) == lg.argmax(-1)).sum()); n += tl.shape[0]
        res["parity_fp32_vs_torch"] = {"sentences_checked": 300, "max_abs_logit_diff": md, "token_argmax_agreement": agree / n}
    print(tag, json.dumps(res[tag]), flush=True)
print("parity", json.dumps(res["parity_fp32_vs_torch"]), flush=True)
(root / "results" / "onnx.json").write_text(json.dumps(res, indent=2))
print("ONNX_DONE")
