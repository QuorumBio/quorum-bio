"""Pipeline-path evaluation of run2a (and run1 for reference) on BC5CDR test (Chemical+Disease) and NCBI-disease test (Disease only; predicted Chemical dropped)."""
import json, time
from pathlib import Path
import torch
from transformers import pipeline
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
root = Path(__file__).resolve().parents[1]
def load(c): return [json.loads(l) for l in open(root / "data" / c / "test.jsonl", encoding="utf-8")]
def evaluate(model_dir, corpus, keep=None):
    test = load(corpus); texts = [e["text"] for e in test]
    pipe = pipeline("token-classification", model=model_dir, aggregation_strategy="simple", device=0)
    params = sum(p.numel() for p in pipe.model.parameters()); pipe(texts[:64], batch_size=32)
    torch.cuda.synchronize(); t0 = time.time(); out = pipe(texts, batch_size=32); torch.cuda.synchronize(); dt = time.time() - t0
    f = (lambda tags: [t if t == "O" or t[2:] in keep else "O" for t in tags]) if keep else (lambda tags: tags)
    P, G = [], []
    for e, ents in zip(test, out):
        tags = ["O"] * len(e["tokens"])
        for g in ents:
            first = True
            for i, (s, t) in enumerate(e["offsets"]):
                if t > g["start"] and s < g["end"]:
                    tags[i] = ("B-" if first else "I-") + g["entity_group"]; first = False
        P.append(f(tags)); G.append(f(e["tags"]))
    return {"params_millions": round(params / 1e6, 1), "sentences": len(test), "gpu_seconds": round(dt, 1),
            "gold_entities": sum(t.startswith("B-") for g in G for t in g),
            "precision": precision_score(G, P), "recall": recall_score(G, P), "f1": f1_score(G, P),
            "per_type": classification_report(G, P, output_dict=True, zero_division=0)}
res = {}
for name in ("run2a", "run1"):
    d = str(root / "models" / name)
    res[name] = {"bc5cdr_test": evaluate(d, "bc5cdr"), "ncbi_disease_test_disease_only": evaluate(d, "ncbi_disease", {"Disease"})}
    for k, v in res[name].items(): print(name, k, "P %.4f R %.4f F1 %.4f" % (v["precision"], v["recall"], v["f1"]), flush=True)
(root / "results" / "eval_v2.json").write_text(json.dumps(res, indent=2, default=float))
print("EVAL2_DONE")
