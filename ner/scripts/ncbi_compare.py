"""Out-of-domain comparison on NCBI-disease test (Disease only), same path for both models:
text -> pipeline(simple aggregation) -> char spans -> word labels -> seqeval. Neither model was trained on NCBI-disease."""
import json, time
from pathlib import Path
import torch
from transformers import pipeline
from seqeval.metrics import f1_score, precision_score, recall_score
root = Path(__file__).resolve().parents[1]
test = [json.loads(l) for l in open(root / "data" / "ncbi_disease" / "test.jsonl", encoding="utf-8")]
texts = [e["text"] for e in test]
def keep_disease(tags): return [t if t == "O" or t[2:] == "Disease" else "O" for t in tags]
def evaluate(model_id, mapping):
    pipe = pipeline("token-classification", model=model_id, aggregation_strategy="simple", device=0)
    out = pipe(texts, batch_size=32); P, G = [], []
    for e, ents in zip(test, out):
        tags = ["O"] * len(e["tokens"])
        for g in ents:
            lab = mapping(g["entity_group"])
            if lab is None: continue
            first = True
            for i, (s, t) in enumerate(e["offsets"]):
                if t > g["start"] and s < g["end"]:
                    tags[i] = ("B-" if first else "I-") + lab; first = False
        P.append(keep_disease(tags)); G.append(keep_disease(e["tags"]))
    return {"model": model_id, "precision": precision_score(G, P), "recall": recall_score(G, P), "f1": f1_score(G, P),
            "sentences": len(texts), "gold_entities": sum(t.startswith("B-") for g in G for t in g)}
res = {"run1": evaluate(str(root / "models" / "run1"), lambda g: g)}
strict = {"Disease_disorder": "Disease"}; generous = dict(strict, Sign_symptom="Disease")
res["d4data_strict_mapping"] = evaluate("d4data/biomedical-ner-all", lambda g: strict.get(g))
res["d4data_generous_mapping"] = evaluate("d4data/biomedical-ner-all", lambda g: generous.get(g))
res["note"] = "NCBI-disease test, Disease only, fp32 pipeline path. Neither model trained on NCBI-disease. d4data label mapping is our approximation."
(root / "results" / "ncbi_compare.json").write_text(json.dumps(res, indent=2, default=float))
for k in ("run1", "d4data_strict_mapping", "d4data_generous_mapping"):
    r = res[k]; print(k, "P %.4f R %.4f F1 %.4f" % (r["precision"], r["recall"], r["f1"]), flush=True)
print("NCBI_DONE")
