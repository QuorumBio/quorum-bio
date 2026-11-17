"""Evaluate run1 and d4data/biomedical-ner-all on the BC5CDR test set through the SAME path:
text -> pipeline(aggregation_strategy=simple) -> char spans -> word labels -> seqeval entity-level P/R/F1."""
import json, time
from pathlib import Path
import torch
from transformers import pipeline
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

root = Path(__file__).resolve().parents[1]
test = [json.loads(l) for l in open(root / "data" / "bc5cdr" / "test.jsonl", encoding="utf-8")]
texts = [e["text"] for e in test]

def evaluate(model_id, mapping):
    pipe = pipeline("token-classification", model=model_id, aggregation_strategy="simple", device=0)
    params = sum(p.numel() for p in pipe.model.parameters())
    pipe(texts[:64], batch_size=32)  # warmup
    torch.cuda.synchronize(); t0 = time.time(); out = pipe(texts, batch_size=32); torch.cuda.synchronize(); dt = time.time() - t0
    preds, golds, seen_groups = [], [], {}
    for e, ents in zip(test, out):
        tags = ["O"] * len(e["tokens"])
        for g in ents:
            seen_groups[g["entity_group"]] = seen_groups.get(g["entity_group"], 0) + 1
            lab = mapping(g["entity_group"])
            if lab is None: continue
            first = True
            for i, (s, t) in enumerate(e["offsets"]):
                if t > g["start"] and s < g["end"]:
                    tags[i] = ("B-" if first else "I-") + lab; first = False
        preds.append(tags); golds.append(e["tags"])
    return {"model": model_id, "params_millions": round(params / 1e6, 1), "sentences": len(texts), "gpu_seconds": round(dt, 1),
            "precision": precision_score(golds, preds), "recall": recall_score(golds, preds), "f1": f1_score(golds, preds),
            "per_type": classification_report(golds, preds, output_dict=True, zero_division=0),
            "predicted_groups_seen": seen_groups}

res = {}
res["run1"] = evaluate(str(root / "models" / "run1"), lambda g: g)
strict = {"Disease_disorder": "Disease", "Medication": "Chemical"}
generous = dict(strict, Sign_symptom="Disease")
res["d4data_strict_mapping"] = evaluate("d4data/biomedical-ner-all", lambda g: strict.get(g))
res["d4data_generous_mapping"] = evaluate("d4data/biomedical-ner-all", lambda g: generous.get(g))
res["note"] = "d4data/biomedical-ner-all is a general 84-type model not trained on BC5CDR; label mapping is our approximation (strict: Disease_disorder->Disease, Medication->Chemical; generous: also Sign_symptom->Disease). run1 was trained on the BC5CDR training set."
(root / "results").mkdir(exist_ok=True)
(root / "results" / "baseline.json").write_text(json.dumps(res, indent=2, default=float))
for k in ("run1", "d4data_strict_mapping", "d4data_generous_mapping"):
    r = res[k]; print(k, "P %.4f R %.4f F1 %.4f" % (r["precision"], r["recall"], r["f1"]), r["params_millions"], "M params", r["gpu_seconds"], "s", r["predicted_groups_seen"] if "d4data" in k else "", flush=True)
print("BASELINE_DONE")
