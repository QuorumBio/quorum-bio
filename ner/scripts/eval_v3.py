"""Pipeline-path evaluation for NER v3 (same input path as the v2 card table: transformers pipeline on sentence text,
aggregation_strategy="simple", entities mapped back onto our word tokens).
Per corpus, only the types that corpus annotates are scored (other predicted types set to O); NLM-Gene X words
(Domain/Other) are set to O in gold and prediction. Also scores the released v2 model (Chemical/Disease only) on the
BC5CDR and NCBI-disease test sets, and on the Chemical/Disease subset of BioRED test, for comparison.
Usage: python eval_v3.py ner/models/v3a  -> writes <model>/eval_pipeline.json"""
import json, sys, time
from pathlib import Path
import torch
from transformers import pipeline
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
root = Path(__file__).resolve().parents[1]; D = root / "data3"
TYPES = json.loads((D / "meta.json").read_text())["types"]
def load(c): return [json.loads(l) for l in open(D / c / "test.jsonl", encoding="utf-8")]

def evaluate(pipe, corpus, keep):
    test = load(corpus); texts = [e["text"] for e in test]
    torch.cuda.synchronize(); t0 = time.time(); out = pipe(texts, batch_size=32); torch.cuda.synchronize(); dt = time.time() - t0
    P, G = [], []
    for e, ents in zip(test, out):
        tags = ["O"] * len(e["tokens"])
        for g in ents:
            first = True
            for i, (s, t) in enumerate(e["offsets"]):
                if t > g["start"] and s < g["end"]:
                    tags[i] = ("B-" if first else "I-") + g["entity_group"]; first = False
        gold = list(e["tags"])
        for i in range(len(gold)):
            if gold[i] == "X": gold[i] = "O"; tags[i] = "O"
            if tags[i] != "O" and tags[i][2:] not in keep: tags[i] = "O"
            if gold[i] != "O" and gold[i][2:] not in keep: gold[i] = "O"
        P.append(tags); G.append(gold)
    rep = classification_report(G, P, output_dict=True, zero_division=0)
    return {"types_scored": sorted(keep), "sentences": len(test), "gpu_seconds": round(dt, 1),
            "precision": precision_score(G, P), "recall": recall_score(G, P), "f1": f1_score(G, P),
            "per_type": {k: {m: float(v[m]) for m in ("precision", "recall", "f1-score", "support")} for k, v in rep.items() if k in TYPES["biored"]}}

def run(model_dir, plan):
    pipe = pipeline("token-classification", model=str(model_dir), aggregation_strategy="simple", device=0)
    pipe(["warm up sentence"] * 8, batch_size=8)
    r = {}
    for name, corpus, keep in plan:
        r[name] = evaluate(pipe, corpus, set(keep)); print(model_dir.name, name, "P %.4f R %.4f F1 %.4f" % (r[name]["precision"], r[name]["recall"], r[name]["f1"]), flush=True)
    return r

if __name__ == "__main__":
    m = Path(sys.argv[1])
    res = {"v3": run(m, [(c, c, TYPES[c]) for c in TYPES] + [("biored_chem_disease", "biored", ["Chemical", "Disease"])]),
           "v2_released": run(root / "release", [("bc5cdr", "bc5cdr", ["Chemical", "Disease"]), ("ncbi_disease", "ncbi_disease", ["Disease"]),
                                                  ("biored_chem_disease", "biored", ["Chemical", "Disease"])])}
    (m / "eval_pipeline.json").write_text(json.dumps(res, indent=2)); print("EVAL3_DONE")
