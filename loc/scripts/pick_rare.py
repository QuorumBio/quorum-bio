"""Pick the rare-class screen winner on DEV only. Rule fixed before seeing screen results (2026-11-19):
winner = variant with best mean dev rare-class AP over seeds 13 and 1; it advances only if
gain vs baseline A >= 0.02 rare AP and dev macro AP drop vs A <= 0.01. Prints the train_loc args of the winner, or NONE."""
import json, sys
from pathlib import Path
root = Path(__file__).resolve().parents[1] / "models" / "screen"
ARGS = {"A": "", "B": "--pw_clip 100", "C": "--loss focal", "D": "--pool attn"}
res = {}
for v in ARGS:
    runs = [json.loads((root / f"{v}_s{s}" / "dev_results.json").read_text())["dev_at_best"] for s in (13, 1) if (root / f"{v}_s{s}" / "dev_results.json").exists()]
    if len(runs) == 2: res[v] = {"rare_ap": sum(r["rare_ap"] for r in runs) / 2, "macro_ap": sum(r["macro_ap_loc"] for r in runs) / 2, "per_seed_rare": [r["rare_ap"] for r in runs]}
out = {"rule": __doc__, "results": res}
if "A" not in res: out["winner"] = None; out["reason"] = "baseline incomplete"
else:
    cand = [v for v in res if v != "A" and res[v]["rare_ap"] - res["A"]["rare_ap"] >= 0.02 and res["A"]["macro_ap"] - res[v]["macro_ap"] <= 0.01]
    w = max(cand, key=lambda v: res[v]["rare_ap"]) if cand else None
    out["winner"] = w; out["reason"] = "passes rule" if w else "no variant passes the rule"
(root / "pick.json").write_text(json.dumps(out, indent=2))
print(json.dumps({k: v for k, v in out.items() if k != "rule"}), file=sys.stderr)
print(ARGS[out["winner"]] if out["winner"] else "NONE")
