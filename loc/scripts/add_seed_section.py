"""Add measured seed variation to the loc card (loc/release/README.md) and its template (loc/scripts/release_loc.py),
write loc/release/seed_results.json, regenerate SHA256SUMS. Run docs/seed_summary.py first."""
import hashlib, json
from pathlib import Path
root = Path(__file__).resolve().parents[2]
S = json.loads((root / "docs/seed_summary.json").read_text())["loc_150m"]
f = lambda x: f"{x:.3f}"
k = S["macro_auroc_loc"]["n"]
def r(name, a): return f"| {name} | {f(a['mean'])} ± {f(a['std'])} | {f(a['min'])} to {f(a['max'])} |"
rows = [r("Location macro AUROC", S["macro_auroc_loc"]), r("Location macro F1", S["macro_f1_loc"]),
        r("Signal peptide AUROC", S["sp_auroc"]), r("Signal peptide MCC", S["sp_mcc"])]
RARE = ("Golgi apparatus", "Lysosome/Vacuole", "Peroxisome")
rows += [r(f"F1, {c}", S["per_class_f1"][c]) for c in RARE] + [r(f"AP, {c}", S["per_class_ap"][c]) for c in RARE]
lucky = [c for c in RARE if S["per_class_f1"][c]["values"][0] == S["per_class_f1"][c]["max"] and S["per_class_f1"][c]["std"] > 0.02]
NOTE = "".join(f" The released checkpoint's {c} F1 ({f(S['per_class_f1'][c]['values'][0])}) is the best of the {k} seeds, so the per-class table above overstates what a typical run gets for {c.lower()} (mean {f(S['per_class_f1'][c]['mean'])})." for c in lucky)
SECTION = f"""

### Seed variation
We retrained the 150M model with the same data and settings using {k - 1} more seed{'s' if k > 2 else ''} ({', '.join(map(str, S['seeds'][1:]))}), chosen before training; all {k} runs (with the released seed 13) are reported, none dropped. Test set, mean ± SD across seeds:

| Metric | Mean ± SD | Range |
|---|---|---|
{chr(10).join(rows)}

Rare-class F1 depends on a threshold picked on few validation positives, so it is much less stable than AUROC or AP.{NOTE} {k} seeds is a small sample, so treat the SD as rough. It covers training randomness only, not test-set sampling. Per-class values for all classes: `seed_results.json`."""
OLD1 = "Single training run, single seed, no confidence intervals."
NEW1 = "Released checkpoint: seed 13; seed variation is measured below."
ANCHOR = "but low in absolute terms."
OLD3 = "- Research use only. One run, one seed, no confidence intervals."
NEW3 = f"- Research use only. Seed variation measured with only {k} seeds; no test-set bootstrap intervals."

def patch(t, tpl):
    sec = SECTION.replace("{", "{{").replace("}", "}}") if tpl else SECTION
    for old, new in [(OLD1, NEW1), (OLD3, NEW3), (ANCHOR, ANCHOR + sec)]:
        assert t.count(old) == 1, old[:50]; t = t.replace(old, new)
    return t
card = root / "loc/release/README.md"; card.write_text(patch(card.read_text(encoding="utf-8"), False), encoding="utf-8")
tpl = root / "loc/scripts/release_loc.py"; tpl.write_text(patch(tpl.read_text(encoding="utf-8"), True), encoding="utf-8", newline="\n")
rel = root / "loc/release"; (rel / "seed_results.json").write_text(json.dumps(S, indent=2))
lines = [hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.relative_to(rel).as_posix() for p in sorted(rel.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"]
(rel / "SHA256SUMS").write_text("\n".join(lines) + "\n"); print("ok", len(lines), "seeds", S["seeds"])
