"""Add measured seed variation to the NER v2 card (ner/release/README.md) and its generator template
(ner/scripts/release_pack2.py), copy the numbers to ner/release/seed_results.json, regenerate SHA256SUMS.
Numbers come from docs/seed_summary.json (run docs/seed_summary.py first)."""
import hashlib, json
from pathlib import Path
root = Path(__file__).resolve().parents[2]
S = json.loads((root / "docs/seed_summary.json").read_text())["ner_v2"]
b, n = S["test_f1_bc5cdr"], S["test_f1_ncbi_disease"]
f = lambda x: f"{x:.3f}"

OLD1 = "Single training run, single seed ({seed}), no confidence intervals; small differences are not meaningful."
NEW1 = "The released checkpoint is seed {seed}; seed variation is measured below."
OLD2 = "Adding NCBI-disease to training barely moved BC5CDR (a difference within what one seed can produce) and raised NCBI-disease F1 a lot"
NEW2 = (f"Adding NCBI-disease to training lowered BC5CDR F1 slightly (v1, a single seed: 0.896; v2 over {b['n']} seeds: "
        f"{f(b['mean'])} ± {f(b['std'])}, so the drop looks real but small) and raised NCBI-disease F1 a lot")
SECTION = f"""

### Seed variation
We retrained v2 with the same data and settings and {b['n'] - 1} other seeds (seeds {', '.join(map(str, S['seeds']))}, chosen before training; all runs reported, none dropped). Test entity F1 from the training script's scorer (same seqeval metric; tokenization differs slightly from the table above, which is why seed 13's NCBI-disease number is 0.870 here and 0.878 there):

| Test set | Mean ± SD | Range |
|---|---|---|
| BC5CDR | {f(b['mean'])} ± {f(b['std'])} | {f(b['min'])} to {f(b['max'])} |
| NCBI-disease | {f(n['mean'])} ± {f(n['std'])} | {f(n['min'])} to {f(n['max'])} |

The released checkpoint is typical, not a lucky seed. These intervals cover training randomness only, not uncertainty from the small test sets (NCBI-disease test has 100 abstracts). Per-seed values: `seed_results.json`."""
ANCHOR = "and was discarded."
OLD3 = "- One run, one seed, no confidence intervals."
NEW3 = f"- Seed variation measured with {b['n']} seeds (above); v1 was a single seed, and no test-set bootstrap intervals are given."

def patch(text, seed_repr):
    for old, new in [(OLD1.format(seed=seed_repr), NEW1.format(seed=seed_repr)), (OLD2, NEW2), (OLD3, NEW3)]:
        assert text.count(old) == 1, old[:60]; text = text.replace(old, new)
    assert text.count(ANCHOR) == 1; return text.replace(ANCHOR, ANCHOR + SECTION.replace("{", "{{").replace("}", "}}") if seed_repr != "13" else ANCHOR + SECTION)

card = root / "ner/release/README.md"; card.write_text(patch(card.read_text(encoding="utf-8"), "13"), encoding="utf-8", newline="\n")
tpl = root / "ner/scripts/release_pack2.py"; tpl.write_text(patch(tpl.read_text(encoding="utf-8"), "{R['args']['seed']}"), encoding="utf-8", newline="\n")
rel = root / "ner/release"; (rel / "seed_results.json").write_text(json.dumps(S, indent=2), newline="\n")
lines = [hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.relative_to(rel).as_posix() for p in sorted(rel.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"]
(rel / "SHA256SUMS").write_text("\n".join(lines) + "\n"); print("ok", len(lines))
