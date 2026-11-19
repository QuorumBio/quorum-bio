"""Build the HF dataset quorumbio/swissprot-localization-signalpeptide from loc/data (the exact split used to train
quorumbio/esm2-localization-signalpeptide). Writes loc/dataset/: data/{train,validation,test}.parquet, README.md, provenance.json, SHA256SUMS."""
import hashlib, json, shutil
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq

root = Path(__file__).resolve().parents[1]
src, out = root / "data", root / "dataset"
if out.exists(): shutil.rmtree(out)
(out / "data").mkdir(parents=True); (out / "scripts").mkdir()
CLASSES = ["Nucleus", "Cytoplasm", "Extracellular", "Mitochondrion", "Cell membrane", "Endoplasmic reticulum", "Plastid", "Golgi apparatus", "Lysosome/Vacuole", "Peroxisome"]
cluster = {}
for l in open(src / "mm" / "clu_cluster.tsv"):
    rep, mem = l.rstrip("\n").split("\t"); cluster[mem] = rep
prov = json.loads((src / "provenance.json").read_text())
prov["sources"]["uniprot_release"] = "2026_04 (released 2026-11-04; file downloaded 2026-11-17, release confirmed via the X-UniProt-Release header, which still read 2026_04 on 2026-11-19)"
counts = {}
for name, f in [("train", "train"), ("validation", "dev"), ("test", "test")]:
    rows = [json.loads(l) for l in open(src / f"{f}.jsonl")]
    for r in rows: assert set(r["locations"]) <= set(CLASSES) and r["locations"]
    t = pa.table({
        "accession": [r["id"] for r in rows],
        "sequence": [r["sequence"] for r in rows],
        "length": pa.array([len(r["sequence"]) for r in rows], pa.int32()),
        "organism_id": pa.array([int(r["organism_id"]) if r["organism_id"] else None for r in rows], pa.int64()),
        "locations": [r["locations"] for r in rows],
        "location_labels": pa.array([[int(c in r["locations"]) for c in CLASSES] for r in rows], pa.list_(pa.int8())),
        "signal_peptide": [bool(r["signal_peptide"]) for r in rows],
        "signal_peptide_end": pa.array([r["sp_end"] for r in rows], pa.int32()),
        "cluster_id": [cluster[r["id"]] for r in rows],
    })
    pq.write_table(t, out / "data" / f"{name}.parquet", compression="zstd")
    counts[name] = {"n": len(rows), "signal_peptide": sum(r["signal_peptide"] for r in rows), **{c: sum(c in r["locations"] for r in rows) for c in CLASSES}}
# no accession appears in two splits; no cluster spans splits
ids = {k: set(pq.read_table(out / "data" / f"{k}.parquet", columns=["accession"])["accession"].to_pylist()) for k in counts}
cl = {k: set(pq.read_table(out / "data" / f"{k}.parquet", columns=["cluster_id"])["cluster_id"].to_pylist()) for k in counts}
for a in counts:
    for b in counts:
        if a < b: assert not ids[a] & ids[b] and not cl[a] & cl[b], (a, b)
prov["dataset_counts"] = counts
(out / "provenance.json").write_text(json.dumps(prov, indent=2))
for s in ("prepare_uniprot.py", "build_dataset.py"): shutil.copy(root / "scripts" / s, out / "scripts" / s)

C = counts
def row(c): return f"| {c} | {C['train'][c]:,} | {C['validation'][c]:,} | {C['test'][c]:,} |"
card = f"""---
license: cc-by-4.0
language: [en]
pretty_name: Swiss-Prot subcellular localization and signal peptide (homology-partitioned)
size_categories: [10K<n<100K]
task_categories: [text-classification]
tags: [biology, protein, subcellular-localization, signal-peptide, uniprot, swiss-prot, multi-label, homology-split]
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.parquet
  - split: validation
    path: data/validation.parquet
  - split: test
    path: data/test.parquet
---

# Swiss-Prot subcellular localization and signal peptide (homology-partitioned split)

{C['train']['n'] + C['validation']['n'] + C['test']['n']:,} reviewed UniProtKB/Swiss-Prot proteins with **experimentally supported** subcellular locations (10 classes, multi-label) and a signal peptide flag. The split is built so that **no validation or test protein has 30% or more sequence identity to any training protein** (MMseqs2). This is the exact data used to train and evaluate [quorumbio/esm2-localization-signalpeptide](https://huggingface.co/quorumbio/esm2-localization-signalpeptide).

Source: UniProt release 2026_04. License: CC BY 4.0, inherited from UniProt (see Attribution).

## Load
```python
from datasets import load_dataset
ds = load_dataset("quorumbio/swissprot-localization-signalpeptide")
```

## Columns
| column | type | meaning |
|---|---|---|
| accession | string | UniProtKB accession |
| sequence | string | amino acid sequence (20 standard letters only, length 30 to 4000) |
| length | int32 | sequence length |
| organism_id | int64 | NCBI taxonomy ID |
| locations | list[string] | location classes with experimental evidence (at least one) |
| location_labels | list[int8] | multi-hot vector of length 10, in the class order below |
| signal_peptide | bool | UniProt has a SIGNAL feature for this entry |
| signal_peptide_end | int32 | last residue of the signal peptide (1-based), null if none |
| cluster_id | string | accession of the MMseqs2 cluster representative (30% identity); use it for grouped cross-validation |

Class order: {", ".join(CLASSES)}.

## Counts
| | train | validation | test |
|---|---|---|---|
| proteins | {C['train']['n']:,} | {C['validation']['n']:,} | {C['test']['n']:,} |
| signal peptide | {C['train']['signal_peptide']:,} | {C['validation']['signal_peptide']:,} | {C['test']['signal_peptide']:,} |
{chr(10).join(row(c) for c in CLASSES)}

Golgi apparatus, Lysosome/Vacuole and Peroxisome are rare (peroxisome is about 1.5% of the test set).

## How it was built
1. Download all reviewed Swiss-Prot entries (UniProt REST, fields: accession, sequence, length, ft_signal, cc_subcellular_location, organism_id): {prov['counts']['swissprot_entries']:,} entries.
2. Keep sequences of length 30 to 4000 with only the 20 standard amino acids.
3. From the subcellular location comment, keep only locations whose evidence includes **ECO:0000269** (experimental evidence used in manual assertion). Map them to 10 classes by name prefix: nucleus/nucleoplasm to Nucleus; cytoplasm/cytosol to Cytoplasm; secreted to Extracellular; mitochondri* to Mitochondrion; cell membrane; endoplasmic reticulum; plastid/chloroplast to Plastid; Golgi; lysosome/vacuole; peroxisome. Entries with no mapped experimental location are dropped ({prov['counts']['kept_with_experimental_location']:,} kept).
4. Signal peptide label: presence of a UniProt SIGNAL feature (any evidence level, unlike the locations).
5. Cluster all kept sequences with MMseqs2 (`cluster --min-seq-id 0.3 -c 0.8 --cov-mode 0`): {prov['clusters']:,} clusters. Assign whole clusters 80/10/10 to train/validation/test (random seed 13).
6. Search validation and test against train (`search --min-seq-id 0.3 -c 0.5 --cov-mode 2 -s 7.5`) and remove every validation/test protein with a hit ({prov['removed_dev_similar_to_train']:,} validation and {prov['removed_test_similar_to_train']:,} test removed).

Checked when building: no accession and no cluster appears in more than one split. Scripts: `scripts/prepare_uniprot.py` (steps 1 to 6) and `scripts/build_dataset.py` (this release). Hashes of the source file: `provenance.json`.

## Limitations
- **A missing location is not a negative.** It means there is no experimental annotation for that compartment, not that the protein is absent from it. Treat negatives as noisy.
- The class mapping is a simple name-prefix rule set; sub-compartments (for example nuclear envelope, mitochondrial membrane) fold into their parent class, and other compartments (for example cytoskeleton, endosome) are ignored.
- Signal peptide labels include SIGNAL features that UniProt annotates from sequence analysis, not only from experiments.
- Isoform-specific locations are merged per protein (one row per accession, canonical sequence).
- Swiss-Prot is biased toward well-studied organisms (human, mouse, yeast, Arabidopsis, E. coli).
- Not comparable to DeepLoc or SignalP benchmark splits; those use different data, partitions and labels.

## Attribution
Data from UniProtKB/Swiss-Prot, (c) the UniProt Consortium, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). We filtered entries, mapped locations to 10 classes, derived labels and made the split; these changes are ours, not UniProt's. Please cite UniProt:

The UniProt Consortium. UniProt: the Universal Protein Knowledgebase in 2025. Nucleic Acids Research 53(D1):D609-D617 (2025).

The split itself may be cited as this dataset (quorumbio/swissprot-localization-signalpeptide).
"""
(out / "README.md").write_text(card, encoding="utf-8")
lines = [hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.relative_to(out).as_posix() for p in sorted(out.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"]
(out / "SHA256SUMS").write_text("\n".join(lines) + "\n")
print("DATASET_DONE", json.dumps({k: v["n"] for k, v in counts.items()}))
