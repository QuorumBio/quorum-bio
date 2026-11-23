---
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

34,927 reviewed UniProtKB/Swiss-Prot proteins with **experimentally supported** subcellular locations (10 classes, multi-label) and a signal peptide flag. The split is built so that **no validation or test protein has 30% or more sequence identity to any training protein** (MMseqs2). This is the exact data used to train and evaluate [quorumbio/esm2-protein-subcellular-localization-signal-peptide](https://huggingface.co/quorumbio/esm2-protein-subcellular-localization-signal-peptide).

Source: UniProt release 2026_04. License: CC BY 4.0, inherited from UniProt (see Attribution).

## Load
```python
from datasets import load_dataset
ds = load_dataset("quorumbio/swissprot-subcellular-localization-signal-peptide")
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

Class order: Nucleus, Cytoplasm, Extracellular, Mitochondrion, Cell membrane, Endoplasmic reticulum, Plastid, Golgi apparatus, Lysosome/Vacuole, Peroxisome.

## Counts
| | train | validation | test |
|---|---|---|---|
| proteins | 30,134 | 2,457 | 2,336 |
| signal peptide | 5,636 | 385 | 450 |
| Nucleus | 9,064 | 835 | 716 |
| Cytoplasm | 10,332 | 948 | 842 |
| Extracellular | 5,621 | 346 | 351 |
| Mitochondrion | 2,659 | 204 | 221 |
| Cell membrane | 4,349 | 305 | 295 |
| Endoplasmic reticulum | 2,027 | 186 | 192 |
| Plastid | 1,029 | 70 | 84 |
| Golgi apparatus | 1,280 | 115 | 78 |
| Lysosome/Vacuole | 858 | 62 | 88 |
| Peroxisome | 343 | 62 | 35 |

Golgi apparatus, Lysosome/Vacuole and Peroxisome are rare (peroxisome is about 1.5% of the test set).

## How it was built
1. Download all reviewed Swiss-Prot entries (UniProt REST, fields: accession, sequence, length, ft_signal, cc_subcellular_location, organism_id): 576,073 entries.
2. Keep sequences of length 30 to 4000 with only the 20 standard amino acids.
3. From the subcellular location comment, keep only locations whose evidence includes **ECO:0000269** (experimental evidence used in manual assertion). Map them to 10 classes by name prefix: nucleus/nucleoplasm to Nucleus; cytoplasm/cytosol to Cytoplasm; secreted to Extracellular; mitochondri* to Mitochondrion; cell membrane; endoplasmic reticulum; plastid/chloroplast to Plastid; Golgi; lysosome/vacuole; peroxisome. Entries with no mapped experimental location are dropped (37,733 kept).
4. Signal peptide label: presence of a UniProt SIGNAL feature (any evidence level, unlike the locations).
5. Cluster all kept sequences with MMseqs2 (`cluster --min-seq-id 0.3 -c 0.8 --cov-mode 0`): 20,269 clusters. Assign whole clusters 80/10/10 to train/validation/test (random seed 13).
6. Search validation and test against train (`search --min-seq-id 0.3 -c 0.5 --cov-mode 2 -s 7.5`) and remove every validation/test protein with a hit (1,328 validation and 1,478 test removed).

Checked when building: no accession and no cluster appears in more than one split. Scripts: `scripts/prepare_uniprot.py` (steps 1 to 6) and `scripts/build_dataset.py` (this release). Hashes of the source file: `provenance.json`.

## Limitations
- **A missing location is not a negative.** It means there is no experimental annotation for that compartment, not that the protein is absent from it. Treat negatives as noisy.
- The class mapping is a simple name-prefix rule set; sub-compartments (for example nuclear envelope, mitochondrial membrane) fold into their parent class, and other compartments (for example cytoskeleton, endosome) are ignored.
- Signal peptide labels include SIGNAL features that UniProt annotates from sequence analysis, not only from experiments. Across all splits, 4793 of 6471 positives (74%) carry ECO:0000255 (sequence-model evidence, typically from prediction tools such as SignalP), 1183 (18%) carry ECO:0000269 (experimental) and 495 have other or no evidence codes. To evaluate on experimentally supported signal peptides only, look up the evidence in UniProt by accession.
- Isoform-specific locations are merged per protein (one row per accession, canonical sequence).
- Swiss-Prot is biased toward well-studied organisms (human, mouse, yeast, Arabidopsis, E. coli).
- Not comparable to DeepLoc or SignalP benchmark splits; those use different data, partitions and labels.

## Attribution
Data from UniProtKB/Swiss-Prot, (c) the UniProt Consortium, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). We filtered entries, mapped locations to 10 classes, derived labels and made the split; these changes are ours, not UniProt's. Please cite UniProt:

The UniProt Consortium. UniProt: the Universal Protein Knowledgebase in 2025. Nucleic Acids Research 53(D1):D609-D617 (2025).

The split itself may be cited as this dataset (quorumbio/swissprot-subcellular-localization-signal-peptide).
