# Quorum Bio

Small open biology models on Hugging Face, with reproducible training.

## Models
- [quorumbio/biomedical-ner-chemical-disease](https://huggingface.co/quorumbio/biomedical-ner-chemical-disease): Chemical and Disease NER (BiomedBERT-base fine-tuned on BC5CDR and NCBI-disease), PyTorch and ONNX (fp32, int8). Apache-2.0. [Browser demo](https://huggingface.co/spaces/quorumbio/biomed-ner-demo).
- [quorumbio/esm2-protein-subcellular-localization-signal-peptide](https://huggingface.co/quorumbio/esm2-protein-subcellular-localization-signal-peptide): protein subcellular localization (10 classes) and signal peptide presence (ESM-2 150M), PyTorch and ONNX. Apache-2.0. [Browser demo](https://huggingface.co/spaces/quorumbio/protein-localization-demo).

## Dataset
- [quorumbio/swissprot-subcellular-localization-signal-peptide](https://huggingface.co/datasets/quorumbio/swissprot-subcellular-localization-signal-peptide): the homology-partitioned (30% identity) Swiss-Prot split used for the localization model. CC BY 4.0.

All of the above: [collection](https://huggingface.co/collections/quorumbio/quorum-bio-open-biology-models-6b03617169bc86a3fd82656b). Technical report draft: `docs/report.md`.

## Reproduce (Windows, NVIDIA GPU)
`setup_env.bat` builds a venv with CUDA PyTorch. `run_ner1.bat` downloads BC5CDR/NCBI-disease from the official sources, trains, and writes results to `ner/models/run1`. `run_eval.bat` runs the baseline comparison and ONNX export. See `ner/scripts/` and the model card for details and limitations.

`COMPANY_STATE.md` is the running log (hypotheses, dataset licenses, evals, costs). `DAILY.md` is the daily routine.

## License
Apache-2.0 (see `LICENSE` and `NOTICE`). Training data licenses are recorded in `COMPANY_STATE.md` and the model card.
