# Quorum Bio

Small open biology models on Hugging Face, with reproducible training.

## Models
- [quorumbio/biomed-ner-chem-disease](https://huggingface.co/quorumbio/biomed-ner-chem-disease): Chemical and Disease NER (BiomedBERT-base fine-tuned on BC5CDR), PyTorch and ONNX (fp32, int8). Apache-2.0.
- [Browser demo](https://huggingface.co/spaces/quorumbio/biomed-ner-demo): runs the int8 ONNX model in the browser.

## Reproduce (Windows, NVIDIA GPU)
`setup_env.bat` builds a venv with CUDA PyTorch. `run_ner1.bat` downloads BC5CDR/NCBI-disease from the official sources, trains, and writes results to `ner/models/run1`. `run_eval.bat` runs the baseline comparison and ONNX export. See `ner/scripts/` and the model card for details and limitations.

`COMPANY_STATE.md` is the running log (hypotheses, dataset licenses, evals, costs). `DAILY.md` is the daily routine.

## License
Apache-2.0 (see `LICENSE` and `NOTICE`). Training data licenses are recorded in `COMPANY_STATE.md` and the model card.
