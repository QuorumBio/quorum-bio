"""Build loc/release650 (ESM-2 650M run): safetensors weights + config, ONNX (fp32 with external data, int8) with parity check, LICENSE, NOTICE.
The card and SHA256SUMS are written afterwards by card_loc650.py (needs bootstrap_ci.json from bootstrap_loc.py)."""
import json, shutil, hashlib, time, traceback, sys
from pathlib import Path
import numpy as np, torch
from transformers import AutoConfig, AutoTokenizer
from safetensors.torch import save_file
sys.path.insert(0, str(Path(__file__).parent))
from quorum_loc import LocModel, trunc
root = Path(__file__).resolve().parents[1]; rel = root / "release650"; rel.mkdir(exist_ok=True); (rel / "onnx").mkdir(exist_ok=True); (rel / "training").mkdir(exist_ok=True)
REPO = "quorumbio/esm2-650m-protein-subcellular-localization-signal-peptide"; BASE = "facebook/esm2_t33_650M_UR50D"
run = root / "models" / "run650"; R = json.load(open(run / "results.json")); P = json.load(open(root / "data" / "provenance.json"))
ck = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
cfg = AutoConfig.from_pretrained(BASE); cfg._attn_implementation = "eager"
net = LocModel(cfg, len(ck["classes"])); net.load_state_dict(ck["state_dict"]); net.eval()
save_file({k: v.detach().clone().contiguous() for k, v in net.state_dict().items()}, str(rel / "model.safetensors"))
meta = {"classes": ck["classes"], "thresholds": ck["thresholds"], "sp_threshold": ck["sp_threshold"], "base": BASE, "encoder_config": cfg.to_dict(), "max_len_policy": "sequences over 1000 residues: first 600 + last 400"}
(rel / "config.json").write_text(json.dumps(meta, indent=1))
tok = AutoTokenizer.from_pretrained(BASE); tok.save_pretrained(rel)
(rel / "vocab_for_js.json").write_text(json.dumps(tok.get_vocab()))
test = [json.loads(l) for l in open(root / "data" / "test.jsonl")]
sub = [x for x in test if len(x["sequence"]) <= 500][:150]
class Wrap(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, ids, mask): return s.m(ids, mask)
onnx_res = {"status": "not_run"}
try:
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    t = tok([sub[0]["sequence"][:80], sub[1]["sequence"][:60]], return_tensors="pt", padding=True)
    path = rel / "onnx" / "model.onnx"; w = Wrap(net).eval()
    torch.onnx.export(w, (t["input_ids"], t["attention_mask"]), str(path), input_names=["input_ids", "attention_mask"], output_names=["loc_logits", "sp_logit"],
                      dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "loc_logits": {0: "batch"}, "sp_logit": {0: "batch"}}, opset_version=17, dynamo=False)
    import onnx
    big = onnx.load(str(path), load_external_data=True)  # >2 GB: keep weights in one external file next to model.onnx
    for f in (rel / "onnx").glob("*"): f.unlink()
    onnx.save_model(big, str(path), save_as_external_data=True, all_tensors_to_one_file=True, location="model.onnx_data", size_threshold=1024); del big
    q = rel / "onnx" / "model_quantized.onnx"; quantize_dynamic(str(path), str(q), weight_type=QuantType.QInt8, use_external_data_format=False)
    def mk(p): return ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    res = {}
    sess = {"fp32": mk(path), "int8": mk(q)}; maxd = {"fp32": 0.0, "int8": 0.0}; agree = {"fp32": [0, 0], "int8": [0, 0]}; thr = np.array(ck["thresholds"])
    for x in sub:
        e = tok([trunc(x["sequence"])], return_tensors="pt", truncation=True, max_length=1022)
        with torch.no_grad(): tl, ts = net(e["input_ids"], e["attention_mask"])
        tp = torch.sigmoid(tl)[0].numpy(); tsp = torch.sigmoid(ts)[0].item()
        for tag, s in sess.items():
            ol, os_ = s.run(None, {"input_ids": e["input_ids"].numpy().astype(np.int64), "attention_mask": e["attention_mask"].numpy().astype(np.int64)})
            op = 1 / (1 + np.exp(-ol[0])); osp = 1 / (1 + np.exp(-os_[0]))
            maxd[tag] = max(maxd[tag], float(np.abs(op - tp).max()), float(abs(osp - tsp)))
            agree[tag][0] += int(((op >= thr) == (tp >= thr)).sum()) + int((osp >= ck["sp_threshold"]) == (tsp >= ck["sp_threshold"])); agree[tag][1] += len(thr) + 1
    onnx_res = {"status": "ok", "sequences_checked": len(sub), "note": "sequences up to 500 residues", "files_mb": {"fp32": round((path.stat().st_size + (rel / "onnx" / "model.onnx_data").stat().st_size) / 1e6, 1), "int8": round(q.stat().st_size / 1e6, 1)},
                "max_abs_probability_diff_vs_torch": maxd, "decision_agreement_vs_torch": {k: v[0] / v[1] for k, v in agree.items()}}
except Exception as ex:
    traceback.print_exc(); onnx_res = {"status": "failed", "error": repr(ex)[:400]}
    for f in (rel / "onnx").glob("*"): f.unlink()
print("ONNX", json.dumps(onnx_res), flush=True)
(rel / "onnx_results.json").write_text(json.dumps(onnx_res, indent=2))
shutil.copy(run / "results.json", rel / "train_results_650m.json"); shutil.copy(root / "release" / "train_results_150m.json", rel / "train_results_150m.json"); shutil.copy(root / "data" / "provenance.json", rel / "training" / "provenance.json")
for f in ("prepare_uniprot.py", "train_loc.py", "quorum_loc.py", "release_loc650.py", "bootstrap_loc.py", "card_loc650.py"): shutil.copy(root / "scripts" / f, rel / "training" / f)
shutil.copy(root / "scripts" / "quorum_loc.py", rel / "quorum_loc.py")
for q_ in (rel / "quorum_loc.py", rel / "training" / "quorum_loc.py"): q_.write_text(q_.read_text(encoding="utf-8").replace("quorumbio/esm2-protein-subcellular-localization-signal-peptide", REPO), encoding="utf-8")
shutil.copy(root.parent / "ner" / "release" / "LICENSE", rel / "LICENSE")
mit = open(root.parent / "ner" / "release" / "NOTICE", encoding="utf-8").read().split("-----")[1].replace("Microsoft Corporation (model authors; see the upstream model card for the exact notice)", "Meta Platforms, Inc. and affiliates (ESM-2 authors; see the upstream repository for the exact notice)").replace("The upstream repository does not ship a LICENSE file; its model card lists the MIT license. https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", "The upstream model card lists the MIT license. https://huggingface.co/" + BASE)
(rel / "NOTICE").write_text(f"""Quorum Bio protein localization and signal peptide model
Copyright 2026 Quorum Bio. Licensed under the Apache License, Version 2.0 (see LICENSE).

This model is fine-tuned from {BASE} (ESM-2, Lin et al. 2023, Science), which is released under the MIT License:
-----{mit}-----

Training data: UniProtKB/Swiss-Prot (reviewed entries), The UniProt Consortium, Nucleic Acids Research. https://www.uniprot.org
UniProt data are licensed CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Retrieved 2026-11-17. Sequences and annotations were filtered and re-split by Quorum Bio (see training/).
""")
print("BUILD650_DONE", flush=True)
