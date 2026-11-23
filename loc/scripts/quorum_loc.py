"""Inference for quorumbio/esm2-protein-subcellular-localization-signal-peptide. Usage:
    from quorum_loc import load, predict
    m = load("path/to/repo")   # folder with model.safetensors + config.json
    predict(m, "MKTAYIAKQR...")"""
import json
from pathlib import Path
import torch, torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from safetensors.torch import load_file

class LocModel(nn.Module):
    def __init__(self, cfg, n_classes=10):
        super().__init__(); self.enc = AutoModel.from_config(cfg); H = cfg.hidden_size
        self.drop = nn.Dropout(0.1); self.loc = nn.Linear(H, n_classes); self.sp = nn.Linear(2 * H, 1)
    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype); pooled = (h * m).sum(1) / m.sum(1)
        pooled = self.drop(pooled); nterm = self.drop(h[:, 1:31].mean(1))
        return self.loc(pooled), self.sp(torch.cat([pooled, nterm], -1))[:, 0]

def trunc(seq): return seq if len(seq) <= 1000 else seq[:600] + seq[-400:]

def load(folder, device="cpu"):
    folder = Path(folder); meta = json.load(open(folder / "config.json"))
    cfg = AutoConfig.for_model(**meta["encoder_config"])
    m = LocModel(cfg, len(meta["classes"])); m.load_state_dict(load_file(folder / "model.safetensors")); m.eval().to(device)
    m.meta = meta; m.tok = AutoTokenizer.from_pretrained(folder); m.device_ = device
    return m

@torch.no_grad()
def predict(m, seq):
    seq = "".join(c for c in seq.upper() if c.isalpha()); t = m.tok([trunc(seq)], return_tensors="pt", truncation=True, max_length=1022)
    lg, sg = m(t["input_ids"].to(m.device_), t["attention_mask"].to(m.device_))
    p = torch.sigmoid(lg[0]).tolist(); sp = torch.sigmoid(sg[0]).item()
    return {"locations": {c: {"probability": round(x, 4), "predicted": x >= th} for c, x, th in zip(m.meta["classes"], p, m.meta["thresholds"])},
            "signal_peptide": {"probability": round(sp, 4), "predicted": sp >= m.meta["sp_threshold"]}}
