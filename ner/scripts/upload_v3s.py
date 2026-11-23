"""Publish NER v3 small: new model repo from ner/release3s, added to the Quorum Bio collection (no separate Space).
Run only after the release is approved. Verifies the local file list against SHA256SUMS before upload."""
import hashlib
from pathlib import Path
from huggingface_hub import HfApi
root = Path(__file__).resolve().parents[1]
REPO = "quorumbio/biomedical-ner-gene-chemical-disease-species-small"; SPACE = None
COLL = "quorumbio/quorum-bio-open-biology-models-6b03617169bc86a3fd82656b"
rel = root / "release3s"; api = HfApi(); print("user:", api.whoami()["name"])
listed = {l.split("  ", 1)[1]: l.split("  ", 1)[0] for l in (rel / "SHA256SUMS").read_text().splitlines()}
local = {p.relative_to(rel).as_posix() for p in rel.rglob("*") if p.is_file() and p.name != "SHA256SUMS"}
assert set(listed) == local, (set(listed) ^ local)
for f, h in listed.items(): assert hashlib.sha256((rel / f).read_bytes()).hexdigest() == h, f
assert not any("__pycache__" in f for f in local)
api.create_repo(REPO, repo_type="model", exist_ok=True)
api.upload_folder(folder_path=str(rel), repo_id=REPO, commit_message="NER v3 small: 6-layer distilled six-type model")
for item, t in ((REPO, "model"),):
    try: api.add_collection_item(COLL, item_id=item, item_type=t, exists_ok=True)
    except Exception as e: print("collection:", type(e).__name__, str(e)[:200])
info = api.model_info(REPO, files_metadata=True)
print("MODEL FILES:", [(s.rfilename, s.size) for s in info.siblings]); print("UPLOAD3_DONE")
