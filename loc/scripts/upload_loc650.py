"""Publish the ESM-2 650M localization model from loc/release650 as its own repo and add it to the collection.
Verifies the local file list and hashes against SHA256SUMS first. Then updates only README.md (+ SHA256SUMS) on the 150M repo to link here."""
import hashlib
from pathlib import Path
from huggingface_hub import HfApi
root = Path(__file__).resolve().parents[1]
REPO = "quorumbio/esm2-650m-protein-subcellular-localization-signal-peptide"; SMALL = "quorumbio/esm2-protein-subcellular-localization-signal-peptide"
COLL = "quorumbio/quorum-bio-open-biology-models-6b03617169bc86a3fd82656b"
api = HfApi(); print("user:", api.whoami()["name"])
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
def check(rel):
    listed = {l.split("  ", 1)[1]: l.split("  ", 1)[0] for l in (rel / "SHA256SUMS").read_text().splitlines()}
    local = {p.relative_to(rel).as_posix() for p in rel.rglob("*") if p.is_file() and p.name != "SHA256SUMS"}
    assert set(listed) == local, (set(listed) ^ local)
    for f, h in listed.items(): assert sha(rel / f) == h, f
    assert not any("__pycache__" in f for f in local)
    return listed
rel = root / "release650"; check(rel)
api.create_repo(REPO, repo_type="model", exist_ok=True)
api.upload_folder(folder_path=str(rel), repo_id=REPO, commit_message="ESM-2 650M subcellular localization + signal peptide (same data and split as the 150M model)")
try: api.add_collection_item(COLL, item_id=REPO, item_type="model", exists_ok=True)
except Exception as e: print("collection:", type(e).__name__, str(e)[:200])
small = check(root / "release")
for f in ("README.md", "SHA256SUMS"): api.upload_file(path_or_fileobj=str(root / "release" / f), path_in_repo=f, repo_id=SMALL, commit_message="Link to the 650M version")
info = api.model_info(REPO, files_metadata=True)
remote = {s.rfilename: (s.lfs.sha256 if s.lfs else None) for s in info.siblings}
for f, h in check(rel).items():
    assert f in remote, ("missing", f)
    if remote[f]: assert remote[f] == h, ("lfs hash", f)
print("MODEL FILES:", [(s.rfilename, s.size) for s in info.siblings]); print("UPLOAD650_DONE")
