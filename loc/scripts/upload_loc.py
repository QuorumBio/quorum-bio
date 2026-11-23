from pathlib import Path
from huggingface_hub import HfApi
root = Path(__file__).resolve().parents[1]; api = HfApi(); print("user:", api.whoami()["name"])
repo = "quorumbio/esm2-protein-subcellular-localization-signal-peptide"; space = "quorumbio/protein-localization-demo"
api.create_repo(repo, exist_ok=True)
api.upload_folder(folder_path=str(root / "release"), repo_id=repo, commit_message="Release: ESM-2 150M subcellular localization + signal peptide")
api.create_repo(space, repo_type="space", space_sdk="static", exist_ok=True)
api.upload_folder(folder_path=str(root / "space"), repo_id=space, repo_type="space", commit_message="Static browser demo (onnxruntime-web)")
info = api.model_info(repo, files_metadata=True); print("MODEL FILES:", [(s.rfilename, s.size) for s in info.siblings]); print("UPLOAD_LOC_DONE")
