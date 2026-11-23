from pathlib import Path
from huggingface_hub import HfApi
root = Path(__file__).resolve().parents[1]
api = HfApi(); print("user:", api.whoami()["name"])
repo = "quorumbio/biomedical-ner-chemical-disease"
try:
    api.create_tag(repo, tag="v1.0", tag_message="v1: BiomedBERT NER trained on BC5CDR only"); print("tagged v1.0")
except Exception as e: print("tag:", type(e).__name__, str(e)[:200])
api.upload_folder(folder_path=str(root / "release"), repo_id=repo, delete_patterns=["crosscorpus_results.json"],
                  commit_message="v2: train on BC5CDR + NCBI-disease; updated card, ONNX, results")
info = api.model_info(repo, files_metadata=True)
print("MODEL FILES:", [(s.rfilename, s.size) for s in info.siblings]); print("downloads_30d:", info.downloads, "likes:", info.likes)
print("UPLOAD2_DONE")
