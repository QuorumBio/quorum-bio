from pathlib import Path
from huggingface_hub import HfApi
root = Path(__file__).resolve().parents[1]
api = HfApi(); print("user:", api.whoami()["name"])
repo = "quorumbio/biomed-ner-chem-disease"; space = "quorumbio/biomed-ner-demo"
api.create_repo(repo, exist_ok=True)
api.upload_folder(folder_path=str(root / "release"), repo_id=repo, commit_message="Release: BiomedBERT-base NER (Chemical, Disease), ONNX fp32 + int8, training scripts")
try:
    api.delete_file("onnx/model_int8.onnx", repo_id=repo, commit_message="Rename int8 file to model_quantized.onnx (transformers.js convention)")
except Exception as e:
    print("no old int8 file to delete:", type(e).__name__)
api.create_repo(space, repo_type="space", space_sdk="static", exist_ok=True)
api.upload_folder(folder_path=str(root / "space"), repo_id=space, repo_type="space", commit_message="Static browser demo (transformers.js)")
info = api.model_info(repo, files_metadata=True)
print("MODEL FILES:", [(s.rfilename, s.size) for s in info.siblings])
print("downloads_30d:", info.downloads, "likes:", info.likes)
print("UPLOAD_DONE", f"https://huggingface.co/{repo}", f"https://huggingface.co/spaces/{space}")
