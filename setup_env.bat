@echo off
cd /d "%~dp0"
if not exist logs mkdir logs
set LOG=logs\setup_env.txt
echo setup start %date% %time% > %LOG%
if not exist venv python -m venv venv >> %LOG% 2>&1
call venv\Scripts\activate.bat
python -m pip install --upgrade pip >> %LOG% 2>&1
echo --- torch cu130 >> %LOG%
pip install torch --index-url https://download.pytorch.org/whl/cu130 >> %LOG% 2>&1
python -c "import torch;print(torch.cuda.is_available())" > logs\cuda_ok.txt 2>&1
findstr /i "True" logs\cuda_ok.txt >nul || (
  echo --- cu130 failed, trying cu128 >> %LOG%
  pip install torch --index-url https://download.pytorch.org/whl/cu128 --force-reinstall >> %LOG% 2>&1
)
echo --- libs >> %LOG%
pip install transformers datasets accelerate evaluate seqeval onnx onnxruntime optimum huggingface_hub gradio scikit-learn >> %LOG% 2>&1
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" >> %LOG% 2>&1
echo SETUP_DONE >> %LOG%
exit
