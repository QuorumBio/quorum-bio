@echo off
cd /d "%~dp0"
if not exist logs mkdir logs
echo === nvidia-smi === > logs\gpu_check.txt
nvidia-smi >> logs\gpu_check.txt 2>&1
echo. >> logs\gpu_check.txt
echo === where python/conda/git/gh === >> logs\gpu_check.txt
where python >> logs\gpu_check.txt 2>&1
where conda >> logs\gpu_check.txt 2>&1
where git >> logs\gpu_check.txt 2>&1
where gh >> logs\gpu_check.txt 2>&1
echo. >> logs\gpu_check.txt
echo === torch === >> logs\gpu_check.txt
python -c "import torch,sys;print(sys.version);print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.version.cuda);print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')" >> logs\gpu_check.txt 2>&1
echo. >> logs\gpu_check.txt
echo === pip list (ml) === >> logs\gpu_check.txt
pip list 2>nul | findstr /i "transformers datasets torch onnx optimum huggingface gradio accelerate" >> logs\gpu_check.txt
echo done >> logs\gpu_check.txt
exit
