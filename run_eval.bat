@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
for /f "usebackq tokens=1,* delims==" %%a in (".env") do set %%a=%%b
echo start %date% %time% > logs\eval.txt
python ner\scripts\prepare_data.py > logs\prep2.txt 2>&1
python ner\scripts\baseline_eval.py >> logs\eval.txt 2>&1
python ner\scripts\onnx_export.py >> logs\eval.txt 2>&1
echo RUN_END >> logs\eval.txt
exit
