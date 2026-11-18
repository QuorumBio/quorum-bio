@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
for /f "usebackq tokens=1,* delims==" %%a in (".env") do set %%a=%%b
echo start %date% %time% > logs\v2.txt
python ner\scripts\eval_v2.py >> logs\v2.txt 2>&1
python ner\scripts\onnx_export2.py >> logs\v2.txt 2>&1
python ner\scripts\release_pack2.py >> logs\v2.txt 2>&1
echo RUN_END >> logs\v2.txt
exit
