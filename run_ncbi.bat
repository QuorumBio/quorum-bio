@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
for /f "usebackq tokens=1,* delims==" %%a in (".env") do set %%a=%%b
echo start %date% %time% > logs\ncbi.txt
python ner\scripts\ncbi_compare.py >> logs\ncbi.txt 2>&1
echo RUN_END >> logs\ncbi.txt
exit
