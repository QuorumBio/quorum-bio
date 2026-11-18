@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
for /f "usebackq tokens=1,* delims==" %%a in (".env") do set %%a=%%b
echo start %date% %time% > logs\upload2.txt
python ner\scripts\upload_v2.py >> logs\upload2.txt 2>&1
echo RUN_END >> logs\upload2.txt
exit
