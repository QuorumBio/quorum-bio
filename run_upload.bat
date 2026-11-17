@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
for /f "usebackq tokens=1,* delims==" %%a in (".env") do set %%a=%%b
echo start %date% %time% > logs\upload.txt
python ner\scripts\upload_release.py >> logs\upload.txt 2>&1
echo RUN_END >> logs\upload.txt
exit
