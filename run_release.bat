@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
for /f "usebackq tokens=1,* delims==" %%a in (".env") do set %%a=%%b
echo start %date% %time% > logs\release.txt
python ner\scripts\release_pack.py >> logs\release.txt 2>&1
echo RUN_END >> logs\release.txt
exit
