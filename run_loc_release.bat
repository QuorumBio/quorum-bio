@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
for /f "usebackq tokens=1,* delims==" %%a in (".env") do set %%a=%%b
echo start %date% %time% > logs\loc_release.txt
python loc\scripts\release_loc.py >> logs\loc_release.txt 2>&1
if errorlevel 1 goto fail
python loc\scripts\upload_loc.py >> logs\loc_release.txt 2>&1
:fail
git add -A >> logs\loc_release.txt 2>&1
git commit -m "NER v2 and localization model: training, eval, release scripts, state" >> logs\loc_release.txt 2>&1
git push https://x-access-token:%GH_TOKEN%@github.com/QuorumBio/quorum-bio.git main >> logs\loc_release.txt 2>&1
echo RUN_END %date% %time% >> logs\loc_release.txt
exit
