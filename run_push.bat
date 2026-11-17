@echo off
cd /d "%~dp0"
for /f "usebackq tokens=1,* delims==" %%a in (".env") do set %%a=%%b
echo start %date% %time% > logs\push.txt
if not exist .git git init -b main >> logs\push.txt 2>&1
git config user.name "Quorum Bio"
git config user.email "quorumbio@users.noreply.github.com"
git add -A >> logs\push.txt 2>&1
git status --short >> logs\push.txt 2>&1
git commit -m "Initial commit: NER training, eval, ONNX export, release scripts, state log" >> logs\push.txt 2>&1
git push https://x-access-token:%GH_TOKEN%@github.com/QuorumBio/quorum-bio.git main >> logs\push.txt 2>&1
git ls-files > logs\push_files.txt 2>&1
echo RUN_END >> logs\push.txt
exit
