@echo off
cd /d "%~dp0"
for /f "usebackq tokens=1,* delims==" %%a in (".env") do set %%a=%%b
echo start %date% %time% > logs\push2.txt
git add -A >> logs\push2.txt 2>&1
git commit -m "NER v2: BC5CDR + NCBI-disease training, eval, ONNX, release scripts" >> logs\push2.txt 2>&1
git push https://x-access-token:%GH_TOKEN%@github.com/QuorumBio/quorum-bio.git main >> logs\push2.txt 2>&1
echo RUN_END >> logs\push2.txt
exit
