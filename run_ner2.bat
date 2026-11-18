@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
for /f "usebackq tokens=1,* delims==" %%a in (".env") do set %%a=%%b
echo start %date% %time% > logs\ner2.txt
python ner\scripts\train2.py --base microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext --out ner\models\run2a >> logs\ner2.txt 2>&1
python ner\scripts\train2.py --base microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext --mask_o_in ncbi_disease --out ner\models\run2b >> logs\ner2.txt 2>&1
echo RUN_END >> logs\ner2.txt
exit
