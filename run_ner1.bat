@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
for /f "usebackq tokens=1,* delims==" %%a in (".env") do set %%a=%%b
echo start %date% %time% > logs\ner1.txt
python ner\scripts\prepare_data.py >> logs\ner1.txt 2>&1
python ner\scripts\train.py --corpus bc5cdr --base microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext --out ner\models\run1 >> logs\ner1.txt 2>&1
echo RUN_END >> logs\ner1.txt
exit
