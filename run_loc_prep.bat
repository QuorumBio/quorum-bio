@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
for /f "usebackq tokens=1,* delims==" %%a in (".env") do set %%a=%%b
echo start %date% %time% > logs\loc_prep.txt
pip install scikit-learn >> logs\loc_prep.txt 2>&1
python loc\scripts\prepare_uniprot.py >> logs\loc_prep.txt 2>&1
python loc\scripts\train_loc.py --base facebook/esm2_t12_35M_UR50D --out loc\models\smoke --smoke --epochs 1 >> logs\loc_prep.txt 2>&1
echo RUN_END >> logs\loc_prep.txt
exit
