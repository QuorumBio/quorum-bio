@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
for /f "usebackq tokens=1,* delims==" %%a in (".env") do set %%a=%%b
echo start %date% %time% > logs\loc1.txt
python loc\scripts\train_loc.py --base facebook/esm2_t12_35M_UR50D --out loc\models\run1_35m --epochs 3 >> logs\loc1.txt 2>&1
echo STAGE_35M_DONE %date% %time% >> logs\loc1.txt
python loc\scripts\train_loc.py --base facebook/esm2_t30_150M_UR50D --out loc\models\run1_150m --epochs 2 >> logs\loc1.txt 2>&1
echo RUN_END %date% %time% >> logs\loc1.txt
exit
