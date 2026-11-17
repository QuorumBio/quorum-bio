@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
for /f "usebackq tokens=1,* delims==" %%a in (".env") do set %%a=%%b
python -c "from huggingface_hub import HfApi;i=HfApi().whoami();print('user',i['name'],'role',i.get('auth',{}).get('accessToken',{}).get('role'))" > logs\verify_hf.txt 2>&1
exit
