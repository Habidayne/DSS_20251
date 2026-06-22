@echo off
REM ==== Gridbreaker DSS demo launcher (Windows) ====
cd /d "%~dp0"
echo Kiem tra streamlit...
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Cai dat thu vien lan dau...
    python -m pip install -r requirements.txt
)
echo Khoi dong giao dien DSS tai http://localhost:8501 ...
python -m streamlit run app.py
pause
