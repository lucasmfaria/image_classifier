@echo off

call python -m pip install virtualenv
call python -m pip install --upgrade pip
if not exist venv\ (
  call python -m virtualenv venv
)
call venv/Scripts/activate
call python -m pip install -r requirements.txt
call streamlit run web_app.py
pause