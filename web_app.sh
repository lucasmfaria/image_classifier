#!/bin/bash

python -m pip install virtualenv
python -m pip install --upgrade pip
DIR="./venv/"
if [ -d "$DIR" ]; then
   echo "'$DIR' found..."
else
   python -m virtualenv venv
fi ;

source ./venv/bin/activate
python -m pip install -r requirements.txt
streamlit run web_app.py
