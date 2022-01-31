#!/bin/bash

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
DATAPATH="$SCRIPTPATH/data"
MODELSPATH="$SCRIPTPATH/models"

docker run -it --rm -v ${dataPath}:/opt/data -v ${modelsPath}:/opt/models --name streamlit -p 8501:8501 lucasmfaria/image_classifier:latest streamlit run web_app.py