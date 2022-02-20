#!/bin/bash

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
PROJECTPATH=$(dirname "$SCRIPTPATH")
DATAPATH="$PROJECTPATH/data"
MODELSPATH="$PROJECTPATH/models"

docker run --gpus all -it --rm -v ${DATAPATH}:/opt/data -v ${MODELSPATH}:/opt/models --name streamlit -p 8501:8501 lucasmfaria/image_classifier:latest-gpu streamlit run web_app.py