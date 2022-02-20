#!/bin/bash

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
PROJECTPATH=$(dirname "$SCRIPTPATH")
DATAPATH="$PROJECTPATH/data"
MODELSPATH="$PROJECTPATH/models"

docker run --gpus all -it --rm -v ${DATAPATH}:/opt/data -v ${MODELSPATH}:/opt/models --name jupyter -p 8888:8888 lucasmfaria/image_classifier:latest-gpu jupyter notebook --ip 0.0.0.0 --no-browser --allow-root