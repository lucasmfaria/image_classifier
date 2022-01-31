#!/bin/bash

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
DATAPATH="$SCRIPTPATH/data"
MODELSPATH="$SCRIPTPATH/models"

docker run -it --rm -v ${dataPath}:/opt/data -v ${modelsPath}:/opt/models --name jupyter -p 8888:8888 lucasmfaria/image_classifier:latest jupyter notebook --ip 0.0.0.0 --no-browser --allow-root