#!/bin/bash

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
DATAPATH="$SCRIPTPATH/data"
MODELSPATH="$SCRIPTPATH/models"

docker run -it --rm -v ${DATAPATH}:/opt/data -v ${MODELSPATH}:/opt/models --name test lucasmfaria/image_classifier:latest python ./scripts/test.py