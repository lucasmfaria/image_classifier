#!/bin/bash

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
PROJECTPATH=$(dirname "$SCRIPTPATH")
DATAPATH="$PROJECTPATH/data"

docker run -it --rm -v ${DATAPATH}:/opt/data --name create_splits lucasmfaria/image_classifier:latest python ./scripts/create_splits.py