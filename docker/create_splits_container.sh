#!/bin/bash

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
DATAPATH="$SCRIPTPATH/data"

docker run -it --rm -v ${DATAPATH}:/opt/data --name create_splits lucasmfaria/image_classifier:latest python ./scripts/create_splits.py