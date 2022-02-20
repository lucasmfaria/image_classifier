$scriptPath = split-path -parent $MyInvocation.MyCommand.Definition
$projectPath = (get-item $scriptPath ).parent.parent.Fullname
$dataPath = Join-Path $projectPath "data"

docker run --gpus all -it --rm -v ${dataPath}:/opt/data --name create_splits lucasmfaria/image_classifier:latest-gpu python ./scripts/create_splits.py
pause