$scriptPath = split-path -parent $MyInvocation.MyCommand.Definition
$projectPath = (get-item $scriptPath ).parent.parent.Fullname
$dataPath = Join-Path $projectPath "data"
$modelsPath = Join-Path $projectPath "models"

docker run --gpus all -it --rm -v ${dataPath}:/opt/data -v ${modelsPath}:/opt/models --name train lucasmfaria/image_classifier:latest-gpu python ./scripts/train.py
pause