$scriptPath = split-path -parent $MyInvocation.MyCommand.Definition
$projectPath = (get-item $scriptPath ).parent.Fullname
$dataPath = Join-Path $projectPath "data"
$modelsPath = Join-Path $projectPath "models"

docker run -it --rm -v ${dataPath}:/opt/data -v ${modelsPath}:/opt/models --name test lucasmfaria/image_classifier:latest python ./scripts/test.py
pause