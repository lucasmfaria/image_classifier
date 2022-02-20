$scriptPath = split-path -parent $MyInvocation.MyCommand.Definition
$projectPath = (get-item $scriptPath ).parent.parent.Fullname
$dataPath = Join-Path $projectPath "data"
$modelsPath = Join-Path $projectPath "models"

docker run --gpus all -it --rm -v ${dataPath}:/opt/data -v ${modelsPath}:/opt/models --name jupyter -p 8888:8888 lucasmfaria/image_classifier:latest-gpu jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
pause