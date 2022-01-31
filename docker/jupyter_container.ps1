$scriptPath = split-path -parent $MyInvocation.MyCommand.Definition
$projectPath = (get-item $scriptPath ).parent.Fullname
$dataPath = Join-Path $projectPath "data"
$modelsPath = Join-Path $projectPath "models"

docker run -it --rm -v ${dataPath}:/opt/data -v ${modelsPath}:/opt/models --name jupyter -p 8888:8888 lucasmfaria/image_classifier:latest jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
pause