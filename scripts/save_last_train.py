from pathlib import Path
import os
import shutil
import subprocess

CHECKPOINTS_FOLDER = Path(__file__).resolve().parent.parent / 'models' / 'vgg16' / 'checkpoints'
SAVED_MODELS_PATH = Path(__file__).resolve().parent.parent / 'models' / 'vgg16' / 'saved_models'
TRAINED_MODEL_NAME = 'trained_weights'

existing_saved_models = [folder for folder in SAVED_MODELS_PATH.iterdir() if folder.is_dir()]
new_saved_model_path = SAVED_MODELS_PATH / ('model_' + str(len(existing_saved_models)))
os.mkdir(new_saved_model_path)  # create the new directory based on the number of older directories
trained_model_file_paths = [file for file in CHECKPOINTS_FOLDER.iterdir() if TRAINED_MODEL_NAME in file.name]  # gets
# the source paths to the trained model files

for file in trained_model_file_paths:  # copies the model weights files
    shutil.copy(file, new_saved_model_path / file.name)

# saves the test split statistics for the trained model
p = subprocess.run(['python', str(Path(r'./scripts/test.py'))], shell=True, check=True, stdout=subprocess.PIPE)

with open(new_saved_model_path / 'results.txt', 'w') as file_txt:
    results_text = p.stdout.decode()
    results_text = results_text[results_text.find('%') + 1:]
    file_txt.write(results_text)
