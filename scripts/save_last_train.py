from pathlib import Path
import os
import shutil
import argparse
import pandas as pd
from test import main as test_main

parser = argparse.ArgumentParser()

# TODO - insert default values from dynaconf
DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / 'models' / 'vgg16' / 'checkpoints' / 'trained_weights'
parser.add_argument('--weights_path', type=str, help='Path of the final model weights file',
                    default=DEFAULT_WEIGHTS_PATH)
DEFAULT_SAVED_MODELS_PATH = Path(__file__).resolve().parent.parent / 'models' / 'vgg16' / 'saved_models'
parser.add_argument('--saved_models_path', type=str, help='Path of the saved models directory',
                    default=DEFAULT_SAVED_MODELS_PATH)

args = parser.parse_args()


def main(saved_models_path=DEFAULT_SAVED_MODELS_PATH, weights_path=DEFAULT_WEIGHTS_PATH):
    existing_saved_models = [folder for folder in saved_models_path.iterdir() if folder.is_dir()]
    new_saved_model_path = saved_models_path / ('model_' + str(len(existing_saved_models)))
    os.mkdir(new_saved_model_path)  # create the new directory based on the number of older directories
    trained_model_file_paths = [file for file in Path(weights_path).parent.iterdir() if
                                Path(weights_path).name in file.name]  # gets the source paths to the trained model files

    for file in trained_model_file_paths:  # copies the model weights files
        shutil.copy(file, new_saved_model_path / file.name)

    classification_report_dict, df_confusion_matrix = test_main(weights_path=Path(weights_path), return_results=True)

    with open(new_saved_model_path / 'results.txt', 'w') as file_txt:
        results_text = pd.DataFrame(classification_report_dict).T.to_string()
        file_txt.write(results_text)
        file_txt.write("\n\n")
        results_text = df_confusion_matrix.to_string()
        file_txt.write(results_text)


if __name__ == '__main__':
    main(saved_models_path=Path(args.saved_models_path), weights_path=Path(args.weights_path))
