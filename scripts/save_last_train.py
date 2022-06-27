from pathlib import Path
import os
import shutil
import argparse
import pandas as pd
import sys
try:
    from scripts.test import main as test_main
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.test import main as test_main

parser = argparse.ArgumentParser()

# TODO - insert default values from dynaconf
DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / 'models' / 'vgg16' / 'checkpoints' / 'trained_weights'
parser.add_argument('--weights_path', type=str, help='Path of the final model weights file',
                    default=DEFAULT_WEIGHTS_PATH)
DEFAULT_SAVED_MODELS_PATH = Path(__file__).resolve().parent.parent / 'models' / 'vgg16' / 'saved_models'
parser.add_argument('--saved_models_path', type=str, help='Path of the saved models directory',
                    default=DEFAULT_SAVED_MODELS_PATH)
parser.add_argument('--img_height', type=int, help='Image height after resize', default=224)
parser.add_argument('--img_width', type=int, help='Image width after resize', default=224)
parser.add_argument('--batch_size', type=int, help='Batch size for training', default=64)
parser.add_argument('--sample_dataset', type=str, help='Name of sample dataset in [mnist, patch_camelyon]',
                    default=None)

args = parser.parse_args()


def main(saved_models_path=DEFAULT_SAVED_MODELS_PATH, weights_path=DEFAULT_WEIGHTS_PATH, sample_dataset=None, 
         batch_size=64, img_height=224, img_width=224):
    existing_saved_models = [folder for folder in saved_models_path.iterdir() if folder.is_dir()]
    new_saved_model_path = saved_models_path / ('model_' + str(len(existing_saved_models)))
    os.mkdir(new_saved_model_path)  # create the new directory based on the number of older directories
    trained_model_file_paths = [file for file in Path(weights_path).parent.iterdir() if
                                Path(weights_path).name in file.name]  # gets the source paths to the trained model files

    for file in trained_model_file_paths:  # copies the model weights files
        shutil.copy(file, new_saved_model_path / file.name)

    classification_report_dict, df_confusion_matrix, precision_recall, roc = test_main(sample_dataset=sample_dataset, weights_path=Path(weights_path), 
                                                                return_results=True, batch_size=batch_size, img_height=img_height, 
                                                                img_width=img_width)
    # TODO - save precision recall and roc curves
    with open(new_saved_model_path / 'results.txt', 'w') as file_txt:
        results_text = pd.DataFrame(classification_report_dict).T.to_string()
        file_txt.write(results_text)
        file_txt.write("\n\n")
        results_text = df_confusion_matrix.to_string()
        file_txt.write(results_text)


if __name__ == '__main__':
    main(saved_models_path=Path(args.saved_models_path), weights_path=Path(args.weights_path), sample_dataset=args.sample_dataset, 
         batch_size=args.batch_size, img_height=args.img_height, img_width=args.img_width)
