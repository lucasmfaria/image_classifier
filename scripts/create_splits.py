from pathlib import Path
import argparse
import sys
try:
    from utils.data import train_test_valid_split, create_split, delete_folder
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
    from utils.data import train_test_valid_split, create_split, delete_folder

parser = argparse.ArgumentParser()

parser.add_argument('--test_size', type=float, help='Test size as float', default=0.15)
parser.add_argument('--valid_size', type=float, help='Validation size as float', default=0.15)
DEFAULT_DATASET_SOURCE_PATH = Path(__file__).parent.parent / 'data' / 'dataset'
parser.add_argument('--dataset_path', type=str, help='Image dataset source path', default=DEFAULT_DATASET_SOURCE_PATH)
DEFAULT_SPLITS_DESTINATION = Path(__file__).parent.parent / 'data'
parser.add_argument('--splits_dest_path', type=str, help='Splits destination path', default=DEFAULT_SPLITS_DESTINATION)
parser.add_argument('--undersample_ratio', type=float, help='Ratio used to under sample the majority classes',
                    default=None)
args = parser.parse_args()

dataset_path = Path(args.dataset_path)
splits_destination_path = Path(args.splits_dest_path)
undersample_ratio = args.undersample_ratio

X_train, X_test, X_valid = train_test_valid_split(dataset_path, test_size=args.test_size, valid_size=args.valid_size,
                                                  under_sample_ratio=undersample_ratio)

splits = [('train', X_train), ('test', X_test), ('valid', X_valid)]

for split in splits:
    destination_path = Path(splits_destination_path) / split[0]
    delete_folder(destination_path)
    create_split(split[1], destination_path)
