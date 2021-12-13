from pathlib import Path
import sys
try:
    from utils.data import train_test_valid_split, create_split, delete_folder
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
    from utils.data import train_test_valid_split, create_split, delete_folder

DATASET_SOURCE_PATH = Path(__file__).parent.parent / 'data' / 'dataset'
SPLITS_DESTINATION = Path(__file__).parent.parent / 'data'

X_train, X_test, X_valid = train_test_valid_split(DATASET_SOURCE_PATH, test_size=0.15, valid_size=0.15)

splits = ['train', 'test', 'valid']

for split in splits:
    destination_path = Path(SPLITS_DESTINATION) / split
    delete_folder(destination_path)
    create_split(X_train, destination_path)
