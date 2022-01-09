from pathlib import Path
import argparse
import sys
try:
    from utils.data import train_test_valid_split, create_split, delete_folder
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.data import train_test_valid_split, create_split, delete_folder

parser = argparse.ArgumentParser()

parser.add_argument('--test_size', type=float, help='Test size as float', default=0.15)
parser.add_argument('--valid_size', type=float, help='Validation size as float', default=0.15)
DEFAULT_DATASET_SOURCE_PATH = Path(__file__).parent.parent / 'data' / 'dataset'
parser.add_argument('--dataset_path', type=str, help='Image dataset source path', default=DEFAULT_DATASET_SOURCE_PATH)
DEFAULT_SPLITS_DESTINATION = Path(__file__).parent.parent / 'data'
parser.add_argument('--splits_dest_path', type=str, help='Splits destination path', default=DEFAULT_SPLITS_DESTINATION)
group = parser.add_mutually_exclusive_group()  # oversample_ratio and undersample_ratio are mutually exclusives
group.add_argument('--oversample_ratio', type=float, help='Ratio used to over sample the minority classes',
                    default=None)
group.add_argument('--undersample_ratio', type=float, help='Ratio used to under sample the majority classes',
                   default=None)
parser.add_argument('--seed', type=int, help='Seed number for reproducibility', default=None)
args = parser.parse_args()


def main(dataset_path=DEFAULT_DATASET_SOURCE_PATH, splits_destination_path=DEFAULT_SPLITS_DESTINATION, test_size=0.15,
         valid_size=0.15, undersample_ratio=None, oversample_ratio=None, seed=None, streamlit_callbacks=None):
    # create the splits dataframes:
    x_train, x_test, x_valid = train_test_valid_split(Path(dataset_path), test_size=test_size,
                                                      valid_size=valid_size,
                                                      undersample_ratio=undersample_ratio,
                                                      oversample_ratio=oversample_ratio, random_state=seed)
    splits = [('train', x_train), ('test', x_test), ('valid', x_valid)]
    # delete and copy the images:
    for split in splits:
        destination_path = Path(splits_destination_path) / split[0]
        delete_folder(destination_path, streamlit_callbacks=streamlit_callbacks)
        create_split(split[1], destination_path, streamlit_callbacks=streamlit_callbacks)


if __name__ == '__main__':
    main(dataset_path=Path(args.dataset_path), splits_destination_path=args.splits_dest_path, test_size=args.test_size,
         valid_size=args.valid_size, undersample_ratio=args.undersample_ratio, oversample_ratio=args.oversample_ratio,
         seed=args.seed)
