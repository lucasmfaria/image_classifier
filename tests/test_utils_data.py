import sys
from pathlib import Path
try:
    from utils.data import create_aux_dataframe, train_test_valid_split, filter_binary_labels, optimize_dataset, \
        delete_folder, create_split
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.data import create_aux_dataframe, train_test_valid_split, filter_binary_labels, optimize_dataset, \
        delete_folder, create_split

