from pathlib import Path
import shutil
from tqdm import tqdm
import sys
import os
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from utils.data import train_test_valid_split

DATASET_SOURCE_PATH = r'../data/dataset'
SPLITS_DESTINATION = r'../data'

X_train, X_test, X_valid = train_test_valid_split(test_size=0.15, valid_size=0.15)

split = 'train'
destination_path = Path(SPLITS_DESTINATION) / split
print('--------------DELETE' + split.upper() + 'SPLIT------------')
for directory in destination_path.iterdir():
    if directory.is_dir():
        shutil.rmtree(directory)
print('--------------COPY' + split.upper() + 'SPLIT------------')
for idx, _ in tqdm(X_train.iterrows(), total=X_train.shape[0]):
    destination = (destination_path / idx.parent.name) / idx.name
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.copy(idx, destination)

split = 'test'
destination_path = Path(SPLITS_DESTINATION) / split
print('--------------DELETE' + split.upper() + 'SPLIT------------')
for directory in destination_path.iterdir():
    if directory.is_dir():
        shutil.rmtree(directory)
print('--------------COPY' + split.upper() + 'SPLIT------------')
for idx, _ in tqdm(X_test.iterrows(), total=X_test.shape[0]):
    destination = (destination_path / idx.parent.name) / idx.name
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.copy(idx, destination)

split = 'valid'
destination_path = Path(SPLITS_DESTINATION) / split
print('--------------DELETE' + split.upper() + 'SPLIT------------')
for directory in destination_path.iterdir():
    if directory.is_dir():
        shutil.rmtree(directory)
print('--------------COPY' + split.upper() + 'SPLIT------------')
for idx, _ in tqdm(X_valid.iterrows(), total=X_valid.shape[0]):
    destination = (destination_path / idx.parent.name) / idx.name
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.copy(idx, destination)