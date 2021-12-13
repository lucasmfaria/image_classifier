import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
import shutil
import tensorflow as tf


def create_aux_dataframe(dataset_path):
    '''
    Generate
    :param dataset_path:
    :return:
    '''
    directory_list = [directory for directory in Path(dataset_path).iterdir() if os.path.isdir(directory)]
    df = pd.DataFrame(columns=[directory.name for directory in directory_list])
    for directory in directory_list:
        files_path_list = [file.resolve() for file in directory.iterdir()]
        df_ = pd.DataFrame(index=files_path_list, columns=df.columns)
        for col in df_.columns:
            if col == directory.name:
                one_hot_vector = np.ones(shape=(df_.shape[0],))
                df_[col] = one_hot_vector
            else:
                one_hot_vector = np.zeros(shape=(df_.shape[0],))
                df_[col] = one_hot_vector
        df = df.append(df_)
    return df


def train_test_valid_split(dataset_source_path, test_size=0.15, valid_size=0.15, shuffle=True, test_stratify=True, valid_stratify=True, \
                           under_sample=None, random_state=None):
    '''

    :param dataset_source_path:
    :param test_size:
    :param valid_size:
    :param shuffle:
    :param test_stratify:
    :param valid_stratify:
    :param under_sample:
    :param random_state:
    :return:
    '''

    df = create_aux_dataframe(dataset_source_path)
    train, test = train_test_split(df, test_size=test_size, shuffle=shuffle, random_state=random_state)

    valid_split = valid_size / (1 - test_size)
    train, valid = train_test_split(train, test_size=valid_split, shuffle=shuffle, random_state=random_state)

    return train, test, valid


def filter_binary_labels(image, label):
    return image, tf.expand_dims(tf.math.argmax(label, axis=1), axis=1)


def optimize_dataset(ds):
    autotune = tf.data.AUTOTUNE
    ds = ds.cache()
    # ds = ds.shuffle(num_examples)
    ds = ds.prefetch(autotune)
    return ds


def delete_folder(destination_path):
    if Path(destination_path).exists():
        print('--------------DELETE ' + destination_path.name.upper() + ' SPLIT------------')
        for directory in destination_path.iterdir():
            if directory.is_dir():
                shutil.rmtree(directory)


def create_split(split, destination_path):
    print('--------------COPY ' + destination_path.name.upper() + ' SPLIT------------')
    for idx, _ in tqdm(split.iterrows(), total=split.shape[0]):
        destination = (destination_path / idx.parent.name) / idx.name
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copy(idx, destination)
