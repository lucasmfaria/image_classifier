import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

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

    :param df:
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
    X_train, X_test = train_test_split(df, test_size=test_size, shuffle=shuffle, random_state=random_state)

    valid_split = valid_size / (1 - test_size)
    X_train, X_valid = train_test_split(X_train, test_size=valid_split, shuffle=shuffle, random_state=random_state)

    return X_train, X_test, X_valid
