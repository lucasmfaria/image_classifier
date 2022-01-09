import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
import shutil
import tensorflow as tf
import tensorflow_datasets as tfds
import uuid
import argparse
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

DEFAULT_TRAIN_PATH = Path(__file__).resolve().parent.parent / 'data' / 'train'
DEFAULT_VALID_PATH = Path(__file__).resolve().parent.parent / 'data' / 'valid'


def create_aux_dataframe(dataset_path):
    """
    Generates an auxiliary pandas.DataFrame that maps each image file inside the dataset directories to its class.
    This makes it easy to split the data into train, test and validation splits.
    :param dataset_path: str or pathlib.Path
        Path to the dataset directory
    :return: pandas.DataFrame
        Index as the file_path (pathlib.Path object) and columns as class (one hot encoded matrix and label vector)
    """
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
    df = df.assign(y=np.argmax(df.values, axis=1))
    return df


def train_test_valid_split(dataset_source_path, test_size=0.15, valid_size=0.15, shuffle=True,
                           undersample_ratio=None, oversample_ratio=None, random_state=None):
    """
    Generates the dataset train, test and validation splits as pandas.DataFrames. First the test split is taken from
    the whole dataset, preserving the original classes distribution, with stratification. After that, the sampling
    techniques are applied, if the user set them so. The manipulated data is then split into train and validation
    datasets.
    :param dataset_source_path: str or pathlib.Path
        Path to the dataset directory
    :param test_size: float
        test split size as float (example 15% -> 0.15)... calculated with respect to the whole dataset
    :param valid_size: float
        validation split size as float (example 15% -> 0.15)... calculated with respect to the train split data, after
        undersampling or oversampling (controlled by the user) if needed
    :param shuffle: bool
        sklearn.model_selection.train_test_split "shuffle" parameter... wether or not to shuffle the data before splitting
    :param undersample_ratio: float
        parameter to control the undersampling technique... if it's a binary classification problem, it is the same of
        imblearn.under_sampling.RandomUnderSampler "sampling_strategy"... if it's a multiclass classification, then it
        is used to calculate the maximum number of images to under sample from the majority classes related to the
        minority classes
    :param oversample_ratio: float
        parameter to control the oversampling technique... if it's a binary classification problem, it is the same of
        imblearn.over_sampling.RandomOverSampler "sampling_strategy"... if it's a multiclass classification, then it
        is used to calculate the minimum number of images to over sample from the minority classes related to the
        majority classes
    :param random_state: int
        used for reproducibility
    :return: tuple
        train pandas.DataFrame, test pandas.DataFrame and validation pandas.DataFrame
    """

    df = create_aux_dataframe(dataset_source_path)
    train, test = train_test_split(df, test_size=test_size, shuffle=shuffle, random_state=random_state,
                                   stratify=df.y)  # test split uses stratification technique and gets the
    # "real data distribution"

    if (undersample_ratio is not None) and (oversample_ratio is None):  # train and valid splits distributions are controlled by the undersample_ratio parameter,
        # if it is used
        train = train.assign(file=train.index).reset_index(drop=True)  # generate "file" column because the
        # RandomUnderSample resets the index
        if np.unique(train.y).shape[0] == 2:  # binary case
            rus = RandomUnderSampler(sampling_strategy=undersample_ratio, replacement=False, random_state=random_state)
        else:  # multiclass case
            under_sample_dict = {}
            classes_examples_dict = train.groupby('y')['y'].count().to_dict()
            classes_examples_dict = dict(sorted(classes_examples_dict.items(), key=lambda item: item[1]))
            for idx, class_ in enumerate(classes_examples_dict.keys()):
                if idx == 0:  # starts with the minority class
                    n_max = int(classes_examples_dict[class_]/undersample_ratio)
                    under_sample_dict[class_] = classes_examples_dict[class_]
                else:
                    under_sample_dict[class_] = n_max if classes_examples_dict[class_] > n_max else classes_examples_dict[class_]
            rus = RandomUnderSampler(sampling_strategy=under_sample_dict, replacement=False, random_state=random_state)
        train, _ = rus.fit_resample(train, train.y)
        train.index = train.file
        train.drop(['file'], axis=1, inplace=True)
    elif (undersample_ratio is None) and (oversample_ratio is not None):
        train = train.assign(file=train.index).reset_index(drop=True)  # generate "file" column because the
        if np.unique(train.y).shape[0] == 2:  # binary case
            ros = RandomOverSampler(sampling_strategy=oversample_ratio, shrinkage=None, random_state=random_state)
        else:  # multiclass case
            over_sample_dict = {}
            classes_examples_dict = train.groupby('y')['y'].count().to_dict()
            classes_examples_dict = dict(sorted(classes_examples_dict.items(), key=lambda item: item[1], reverse=True))
            for idx, class_ in enumerate(classes_examples_dict.keys()):
                if idx == 0:  # starts with the majority class
                    n_min = int(classes_examples_dict[class_] * oversample_ratio)
                    over_sample_dict[class_] = classes_examples_dict[class_]
                else:
                    over_sample_dict[class_] = n_min if classes_examples_dict[class_] < n_min else classes_examples_dict[class_]
            ros = RandomOverSampler(sampling_strategy=over_sample_dict, shrinkage=None, random_state=random_state)
        train, _ = ros.fit_resample(train, train.y)
        train.index = train.file
        train.drop(['file'], axis=1, inplace=True)

    valid_split = valid_size / (1 - test_size)  # TODO: check if its correct
    train, valid = train_test_split(train, test_size=valid_split, shuffle=shuffle, random_state=random_state,
                                    stratify=train.y)
    return train, test, valid


def filter_binary_labels(image, label):
    """
    Used if you need to transform a matrix of one hot encoded labels of a binary classification problem into a label
    vector of 0 and 1.
    :param image: tf.Tensor batch
    :param label: tf.Tensor batch
    :return: tuple
    """
    return image, tf.expand_dims(tf.math.argmax(label, axis=1), axis=1)


def optimize_dataset(ds, sample_dataset=None, batch_size=None):
    autotune = tf.data.AUTOTUNE
    if sample_dataset == 'mnist':
        ds = ds.batch(batch_size)
    elif sample_dataset == 'cats_vs_dogs':
        ds = ds.batch(batch_size)
    ds = ds.cache()
    # ds = ds.shuffle(num_examples)
    ds = ds.prefetch(autotune)
    return ds


def prepare_sample_dataset(sample_dataset, batch_size=64, img_height=224, img_width=224):
    if sample_dataset == 'mnist':
        (train_ds, valid_ds), ds_info = tfds.load(sample_dataset, split=['train', 'test'], shuffle_files=True,
                                                  as_supervised=True, with_info=True)
        # TODO - fix this class names - get from dataset
        class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        def preprocess(image, label):
            image = tf.cast(image, tf.float32) / 255.
            return tf.image.grayscale_to_rgb(tf.image.resize(image, size=(img_height, img_width))), \
                                             tf.one_hot(label, depth=len(class_names), dtype=tf.uint8)

        train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.batch(batch_size)
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

        valid_ds = valid_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        valid_ds = valid_ds.batch(batch_size)
        valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)

        return train_ds, valid_ds, class_names
    # TODO - include other sample datasets


def dataset_definition(train_path=DEFAULT_TRAIN_PATH, valid_path=DEFAULT_VALID_PATH, sample_dataset=None, batch_size=64,
                       img_height=224, img_width=224, seed=None, unit_test_dataset=False):
    if sample_dataset in ['mnist']:  # loads a sample dataset for user/unit testing
        train_ds, valid_ds, class_names = prepare_sample_dataset(sample_dataset=sample_dataset, batch_size=batch_size,
                                                                 img_height=img_height, img_width=img_width)

    else:  # loads a user defined dataset in path
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(train_path,
                                                                       image_size=(img_height, img_width),
                                                                       batch_size=batch_size, shuffle=True,
                                                                       label_mode='categorical', seed=seed)

        valid_ds = tf.keras.preprocessing.image_dataset_from_directory(valid_path,
                                                                       image_size=(img_height, img_width),
                                                                       batch_size=batch_size, shuffle=True,
                                                                       label_mode='categorical', seed=seed)

        class_names = train_ds.class_names
        assert class_names == valid_ds.class_names

        AUTOTUNE = tf.data.AUTOTUNE

        if len(class_names) == 2:  # take the one-hot-encoded matrix of labels and convert to a vector if binary classification
            train_ds = train_ds.map(filter_binary_labels, num_parallel_calls=AUTOTUNE)
            valid_ds = valid_ds.map(filter_binary_labels, num_parallel_calls=AUTOTUNE)
        train_ds = optimize_dataset(train_ds)
        valid_ds = optimize_dataset(valid_ds)

    if unit_test_dataset:  # take only some elements of dataset, only used for unit testing
        train_ds = train_ds.take(5)
        valid_ds = valid_ds.take(5)

    return train_ds, valid_ds, class_names


def delete_folder(destination_path):
    """
    Deletes the last train, test and validation splits directories, if they already exist.
    :param destination_path: str or pathlib.Path
    """
    if Path(destination_path).exists():
        print('--------------DELETE ' + destination_path.name.upper() + ' SPLIT------------')
        for directory in destination_path.iterdir():
            if directory.is_dir():
                shutil.rmtree(directory)


def create_split(split, destination_path):
    """
    Creates the dataset split given by the pandas.DataFrame. Each image given by the DataFrame is copied to the
    split destination directory, together with its class information.
    :param split: pandas.DataFrame
    :param destination_path: pathlib.Path
    """
    print('--------------COPY ' + destination_path.name.upper() + ' SPLIT------------')
    for idx, _ in tqdm(split.iterrows(), total=split.shape[0]):
        destination = (destination_path / idx.parent.name) / idx.name
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        if destination.exists():  # used in oversampling case
            filename_append = str(uuid.uuid4())
            destination = destination.parent / (destination.name.split('.')[0] + filename_append + '.' +
                                                destination.name.split('.')[-1])
        shutil.copy(idx, destination)


def get_platform_shell():
    """
    Used for subprocess.Run "shell" parameter. Changes behavior between Windows and Unix platforms.
    :return: bool
    """
    if os.name == 'nt':
        shell = True
    elif os.name == 'posix':
        shell = False
    return shell


def true_or_false(arg):
    """
    Used for boolean arg parsing
    :param arg: str
    :return: bool
    """
    upper_arg = str(arg).upper()
    if 'TRUE'.startswith(upper_arg):
       return True
    elif 'FALSE'.startswith(upper_arg):
       return False
    else:
       raise argparse.ArgumentError(arg)
