import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
import shutil
import tensorflow as tf
import tensorflow_datasets as tfds
from imblearn.under_sampling import RandomUnderSampler


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
    df = df.assign(y=np.argmax(df.values, axis=1))
    return df


def train_test_valid_split(dataset_source_path, test_size=0.15, valid_size=0.15, shuffle=True,
                           under_sample_ratio=None, random_state=None):
    '''

    :param under_sample_ratio:
    :param dataset_source_path:
    :param test_size:
    :param valid_size:
    :param shuffle:
    :param random_state:
    :return:
    '''

    df = create_aux_dataframe(dataset_source_path)
    train, test = train_test_split(df, test_size=test_size, shuffle=shuffle, random_state=random_state,
                                   stratify=df.y)  # test split uses stratification technique and gets the
    # "real data distribution"

    if under_sample_ratio:  # train and valid splits distributions are controlled by the under_sample_ratio parameter,
        # if it is used
        train = train.assign(file=train.index).reset_index(drop=True)  # generate "file" column because the
        # RandomUnderSample resets the index
        rus = RandomUnderSampler(sampling_strategy=under_sample_ratio, replacement=False, random_state=random_state)
        train, _ = rus.fit_resample(train, train.y)
        train.index = train.file
        train.drop(['file'], axis=1, inplace=True)

    valid_split = valid_size / (1 - test_size)
    train, valid = train_test_split(train, test_size=valid_split, shuffle=shuffle, random_state=random_state,
                                    stratify=train.y)
    return train, test, valid


def filter_binary_labels(image, label):
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
