from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import vgg16
from tensorflow.keras import layers

def make_model(n_classes, include_top_vgg=False, n_hidden=512, img_height=224, img_width=224):
    '''

    :param n_classes:
    :param include_top_vgg:
    :param n_hidden:
    :param img_height:
    :param img_width:
    :return:
    '''
    vgg_model = vgg16.VGG16(include_top=False, pooling='max')

    data_augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip('horizontal'),
        layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    inputs = layers.Input(shape=(img_height, img_width, 3))
    x = data_augmentation(inputs)
    x = vgg16.preprocess_input(x)
    x = vgg_model(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(n_hidden, activation='relu', name='dense_1')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(n_hidden, activation='relu', name='dense_2')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(n_classes, activation='softmax', name='output')(x)
    model = tf.keras.Model(inputs, outputs)

    return model

def freeze_all_vgg(model):
    for layer in model.layers:
        if 'vgg' in layer.name:
            for vgg_layer in layer.layers:
                vgg_layer.trainable = False

def unfreeze_last_vgg(model, which_freeze=15):
    for layer in model.layers:
        if 'vgg' in layer.name:
            for vgg_layer in layer.layers[:which_freeze]:
                vgg_layer.trainable = False
            for vgg_layer in layer.layers[which_freeze:]:
                vgg_layer.trainable = True

def unfreeze_all_vgg(model):
    for layer in model.layers:
        if 'vgg' in layer.name:
            for vgg_layer in layer.layers:
                vgg_layer.trainable = True

def print_vgg_trainable(model):
    for layer in model.layers:
        if 'vgg' in layer.name:
            for i, vgg_layer in enumerate(layer.layers):
                print(i, vgg_layer.name, vgg_layer.trainable)
