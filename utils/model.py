import tensorflow as tf
from tensorflow.keras.applications import vgg16
from tensorflow.keras import layers


def make_model(n_classes, include_top_vgg=False, n_hidden=512, img_height=224, img_width=224):
    """
    Creates a ConvNet classification model using a VGG16 pre-trained model for transfer learning.
    :param n_classes: int - number of classes required for the classification problem
    :param include_top_vgg: bool - whether or not to include the top of the pre-trained model
    :param n_hidden: int - number of hidden layers to add to the pre-trained model
    :param img_height: int - image height
    :param img_width: int - image width
    :return: tf.keras.Model - final model
    """
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
    n_outputs = n_classes if n_classes != 1 else 1  # only one output neuron if it's a binary classification problem
    outputs = layers.Dense(n_outputs, activation='softmax', name='output')(x)
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
