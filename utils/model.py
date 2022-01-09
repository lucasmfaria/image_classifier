from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import vgg16
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

DEFAULT_CHECKPOINTS_PATH = Path(__file__).resolve().parent.parent / 'models' / 'vgg16' / 'checkpoints'
DEFAULT_LOG_PATH = Path(__file__).resolve().parent.parent / 'models' / 'vgg16' / 'logs'

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
    n_outputs = n_classes if n_classes != 2 else 1  # only one output neuron if it's a binary classification problem
    activation = 'softmax' if n_classes != 2 else 'sigmoid'  # sigmoid if it's a binary classification problem
    outputs = layers.Dense(n_outputs, activation=activation, name='output')(x)
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


def loss_definition(n_classes):
    return tf.keras.losses.CategoricalCrossentropy() if n_classes > 2 else tf.keras.losses.BinaryCrossentropy()


def initial_model(n_classes, n_hidden=512, img_height=224, img_width=224, seed=None, base_lr=0.001):
    if seed is not None:
        tf.random.set_seed(seed)

    model = make_model(n_classes=n_classes, n_hidden=n_hidden, img_height=img_height, img_width=img_width)
    freeze_all_vgg(model)

    loss = loss_definition(n_classes=n_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_lr), loss=loss, metrics=['accuracy'])
    return model


def callbacks_definition(log_path=DEFAULT_LOG_PATH, checkpoints_path=DEFAULT_CHECKPOINTS_PATH):
    tb = TensorBoard(log_dir=log_path)
    checkpoint = ModelCheckpoint(checkpoints_path / 'train_{epoch}.tf', verbose=1, save_weights_only=True,
                                 save_best_only=True, monitor='val_loss')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1)
    callbacks = [tb, checkpoint, reduce_lr, early_stopping]
    return callbacks


def train(model, train_ds, valid_ds, n_classes, base_epochs=30, fine_tuning_epochs=30, fine_tune_at_layer=15,
          fine_tuning_lr=0.001, callbacks=None, seed=None):
    if seed is not None:
        tf.random.set_seed(seed)

    history = model.fit(train_ds, epochs=base_epochs, validation_data=valid_ds, callbacks=callbacks)
    unfreeze_last_vgg(model, which_freeze=fine_tune_at_layer)

    total_epochs = base_epochs + fine_tuning_epochs
    loss = loss_definition(n_classes=n_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tuning_lr), loss=loss, metrics=['accuracy'])

    if seed is not None:
        tf.random.set_seed(seed)

    history = model.fit(train_ds, epochs=total_epochs, validation_data=valid_ds, callbacks=callbacks,
                        initial_epoch=history.epoch[-1])
    return model, history
