from pathlib import Path
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
import sys
try:
    from utils.model import make_model, freeze_all_vgg, unfreeze_last_vgg
    from utils.data import filter_binary_labels, optimize_dataset
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
    from utils.model import make_model, freeze_all_vgg, unfreeze_last_vgg
    from utils.data import filter_binary_labels, optimize_dataset

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 64
SEED = None

#Model parameters:
N_HIDDEN = 512

#Train parameters:
BASE_LEARNING_RATE = 0.001
FINE_TUNING_LEARNING_RATE = 0.001
INITIAL_EPOCHS = 10
FINE_TUNING_EPOCHS = 10
FINE_TUNE_AT_LAYER = 15
LOG_DIR = Path(__file__).parent.parent / 'models' / 'vgg16' / 'logs'
CHECKPOINT_DIR = Path(__file__).parent.parent / 'models' / 'vgg16' / 'checkpoints'
FINAL_MODEL_SAVE_PATH = CHECKPOINT_DIR / 'trained_weights'

train_path = Path(__file__).parent.parent / 'data' / 'train'
train_ds = tf.keras.preprocessing.image_dataset_from_directory(train_path, image_size=(IMG_HEIGHT, IMG_WIDTH),
                                                               batch_size=BATCH_SIZE, shuffle=True,
                                                               label_mode='categorical', seed=SEED)

valid_path = Path(__file__).parent.parent / 'data' / 'valid'
valid_ds = tf.keras.preprocessing.image_dataset_from_directory(valid_path, image_size=(IMG_HEIGHT, IMG_WIDTH),
                                                               batch_size=BATCH_SIZE, shuffle=True,
                                                               label_mode='categorical', seed=SEED)

class_names = train_ds.class_names
assert class_names == valid_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE

if len(class_names) == 2:  # take the one-hot-encoded matrix of labels and convert to a vector if binary classification
    train_ds = train_ds.map(filter_binary_labels, num_parallel_calls=AUTOTUNE)
    valid_ds = valid_ds.map(filter_binary_labels, num_parallel_calls=AUTOTUNE)
train_ds = optimize_dataset(train_ds)
valid_ds = optimize_dataset(valid_ds)

model = make_model(n_classes=len(class_names), n_hidden=N_HIDDEN)
freeze_all_vgg(model)

# TODO - use flake8 for python style test
# TODO - create user interface to train and predict
# TODO - use dynaconf for configurations
# TODO - save class_names taken from the train labels (image_dataset_from_directory)
# TODO - include unit tests
# TODO - set seed usability
# TODO - stratify and undersample for majority class in train, test, valid splits creation
# TODO - create logs
loss = tf.keras.losses.CategoricalCrossentropy() if len(class_names) > 2 else tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=BASE_LEARNING_RATE),
              loss=loss, metrics=['accuracy'])

tb = TensorBoard(log_dir=LOG_DIR)
checkpoint = ModelCheckpoint(CHECKPOINT_DIR / 'train_{epoch}.tf', verbose=1, save_weights_only=True,
                             save_best_only=True, monitor='val_loss')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1)

history = model.fit(train_ds, epochs=INITIAL_EPOCHS, validation_data=valid_ds, callbacks=[tb, checkpoint, reduce_lr,
                                                                                          early_stopping])
unfreeze_last_vgg(model, which_freeze=FINE_TUNE_AT_LAYER)

total_epochs = INITIAL_EPOCHS + FINE_TUNING_EPOCHS
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNING_LEARNING_RATE),
              loss=loss, metrics=['accuracy'])
history = model.fit(train_ds, epochs=total_epochs, validation_data=valid_ds,
                    callbacks=[tb, checkpoint, reduce_lr, early_stopping], initial_epoch=history.epoch[-1])

model.save_weights(FINAL_MODEL_SAVE_PATH)
