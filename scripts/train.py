from pathlib import Path
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
import sys
import argparse
try:
    from utils.model import make_model, freeze_all_vgg, unfreeze_last_vgg
    from utils.data import filter_binary_labels, optimize_dataset, prepare_sample_dataset
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.model import make_model, freeze_all_vgg, unfreeze_last_vgg
    from utils.data import filter_binary_labels, optimize_dataset, prepare_sample_dataset

parser = argparse.ArgumentParser()

parser.add_argument('--img_height', type=int, help='Image height after resize', default=224)
parser.add_argument('--img_width', type=int, help='Image width after resize', default=224)
parser.add_argument('--batch_size', type=int, help='Batch size for training', default=64)
parser.add_argument('--seed', type=int, help='Seed number for reproducibility', default=None)
parser.add_argument('--n_hidden', type=int, help='Number of neurons in hidden dense layers', default=512)
parser.add_argument('--base_lr', type=float,
                    help='Learning rate of initial training (frozen VGG)', default=0.001)
parser.add_argument('--fine_tuning_lr', type=float,
                    help='Learning rate of fine tuning training (unfrozen VGG)', default=0.001)
parser.add_argument('--base_epochs', type=int,
                    help='Number of epochs during the initial training (frozen VGG)', default=30)
parser.add_argument('--fine_tuning_epochs', type=int,
                    help='Number of epochs during fine tuning training (unfrozen VGG)', default=30)
parser.add_argument('--fine_tune_at_layer', type=int, help='Index of VGG layer to unfreeze', default=15)
DEFAULT_LOG_PATH = Path(__file__).parent.parent / 'models' / 'vgg16' / 'logs'
parser.add_argument('--log_path', type=str, help='Path of the training logs', default=DEFAULT_LOG_PATH)
DEFAULT_CHECKPOINTS_PATH = Path(__file__).parent.parent / 'models' / 'vgg16' / 'checkpoints'
parser.add_argument('--checkpoints_path', type=str, help='Path of the model checkpoints',
                    default=DEFAULT_CHECKPOINTS_PATH)
parser.add_argument('--final_model_name', type=str, help='Name of the final model file', default='trained_weights')
DEFAULT_TRAIN_PATH = Path(__file__).parent.parent / 'data' / 'train'
parser.add_argument('--train_path', type=str, help='Path of the train dataset', default=DEFAULT_TRAIN_PATH)
DEFAULT_VALID_PATH = Path(__file__).parent.parent / 'data' / 'valid'
parser.add_argument('--valid_path', type=str, help='Path of the validation dataset', default=DEFAULT_VALID_PATH)
parser.add_argument('--sample_dataset', type=str, help='Name of sample dataset in [mnist]',
                    default=None)
parser.add_argument('--unit_test_dataset', type=bool, help='Whether or not to load only a few images, only for unit testing',
                    default=False)

args = parser.parse_args()
img_height = args.img_height
img_width = args.img_width
batch_size = args.batch_size
seed = args.seed
n_hidden = args.n_hidden
base_lr = args.base_lr
fine_tuning_lr = args.fine_tuning_lr
base_epochs = args.base_epochs
fine_tuning_epochs = args.fine_tuning_epochs
fine_tune_at_layer = args.fine_tune_at_layer
log_path = Path(args.log_path)
checkpoints_path = Path(args.checkpoints_path)
final_model_name = args.final_model_name
final_model_save_path = checkpoints_path / final_model_name
train_path = Path(args.train_path)
valid_path = Path(args.valid_path)
sample_dataset = args.sample_dataset
unit_test_dataset = args.unit_test_dataset

if sample_dataset in ['mnist']:  # loads a sample dataset for user/unit testing
    train_ds, valid_ds, class_names = prepare_sample_dataset(sample_dataset=sample_dataset, batch_size=batch_size,
                                                             img_height=img_height, img_width=img_width)

else:  # loads a user defined dataset in path
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(train_path, image_size=(img_height, img_width),
                                                                   batch_size=batch_size, shuffle=True,
                                                                   label_mode='categorical', seed=seed)

    valid_ds = tf.keras.preprocessing.image_dataset_from_directory(valid_path, image_size=(img_height, img_width),
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

if seed is not None:
    tf.random.set_seed(seed)

model = make_model(n_classes=len(class_names), n_hidden=n_hidden, img_height=img_height, img_width=img_width)
freeze_all_vgg(model)

# TODO - use flake8 for python style test
# TODO - create functions docstring
# TODO - create filter for corrupted images before train
# TODO - use dynaconf for configurations
# TODO - save class_names taken from the train labels (image_dataset_from_directory)
# TODO - create logs
# TODO - debug seed for real reproducibility
# TODO - script to verify if there are duplicated images/files
# TODO - create script to save the models
loss = tf.keras.losses.CategoricalCrossentropy() if len(class_names) > 2 else tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_lr),
              loss=loss, metrics=['accuracy'])

tb = TensorBoard(log_dir=log_path)
checkpoint = ModelCheckpoint(checkpoints_path / 'train_{epoch}.tf', verbose=1, save_weights_only=True,
                             save_best_only=True, monitor='val_loss')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1)

if seed is not None:
    tf.random.set_seed(seed)

history = model.fit(train_ds, epochs=base_epochs, validation_data=valid_ds, callbacks=[tb, checkpoint, reduce_lr,
                                                                                       early_stopping])
unfreeze_last_vgg(model, which_freeze=fine_tune_at_layer)

total_epochs = base_epochs + fine_tuning_epochs
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tuning_lr),
              loss=loss, metrics=['accuracy'])

if seed is not None:
    tf.random.set_seed(seed)

history = model.fit(train_ds, epochs=total_epochs, validation_data=valid_ds,
                    callbacks=[tb, checkpoint, reduce_lr, early_stopping], initial_epoch=history.epoch[-1])

model.save_weights(final_model_save_path)
