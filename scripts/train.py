from pathlib import Path
import sys
import argparse
try:
    from utils.model import make_model, freeze_all_vgg, unfreeze_last_vgg, loss_definition, initial_model, \
        callbacks_definition, train
    from utils.data import filter_binary_labels, optimize_dataset, prepare_sample_dataset, true_or_false, \
        dataset_definition
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.model import make_model, freeze_all_vgg, unfreeze_last_vgg, loss_definition, initial_model, \
        callbacks_definition, train
    from utils.data import filter_binary_labels, optimize_dataset, prepare_sample_dataset, true_or_false, \
        dataset_definition

parser = argparse.ArgumentParser()

# TODO - insert default values from dynaconf
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
DEFAULT_LOG_PATH = Path(__file__).resolve().parent.parent / 'models' / 'vgg16' / 'logs'
parser.add_argument('--log_path', type=str, help='Path of the training logs', default=DEFAULT_LOG_PATH)
DEFAULT_CHECKPOINTS_PATH = Path(__file__).resolve().parent.parent / 'models' / 'vgg16' / 'checkpoints'
parser.add_argument('--checkpoints_path', type=str, help='Path of the model checkpoints',
                    default=DEFAULT_CHECKPOINTS_PATH)
parser.add_argument('--final_model_name', type=str, help='Name of the final model file', default='trained_weights')
DEFAULT_TRAIN_PATH = Path(__file__).resolve().parent.parent / 'data' / 'train'
parser.add_argument('--train_path', type=str, help='Path of the train dataset', default=DEFAULT_TRAIN_PATH)
DEFAULT_VALID_PATH = Path(__file__).resolve().parent.parent / 'data' / 'valid'
parser.add_argument('--valid_path', type=str, help='Path of the validation dataset', default=DEFAULT_VALID_PATH)
parser.add_argument('--sample_dataset', type=str, help='Name of sample dataset in [mnist]',
                    default=None)
parser.add_argument('--unit_test_dataset', type=true_or_false, help='Whether or not to load only a few images, only for unit testing',
                    default=False)
args = parser.parse_args()

# TODO - use flake8 for python style test
# TODO - create functions docstring
# TODO - create filter for corrupted images before train
# TODO - use dynaconf for configurations
# TODO - save class_names taken from the train labels (image_dataset_from_directory)
# TODO - create logs
# TODO - debug seed for real reproducibility
# TODO - script to verify if there are duplicated images/files
# TODO - create .bat file for windows users -> activate venv (or root python), download libs, run UI or scripts
# TODO - new metrics


def main(train_path=DEFAULT_TRAIN_PATH, valid_path=DEFAULT_VALID_PATH, sample_dataset=None, batch_size=64,
         img_height=224, img_width=224, seed=None, unit_test_dataset=False, n_hidden=512, base_lr=0.001,
         log_path=DEFAULT_LOG_PATH, checkpoints_path=DEFAULT_CHECKPOINTS_PATH, base_epochs=30, fine_tuning_epochs=30,
         fine_tune_at_layer=15, fine_tuning_lr=0.001, final_model_name='trained_weights'):

    # load the dataset:
    train_ds, valid_ds, class_names = dataset_definition(train_path=Path(train_path), valid_path=Path(valid_path),
                                                         sample_dataset=sample_dataset, batch_size=batch_size,
                                                         img_height=img_height, img_width=img_width, seed=seed,
                                                         unit_test_dataset=unit_test_dataset)
    # build the initial model with frozen VGG16 layers:
    model = initial_model(n_classes=len(class_names), n_hidden=n_hidden, img_height=img_height, img_width=img_width,
                          seed=seed, base_lr=base_lr)
    # create the callback functions:
    callbacks = callbacks_definition(log_path=Path(log_path), checkpoints_path=Path(checkpoints_path))
    # train the model:
    model, history = train(model=model, train_ds=train_ds, valid_ds=valid_ds, n_classes=len(class_names),
                           base_epochs=base_epochs, fine_tuning_epochs=fine_tuning_epochs,
                           fine_tune_at_layer=fine_tune_at_layer, fine_tuning_lr=fine_tuning_lr,
                           callbacks=callbacks, seed=seed)
    # save the model
    model.save_weights(Path(checkpoints_path) / final_model_name)


if __name__ == '__main__':
    main(train_path=Path(args.train_path), valid_path=Path(args.valid_path), sample_dataset=args.sample_dataset,
         batch_size=args.batch_size, img_height=args.img_height, img_width=args.img_width, seed=args.seed,
         unit_test_dataset=args.unit_test_dataset, n_hidden=args.n_hidden, base_lr=args.base_lr,
         log_path=Path(args.log_path), checkpoints_path=Path(args.checkpoints_path), base_epochs=args.base_epochs,
         fine_tuning_epochs=args.fine_tuning_epochs, fine_tune_at_layer=args.fine_tune_at_layer,
         fine_tuning_lr=args.fine_tuning_lr, final_model_name=args.final_model_name)
