import tensorflow as tf
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import sys
import argparse
try:
    from utils.model import make_model
    from utils.data import filter_binary_labels, optimize_dataset
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
    from utils.model import make_model
    from utils.data import filter_binary_labels, optimize_dataset

parser = argparse.ArgumentParser()

parser.add_argument('--img_height', type=int, help='Image height after resize', default=224)
parser.add_argument('--img_width', type=int, help='Image width after resize', default=224)
parser.add_argument('--batch_size', type=int, help='Batch size for training', default=64)
parser.add_argument('--n_hidden', type=int, help='Number of neurons in hidden dense layers', default=512)
DEFAULT_TEST_PATH = Path(__file__).parent.parent / 'data' / 'test'
parser.add_argument('--test_path', type=str, help='Path of the test dataset', default=DEFAULT_TEST_PATH)
DEFAULT_WEIGHTS_PATH = Path(__file__).parent.parent / 'models' / 'vgg16' / 'checkpoints' / 'trained_weights'
parser.add_argument('--weights_path', type=str, help='Path of the final model weights file',
                    default=DEFAULT_WEIGHTS_PATH)

args = parser.parse_args()
img_height = args.img_height
img_width = args.img_width
batch_size = args.batch_size
n_hidden = args.n_hidden
test_path = Path(args.test_path)
weights_path = Path(args.weights_path)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(test_path, image_size=(img_height, img_width),
                                                              batch_size=batch_size, shuffle=False,
                                                              label_mode='categorical')

class_names = test_ds.class_names
AUTOTUNE = tf.data.experimental.AUTOTUNE

if len(class_names) == 2:  # take the one-hot-encoded matrix of labels and convert to a vector if binary classification
    test_ds = test_ds.map(filter_binary_labels, num_parallel_calls=AUTOTUNE)
test_ds = optimize_dataset(test_ds)

model = make_model(n_classes=len(class_names), n_hidden=n_hidden)
model.load_weights(weights_path)
loss = tf.keras.losses.CategoricalCrossentropy() if len(class_names) > 2 else tf.keras.losses.BinaryCrossentropy()
model.compile(loss=loss, metrics=['accuracy'])
metrics = model.evaluate(test_ds)
print('Loss: {} --------- Accuracy: {}%'.format(metrics[0], np.round(metrics[1]*100, 2)))

y_pred = model.predict(test_ds)
y_true = tf.concat([y for x, y in test_ds], axis=0)
if len(class_names) == 2:  # uses a threshold for the predictions if binary classification problem
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    y_true = y_true.numpy()
else:  # uses argmax if not binary classification
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true.numpy(), axis=1)

print(classification_report(y_true, y_pred, target_names=class_names, digits=2))

pred_labels = [('PRED_' + class_name) for class_name in class_names]
real_labels = [('REAL_' + class_name) for class_name in class_names]
print(pd.DataFrame(confusion_matrix(y_true, y_pred), columns=pred_labels, index=real_labels))
