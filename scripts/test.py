import tensorflow as tf
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import sys
import os
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from utils.model import make_model

WEIGHTS_PATH = Path(r'../models/vgg16/checkpoints/trained_weights')

#Dataset parameters:
IMG_HEIGTH = 224
IMG_WIDTH = 224
BATCH_SIZE = 64

N_HIDDEN = 512

test_path = Path(r'../data/test')
test_ds = tf.keras.preprocessing.image_dataset_from_directory(test_path, image_size=(IMG_HEIGTH, IMG_WIDTH), \
                                                              batch_size=BATCH_SIZE, shuffle=False, \
                                                              label_mode='categorical')

AUTOTUNE = tf.data.experimental.AUTOTUNE
class_names = test_ds.class_names
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

model = make_model(n_classes=len(class_names), n_hidden=N_HIDDEN)
model.load_weights(WEIGHTS_PATH)
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
metrics = model.evaluate(test_ds)
print('Loss: {} --------- Accuracy: {}%'.format(metrics[0], np.round(metrics[1]*100, 2)))

y_pred = model.predict(test_ds)
y_pred = np.argmax(y_pred, axis=1)
y_true = tf.concat([y for x, y in test_ds], axis=0)
y_true = np.argmax(y_true.numpy(), axis=1)

print(classification_report(y_true, y_pred, target_names=class_names, digits=2))

pred_labels = [('PRED_' + class_name) for class_name in class_names]
real_labels = [('REAL_' + class_name) for class_name in class_names]
print(pd.DataFrame(confusion_matrix(y_true, y_pred), columns=pred_labels, index=real_labels))
