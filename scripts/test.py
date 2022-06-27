import tensorflow as tf
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix, PrecisionRecallDisplay, average_precision_score, \
    RocCurveDisplay, roc_curve, auc, precision_recall_curve
import sys
import argparse
try:
    from utils.model import make_model, loss_definition
    from utils.data import filter_binary_labels, optimize_dataset, prepare_sample_dataset, true_or_false, \
        dataset_definition
    from utils.charts import build_roc, build_precision_recall
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.model import make_model, loss_definition
    from utils.data import filter_binary_labels, optimize_dataset, prepare_sample_dataset, true_or_false, \
        dataset_definition
    from utils.charts import build_roc, build_precision_recall

parser = argparse.ArgumentParser()

parser.add_argument('--img_height', type=int, help='Image height after resize', default=224)
parser.add_argument('--img_width', type=int, help='Image width after resize', default=224)
parser.add_argument('--batch_size', type=int, help='Batch size for testing', default=64)
parser.add_argument('--n_hidden', type=int, help='Number of neurons in hidden dense layers', default=512)
DEFAULT_TEST_PATH = Path(__file__).resolve().parent.parent / 'data' / 'test'
parser.add_argument('--test_path', type=str, help='Path of the test dataset', default=DEFAULT_TEST_PATH)
DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / 'models' / 'vgg16' / 'checkpoints' / 'trained_weights'
parser.add_argument('--weights_path', type=str, help='Path of the final model weights file',
                    default=DEFAULT_WEIGHTS_PATH)
parser.add_argument('--sample_dataset', type=str, help='Name of sample dataset in [mnist, patch_camelyon]',
                    default=None)
parser.add_argument('--unit_test_dataset', type=true_or_false, help='Whether or not to load only a few images, only for unit testing',
                    default=False)
args = parser.parse_args()


def main(test_path=DEFAULT_TEST_PATH, sample_dataset=None, batch_size=64, img_height=224, img_width=224,
         unit_test_dataset=False, n_hidden=512, weights_path=DEFAULT_WEIGHTS_PATH, return_results=False):
    _, _, test_ds, class_names = dataset_definition(test_path=Path(test_path), sample_dataset=sample_dataset,
                                                   batch_size=batch_size, img_height=img_height, img_width=img_width,
                                                   unit_test_dataset=unit_test_dataset)
    model = tf.keras.models.load_model(Path(weights_path))

    y_score = model.predict(test_ds)
    y_true = tf.concat([y for x, y in test_ds], axis=0)
    y_true = y_true.numpy()
    if len(class_names) == 2:  # uses a threshold for the predictions if binary classification problem
        y_pred = (y_score >= 0.5).astype(np.uint8)
    else:  # uses argmax if not binary classification
        y_pred = np.argmax(y_score, axis=1)
        y_true = np.argmax(y_true, axis=1)

    print(classification_report(y_true, y_pred, target_names=class_names, digits=2))  # always print on console
    pred_labels = [('PRED_' + class_name) for class_name in class_names]
    real_labels = [('REAL_' + class_name) for class_name in class_names]
    df_confusion_matrix = pd.DataFrame(confusion_matrix(y_true, y_pred), columns=pred_labels, index=real_labels)
    print(df_confusion_matrix)

    precision_recall = None
    roc = None
    if len(class_names) == 2:
        auc_results = dict()
        # Calculates precision, recall, thresholds and auc for positive class (class_names[1])
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        thresholds = np.append(thresholds, 1)  # append the threshold "1", due to sklearn behavior
        auc_results[class_names[1]] = average_precision_score(y_true, y_score)
        classes = [class_names[1]] * recall.shape[0]

        # Calculates precision, recall, thresholds and auc for negative class (class_names[0])
        precision_, recall_, thresholds_ = precision_recall_curve(1 - y_true, 1 - y_score)
        thresholds_ = np.append(thresholds_, 1)  # append the threshold "1", due to sklearn behavior
        auc_results[class_names[0]] = average_precision_score(1 - y_true, 1 - y_score)
        classes = classes + [class_names[0]] * recall_.shape[0]

        # generates full dataframe with both classes
        thresholds = np.append(thresholds, thresholds_)
        precision = np.append(precision, precision_)
        recall = np.append(recall, recall_)
        df = pd.DataFrame({'recall': recall, 'precision': precision, 'class': classes, 'thresholds': thresholds})
        precision_recall = build_precision_recall(df, auc_results)

        # Calculates roc metrics and auc for positive class (class_names[1])
        auc_results = dict()
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        auc_results[class_names[1]] = auc(fpr, tpr)
        classes = [class_names[1]] * fpr.shape[0]

        # Calculates roc metrics and auc for negative class (class_names[0])
        fpr_, tpr_, thresholds_ = roc_curve(1 - y_true, 1 - y_score)
        auc_results[class_names[0]] = auc(fpr_, tpr_)
        classes = classes + [class_names[0]] * fpr_.shape[0]

        # generates full dataframe with both classes
        thresholds = np.append(thresholds, thresholds_)
        fpr = np.append(fpr, fpr_)
        tpr = np.append(tpr, tpr_)
        df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'class': classes, 'thresholds': thresholds})
        roc = build_roc(df, auc_results)

    if return_results:
        classification_report_dict = classification_report(y_true, y_pred, target_names=class_names, digits=2,
                                                           output_dict=return_results)
        return classification_report_dict, df_confusion_matrix, precision_recall, roc
    elif (roc is not None) and (precision_recall is not None):
        roc.show()
        precision_recall.show()


if __name__ == '__main__':
    main(test_path=Path(args.test_path), sample_dataset=args.sample_dataset, batch_size=args.batch_size,
         img_height=args.img_height, img_width=args.img_width, unit_test_dataset=args.unit_test_dataset,
         n_hidden=args.n_hidden, weights_path=Path(args.weights_path))
