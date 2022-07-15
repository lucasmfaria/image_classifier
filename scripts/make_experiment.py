import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
from tensorflow.keras.backend import clear_session
from tensorflow.python.framework.ops import disable_eager_execution
import subprocess
from multiprocessing import Process
import time
import gc
import json
from pathlib import Path
import sys
try:
    from scripts.train import main as train_main
    from scripts.test import main as test_main
    from utils.data import get_platform_shell
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.train import main as train_main
    from scripts.test import main as test_main
    from utils.data import get_platform_shell


''' Metrics:
Time during training
Time during testing
Accuracy
ROC AUC
P-R AUC
'''

N_SAME_EXPERIMENT = 10
#SAMPLE_DATASET = 'patch_camelyon_resnet152v2'  # IMG_WIDTH = 7  IMG_HEIGHT = 7 IMG_DEPTH = 2048
#SAMPLE_DATASET = 'patch_camelyon_vgg16'  # IMG_WIDTH = 7  IMG_HEIGHT = 7 IMG_DEPTH = 512
SAMPLE_DATASET = 'patch_camelyon_inceptionv3'
BATCH_SIZE = 1024
IMG_HEIGHT = 5
IMG_WIDTH = 5
IMG_DEPTH = 2048
N_HIDDEN = 512
BASE_EPOCHS = 35
FINE_TUNING_EPOCHS = 0
TRANSFER_LEARNING = False
BASE_MODEL = 'classifier'


def make_experiment():
    parameters = {
        'SAMPLE_DATASET': SAMPLE_DATASET,
        'BATCH_SIZE': BATCH_SIZE,
        'IMG_HEIGHT': IMG_HEIGHT,
        'IMG_WIDTH': IMG_WIDTH,
        'IMG_DEPTH': IMG_DEPTH,
        'N_HIDDEN': N_HIDDEN,
        'BASE_EPOCHS': BASE_EPOCHS,
        'FINE_TUNING_EPOCHS': FINE_TUNING_EPOCHS,
        'TRANSFER_LEARNING': TRANSFER_LEARNING,
        'BASE_MODEL': BASE_MODEL
    }
    
    metrics = {
            'fit_time': [],
            'test_time': [],
            'accuracy': [],
            'roc_auc': [],
            'pr_auc': [],
        }
    p = subprocess.run(['python', str(Path(__file__).resolve().parent / 'delete_last_train.py')], shell=get_platform_shell(), check=True)
    
    clear_session()
    #disable_eager_execution()
    
    model_metrics = set(metrics.keys()) - set(['fit_time', 'test_time'])
    start_time = time.time()
    train_main(sample_dataset=SAMPLE_DATASET, batch_size=BATCH_SIZE, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, n_hidden=N_HIDDEN,
               base_epochs=BASE_EPOCHS, fine_tuning_epochs=FINE_TUNING_EPOCHS, transfer_learning=TRANSFER_LEARNING, base_model=BASE_MODEL, metrics=model_metrics)
    metrics['fit_time'].append(time.time() - start_time)
    gc.collect()
    
    start_time = time.time()
    (classification_report_dict, auc_values), df_confusion_matrix, precision_recall, roc = test_main(sample_dataset=SAMPLE_DATASET, return_results=True, batch_size=BATCH_SIZE, img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
    metrics['test_time'].append(time.time() - start_time)
    gc.collect()
    
    if auc_values is not None:
        metrics['pr_auc'].append(auc_values['pr_auc'])  # precision recall AUC:
        metrics['roc_auc'].append(auc_values['roc_auc'])  # roc curve AUC:
    
    metrics['accuracy'].append(classification_report_dict['accuracy'])
    
    experiment_result = {
            'metrics': metrics,
            'parameters': parameters
        }
    
    experiment_output_path = Path(__file__).resolve().parent.parent / 'data' / 'experiments'
    try:
        max_experiment_file_num = max([int(file.name.split('_')[-1].split('.')[0]) for file in list(experiment_output_path.iterdir()) if file.name != '.gitkeep'])
    except ValueError:
        max_experiment_file_num = 0
    
    with open(experiment_output_path / ("experiment_" + str(max_experiment_file_num + 1) + ".json"), "w") as outfile:
        json.dump(experiment_result, outfile)


def main():
    for i in range(N_SAME_EXPERIMENT):
        p = Process(target=make_experiment)  # using Process class to make sure the memory is clean after each experiment. This prevents training and testing from getting slow after each experiment.
        p.start()
        flag = p.join()
        print('Subprocess exited with code', str(flag))


if __name__ == '__main__':
    main()
