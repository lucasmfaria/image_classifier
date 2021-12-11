from pathlib import Path
import shutil

CHECKPOINTS_FOLDER = r'../models/vgg16/checkpoints'
LIST_CHECKPOINTS_SUBSTRINGS_DELETE = ['.tf.', 'checkpoint', 'trained_weights']
LOGS_FOLDER = r'../models/vgg16/logs'
LIST_LOGS_FOLDERS_DELETE = ['train', 'validation']

# Delete the tensorflow model checkpoints
list_files_delete = [file for file in Path(CHECKPOINTS_FOLDER).iterdir() if any(x in file.name for x in \
                                                                                LIST_CHECKPOINTS_SUBSTRINGS_DELETE)]
for file in list_files_delete:
    file.unlink()

# Delete the tensorflow training logs
list_directories_delete = [directory for directory in Path(LOGS_FOLDER).iterdir() if any(x in directory.name for x in \
                                                                                         LIST_LOGS_FOLDERS_DELETE)]
list_directories_delete = [directory for directory in list_directories_delete if directory.is_dir()]
for directory in list_directories_delete:
    shutil.rmtree(directory)
