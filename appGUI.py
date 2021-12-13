import PySimpleGUI as sg
import subprocess
import sys
import os
from pathlib import Path
import shutil
from utils.data import train_test_valid_split


layout_create_splits = [
    [
        sg.Text("Dataset source folder"),
        sg.In(size=(25,1), enable_events=True, key="-DATASET_SOURCE_FOLDER-"),
        sg.FolderBrowse(),
    ],
    [
        sg.Text("Splits destination folder"),
        sg.In(size=(25,1), enable_events=True, key="-SPLITS_DESTINATION_FOLDER-"),
        sg.FolderBrowse(),
    ],
    [
        sg.Button('Create train, test and validation splits', key="-CREATE_SPLITS-"),
    ],
    [
        sg.ProgressBar(1000, orientation='h', size=(20, 20), key='-TRAIN_PROGRESSBAR-'),
    ],
    [
        sg.ProgressBar(1000, orientation='h', size=(20, 20), key='-TEST_PROGRESSBAR-'),
    ],
    [
        sg.ProgressBar(1000, orientation='h', size=(20, 20), key='-VALID_PROGRESSBAR-'),
    ],
    [
        sg.Text("Change screen"),
        sg.Button('<<', key="-CHANGE_LEFT_SPLITS-"),
        sg.Button('>>', key="-CHANGE_RIGHT_SPLITS-"),
    ],
]

layout_train = [
    [
        sg.Button('TRAIN', key="-TRAIN-"),
        sg.Text("Change screen"),
        sg.Button('<<', key="-CHANGE_LEFT_TRAIN-"),
        sg.Button('>>', key="-CHANGE_RIGHT_TRAIN-"),
    ]
]

layout_test = [
    [
        sg.Button('TEST', key="-TEST-"),
        sg.Text("Change screen"),
        sg.Button('<<', key="-CHANGE_LEFT_TEST-"),
        sg.Button('>>', key="-CHANGE_RIGHT_TEST-"),
    ]
]


layout = [
    [
        sg.Column(layout_create_splits, visible=True, key='-COL_SPLITS-'),
        sg.Column(layout_train, visible=False, key='-COL_TRAIN-'),
        sg.Column(layout_test, visible=False, key='-COL_TEST-'),
    ]
]

window = sg.Window("APP", layout)
dataset_source_folder = ''
splits_destination_folder = ''
while True:
    event, values = window.read()
    if event == 'Exit' or event == sg.WIN_CLOSED:
        break

    if event == '-DATASET_SOURCE_FOLDER-':
        dataset_source_folder = values['-DATASET_SOURCE_FOLDER-']
    elif event == '-SPLITS_DESTINATION_FOLDER-':
        splits_destination_folder = values['-SPLITS_DESTINATION_FOLDER-']
    elif event == '-CHANGE_RIGHT_SPLITS-':
        window['-COL_SPLITS-'].update(visible=False)
        window['-COL_TRAIN-'].update(visible=True)
    elif event == '-CHANGE_LEFT_SPLITS-':
        window['-COL_SPLITS-'].update(visible=False)
        window['-COL_TEST-'].update(visible=True)
    elif event == '-CHANGE_LEFT_TRAIN-':
        window['-COL_SPLITS-'].update(visible=True)
        window['-COL_TRAIN-'].update(visible=False)
    elif event == '-CHANGE_RIGHT_TRAIN-':
        window['-COL_TEST-'].update(visible=True)
        window['-COL_TRAIN-'].update(visible=False)
    elif event == '-CHANGE_LEFT_TEST-':
        window['-COL_TEST-'].update(visible=False)
        window['-COL_TRAIN-'].update(visible=True)
    elif event == '-CHANGE_RIGHT_TEST-':
        window['-COL_SPLITS-'].update(visible=True)
        window['-COL_TEST-'].update(visible=False)
    elif event == '-CREATE_SPLITS-':
        if (dataset_source_folder != '') and (splits_destination_folder != ''):
            X_train, X_test, X_valid = train_test_valid_split(dataset_source_folder, test_size=0.15, valid_size=0.15)
            window['-TRAIN_PROGRESSBAR-'].MaxValue = X_train.shape[0]
            window['-TEST_PROGRESSBAR-'].MaxValue = X_test.shape[0]
            window['-VALID_PROGRESSBAR-'].MaxValue = X_valid.shape[0]

            window['-TRAIN_PROGRESSBAR-'].Size = (20, 20)
            window['-TEST_PROGRESSBAR-'].Size = (20, 20*(X_test.shape[0]/X_train.shape[0]))
            window['-VALID_PROGRESSBAR-'].Size = (20, 20*(X_valid.shape[0]/X_train.shape[0]))

            split = 'train'
            destination_path = Path(splits_destination_folder) / split
            if Path(destination_path).exists():
                print('--------------DELETE ' + split.upper() + ' SPLIT------------')
                for directory in destination_path.iterdir():
                    if directory.is_dir():
                        shutil.rmtree(directory)
            print('--------------COPY ' + split.upper() + ' SPLIT------------')
            for bar_idx, iterrows_tuple in enumerate(X_train.iterrows()):
                destination = (destination_path / iterrows_tuple[0].parent.name) / iterrows_tuple[0].name
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                shutil.copy(iterrows_tuple[0], destination)
                window['-TRAIN_PROGRESSBAR-'].update(bar_idx)

            split = 'test'
            destination_path = Path(splits_destination_folder) / split
            if Path(destination_path).exists():
                print('--------------DELETE ' + split.upper() + ' SPLIT------------')
                for directory in destination_path.iterdir():
                    if directory.is_dir():
                        shutil.rmtree(directory)
            print('--------------COPY ' + split.upper() + ' SPLIT------------')
            for bar_idx, iterrows_tuple in enumerate(X_test.iterrows()):
                destination = (destination_path / iterrows_tuple[0].parent.name) / iterrows_tuple[0].name
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                shutil.copy(iterrows_tuple[0], destination)
                window['-TEST_PROGRESSBAR-'].update(bar_idx)

            split = 'valid'
            destination_path = Path(splits_destination_folder) / split
            if Path(destination_path).exists():
                print('--------------DELETE ' + split.upper() + ' SPLIT------------')
                for directory in destination_path.iterdir():
                    if directory.is_dir():
                        shutil.rmtree(directory)
            print('--------------COPY ' + split.upper() + ' SPLIT------------')
            for bar_idx, iterrows_tuple in enumerate(X_valid.iterrows()):
                destination = (destination_path / iterrows_tuple[0].parent.name) / iterrows_tuple[0].name
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                shutil.copy(iterrows_tuple[0], destination)
                window['-VALID_PROGRESSBAR-'].update(bar_idx)
    elif event == '-TRAIN-':
        subprocess.run(['python', r'./scripts/train.py'], shell=True)
    elif event == '-TEST-':
        subprocess.run(['python', r'./scripts/test.py'], shell=True)

        #subprocess.run(['python', r'./scripts/create_splits.py'], shell=True)


window.close()

