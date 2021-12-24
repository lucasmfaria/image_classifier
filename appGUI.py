import PySimpleGUI as sg
import subprocess
from pathlib import Path

DEFAULT_DATASET_SOURCE_PATH = Path(__file__).resolve().parent / 'data' / 'dataset'
DEFAULT_SPLITS_DESTINATION = Path(__file__).resolve().parent / 'data'

layout_create_splits = [
    [
        sg.Text("Dataset source folder"),
        sg.In(size=(25,1), enable_events=True, key="-DATASET_SOURCE_FOLDER-", default_text=DEFAULT_DATASET_SOURCE_PATH),
        sg.FolderBrowse(),
    ],
    [
        sg.Text("Splits destination folder"),
        sg.In(size=(25,1), enable_events=True, key="-SPLITS_DESTINATION_FOLDER-",
              default_text=DEFAULT_SPLITS_DESTINATION),
        sg.FolderBrowse(),
    ],
    [
        sg.Text("Test size (float)"),
        sg.In(size=(5,1), enable_events=True, key="-TEST_SIZE-", default_text=0.15),
        sg.Text("Valid size (float)"),
        sg.In(size=(5,1), enable_events=True, key="-VALID_SIZE-", default_text=0.15),
        sg.Button('Create splits', key="-CREATE_SPLITS-"),
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

while True:
    event, values = window.read()
    if event == 'Exit' or event == sg.WIN_CLOSED:
        break
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
        # TODO - verify if the values are possible/wrong inputs -> data Paths exist? test and valid sizes are floats?
        dataset_source_path = Path(values['-DATASET_SOURCE_FOLDER-'])
        splits_destination_path = Path(values['-SPLITS_DESTINATION_FOLDER-'])
        test_size = values['-TEST_SIZE-']
        valid_size = values['-VALID_SIZE-']
        p = subprocess.run(['python', r'./scripts/create_splits.py', '--test_size', test_size,
                            '--valid_size', valid_size, '--dataset_path', dataset_source_path,
                            '--splits_dest_path', splits_destination_path], shell=True)
    elif event == '-TRAIN-':
        subprocess.run(['python', r'./scripts/train.py'], shell=True)
    elif event == '-TEST-':
        subprocess.run(['python', r'./scripts/test.py'], shell=True)

        #subprocess.run(['python', r'./scripts/create_splits.py'], shell=True)


window.close()

