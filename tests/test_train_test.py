import subprocess
from pathlib import Path


def test_delete_last_train_command():
    p = subprocess.run(['python', str(Path(r'./scripts/delete_last_train.py'))], shell=True, check=True)
    assert p.returncode == 0


# TODO - define simple/dummy dataset to train only for testing
def test_train_command():
    p = subprocess.run(['python', str(Path(r'./scripts/train.py'))], shell=True, check=True)
    assert p.returncode == 0


# TODO - define simple/dummy dataset to train only for testing
def test_test_command():
    p = subprocess.run(['python', str(Path(r'./scripts/test.py'))], shell=True, check=True)
    assert p.returncode == 0
