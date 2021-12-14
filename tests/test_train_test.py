import subprocess
from pathlib import Path


# TODO - define simple/dummy dataset to train only for testing
def test_train_command():
    p = subprocess.run(['python', str(Path(r'./scripts/train.py'))], shell=True, check=True)
    assert p.returncode == 0
