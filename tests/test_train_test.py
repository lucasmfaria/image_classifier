import subprocess
from pathlib import Path
import sys
try:
    from utils.data import get_platform_shell
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.data import get_platform_shell


def test_delete_last_train_command():
    p = subprocess.run(['python', str(Path(r'./scripts/delete_last_train.py'))], shell=get_platform_shell(), check=True)
    assert p.returncode == 0


# TODO - define simple/dummy dataset to train only for testing
def test_train_command():
    p = subprocess.run(['python', str(Path(r'./scripts/train.py')), '--sample_dataset', 'mnist', '--base_epochs', '3',
                        '--fine_tuning_epochs', '3', '--unit_test_dataset', 'True'], shell=get_platform_shell(),
                       check=True)
    assert p.returncode == 0


# TODO - define simple/dummy dataset to train only for testing
def test_test_command():
    p = subprocess.run(['python', str(Path(r'./scripts/test.py')), '--sample_dataset', 'mnist',
                        '--unit_test_dataset', 'True'], shell=True, check=True)
    assert p.returncode == 0
    p = subprocess.run(['python', str(Path(r'./scripts/delete_last_train.py'))], shell=get_platform_shell(),
                       check=True)  # delete after model testing
