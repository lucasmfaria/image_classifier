import pytest
import gc
import tensorflow as tf
import sys
import itertools
from pathlib import Path
try:
    from utils.model import make_model, freeze_all_vgg, unfreeze_last_vgg
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
    from utils.model import make_model, freeze_all_vgg, unfreeze_last_vgg


def test_make_model():
    params_dict = {
        'n_classes': [2, 3, 4],
        'include_top_vgg': [True, False],
        'n_hidden': [512, 256],
        'img_height': [224, 112, 200],
        'img_width': [224, 112, 200],
    }
    params_product_list = list(itertools.product(params_dict['n_classes'], params_dict['include_top_vgg'],
                                                 params_dict['n_hidden'], params_dict['img_height'],
                                                 params_dict['img_width']))
    for params in params_product_list:
        model = make_model(n_classes=params[0], include_top_vgg=params[1], n_hidden=params[2], img_height=params[3],
                           img_width=params[4])

        # checking inputs shape:
        assert model.inputs[0].shape.as_list() == tf.TensorShape((None, params[3], params[4], 3)).as_list()

        # checking outputs shape:
        assert model.outputs[0].shape.as_list() == tf.TensorShape((None, params[0] if params[0] > 2 else 1)).as_list()
        del model
        gc.collect()
