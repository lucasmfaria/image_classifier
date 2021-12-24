import pytest
import gc
import tensorflow as tf
import sys
import itertools
from pathlib import Path
try:
    from utils.model import make_model, freeze_all_vgg, unfreeze_last_vgg, unfreeze_all_vgg, print_vgg_trainable
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.model import make_model, freeze_all_vgg, unfreeze_last_vgg, unfreeze_all_vgg, print_vgg_trainable


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


@pytest.fixture
def test_simple_model():
    return make_model(n_classes=2, include_top_vgg=False, n_hidden=30, img_height=40, img_width=40)


def test_freeze_all_vgg(test_simple_model):
    model = test_simple_model
    freeze_all_vgg(model)
    for layer in model.layers:
        if 'vgg' in layer.name:
            for vgg_layer in layer.layers:
                assert vgg_layer.trainable is False


def test_unfreeze_last_vgg(test_simple_model):
    model = test_simple_model
    which_unfreeze = 10
    unfreeze_last_vgg(model, which_unfreeze)
    for layer in model.layers:
        if 'vgg' in layer.name:
            for vgg_layer in layer.layers[:which_unfreeze]:
                assert vgg_layer.trainable is False
            for vgg_layer in layer.layers[which_unfreeze:]:
                assert vgg_layer.trainable is True
    which_unfreeze = 15
    unfreeze_last_vgg(model, which_unfreeze)
    for layer in model.layers:
        if 'vgg' in layer.name:
            for vgg_layer in layer.layers[:which_unfreeze]:
                assert vgg_layer.trainable is False
            for vgg_layer in layer.layers[which_unfreeze:]:
                assert vgg_layer.trainable is True


def test_unfreeze_all_vgg(test_simple_model):
    model = test_simple_model
    unfreeze_all_vgg(model)
    for layer in model.layers:
        if 'vgg' in layer.name:
            for vgg_layer in layer.layers:
                assert vgg_layer.trainable is True
