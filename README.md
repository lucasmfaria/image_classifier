# Image Classifier

> Convolutional Neural Network for image classification (binary or multiclass). Uses transfer learning with pre-trained
> VGG16 model. Tested with python 3.9.

## 💻 Prerequisites

```
pip install -r requirements.txt
```

## ☕ Using the code

```
- Use "create_splits.py" to create the train, test and validation splits.
- Use "train.py" to train your neural network
- Use "test.py" to evaluate your model
```

### TODOs
- [X] Jupyter notebook with the Transfer Learning experiment
- [X] Script to generate the data splits (train, test and validation)
- [X] Train and test scripts
- [ ] Create API for model serving/deploy
- [ ] Create user interface to train, test, predict and serve model
- [ ] Create unit tests
- [ ] User dynaconf library to centralize application configurations
- [ ] Save "class_names" after training as model configuration