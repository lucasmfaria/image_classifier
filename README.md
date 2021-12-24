# Image Classifier

> Convolutional Neural Network for image classification (binary or multiclass). Uses transfer learning with pre-trained
> VGG16 model. Tested with python 3.9.

## 💻 Prerequisites

```
pip install -r requirements.txt
```

## ☕ Using the code

```
- Move your image dataset into the folder 'data', with the classes separation. Final directory 
example for a multiclass classification: 'data/dataset/class1', 'data/dataset/class2', 'data/dataset/class3'.
- Use "create_splits.py" to create the train, test and validation splits.
- Use "train.py" to train your neural network
- Use "test.py" to evaluate your model
- Use "delete_last_train.py" to delete the trained model.
```

### TODOs
- [X] Jupyter notebook with the Transfer Learning experiment
- [X] Script to generate the data splits (train, test and validation).
- [X] Under sampling for the majority classes, controlled by the user.
- [X] Train and test scripts
- [X] Create user interface to train, test, predict and serve model (need to improve)
- [X] Create unit tests (need to improve)
- [ ] Create API for model serving/deploy
- [ ] User dynaconf library to centralize application configurations
- [ ] Save "class_names" after training as model configuration