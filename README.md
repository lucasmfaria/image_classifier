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
- Use "train.py" to train your neural network.
- Use "test.py" to evaluate your model.
- Use "save_last_train.py" if you liked your tested model and want to save it on "saved_models" directory.
- Use "delete_last_train.py" to delete the trained model from the "checkpoints" and "logs" directories.
```

## 🚀 Using the Web Application
```
- Move your image dataset into the folder 'data', with the classes separation. Final directory 
example for a multiclass classification: 'data/dataset/class1', 'data/dataset/class2', 'data/dataset/class3'.
- Run "streamlit run web_app.py".
- Open the browser - http://localhost:8501/
- Create the train, validation and test splits.
- Train the neural network.
- Test the neural network.
```

## 🔨 Windows users:
You have the option to just run the "web_app.bat" file. It will create the virtual env and download the requirements for you.
You need to have python installed (version 3.9) and pointed by "PATH" environment variable in order to follow the steps:

```
- Move your image dataset into the folder 'data', with the classes separation. Final directory 
example for a multiclass classification: 'data/dataset/class1', 'data/dataset/class2', 'data/dataset/class3'.
- Run "web_app.bat" to start the application.
- Open the browser - http://localhost:8501/
- Create the train, validation and test splits.
- Train the neural network.
- Test the neural network.
```

## 📫 Contribute to the project
Follow these steps if you want to contribute:

1. Fork this repo.
2. Create a branch: `git checkout -b <branch_name>`.
3. Change the code and commit: `git commit -m '<commit_message>'`
4. Send to original branch: `git push origin image_classifier / <local>`
5. Create a pull request.

### TODOs
- [X] Jupyter notebook with the Transfer Learning experiment
- [X] Script to generate the data splits (train, test and validation).
- [X] Support for under/over sampling the majority/minority classes, controlled by the user.
- [X] Train and test scripts
- [X] Script to save the last trained model with test statistics
- [X] Create user interface to train, test, predict and serve model (need to improve)
- [X] Create unit tests (need to improve)
- [X] Added support for Streamlit web application to control the parameters
- [ ] Create API for model serving/deploy
- [ ] User dynaconf library to centralize application configurations
- [ ] Save "class_names" after training as model configuration