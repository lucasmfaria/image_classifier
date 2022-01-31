# Image Classifier
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Tensorflow](https://badges.aleen42.com/src/tensorflow.svg)](https://www.tensorflow.org)


Need to train **your own** Image Classifier, with **your own dataset**, but don't know how to? Or maybe you know how to do it, 
but don't want to worry about neural network implementation details? Ok, this is your place.

The code is built to train a **ConvNet (Convolutional Neural Network) based image classifier**. Probably you won't 
need to change parameters, but even if you need to (because the classifier didn't get a good score) they are easy to 
change.

The code covers image classification for **binary or multiclass problems**. It uses **transfer learning** with pre-trained 
**VGG16** model. Tested with python 3.9 (so far).

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

## 🔨 Windows users
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

## 🐳 Docker users
You can use the images based on "https://hub.docker.com/r/lucasmfaria/image_classifier" repository, with the python requirements and pre-trained weights already in place. If you want to build your own image, you can use the Dockerfile on this project.

```
- Move your image dataset into the folder 'data', with the classes separation. Final directory 
example for a multiclass classification: 'data/dataset/class1', 'data/dataset/class2', 'data/dataset/class3'.
- For Streamlit app (http://localhost:8501/) -> run "docker run -it --rm -v [YOUR_DATA_PATH]:/opt/data -v [YOUR_MODELS_PATH]:/opt/models --name streamlit -p 8501:8501 lucasmfaria/image_classifier:latest streamlit run web_app.py"
- For create_splits.py -> run "docker run -it --rm -v [YOUR_DATA_PATH]:/opt/data --name create_splits lucasmfaria/image_classifier:latest python ./scripts/create_splits.py"
- For train.py -> run "docker run -it --rm -v [YOUR_DATA_PATH]:/opt/data -v [YOUR_MODELS_PATH]:/opt/models --name train lucasmfaria/image_classifier:latest python ./scripts/train.py"
- For test.py -> run "docker run -it --rm -v [YOUR_DATA_PATH]:/opt/data -v [YOUR_MODELS_PATH]:/opt/models --name test lucasmfaria/image_classifier:latest python ./scripts/test.py"
```

If you are on Windows platform, you can use the PowerShell scripts inside the "docker" directory that automate the code above.

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
- [X] Added support for Docker containers (python requirements and pre-trained neural net already in place)
- [ ] Create API for model serving/deploy
- [ ] User dynaconf library to centralize application configurations
- [ ] Save "class_names" after training as model configuration