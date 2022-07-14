from pathlib import Path
import gc
import tensorflow as tf
from tensorflow.keras.applications import vgg16, vgg19, densenet, resnet_v2, inception_v3, resnet50, resnet
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.backend import clear_session
from tensorflow.python.framework.ops import disable_eager_execution
import streamlit as st

DEFAULT_CHECKPOINTS_PATH = Path(__file__).resolve().parent.parent / 'models' / 'vgg16' / 'checkpoints'
DEFAULT_LOG_PATH = Path(__file__).resolve().parent.parent / 'models' / 'vgg16' / 'logs'

# TODO - change img_depth depending on features from dataset - 2048, 512 until now
def make_classifier(n_classes, n_hidden=512, img_height=7, img_width=7, img_depth=512):
    
    inputs = layers.Input(shape=(img_height, img_width, img_depth))
    x = layers.Flatten(name='flatten')(inputs)
    x = layers.Dense(n_hidden, activation='relu', name='dense_1')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(n_hidden, activation='relu', name='dense_2')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(n_hidden, activation='relu', name='dense_3')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    n_outputs = n_classes if n_classes != 2 else 1  # only one output neuron if it's a binary classification problem
    activation = 'softmax' if n_classes != 2 else 'sigmoid'  # sigmoid if it's a binary classification problem
    outputs = layers.Dense(n_outputs, activation=activation, name='output')(x)
    model = tf.keras.Model(inputs, outputs)

    return model


def make_model(n_classes, include_top_vgg=False, n_hidden=512, img_height=224, img_width=224, transfer_learning=True, base_model='vgg16'):
    """
    Creates a ConvNet classification model using a VGG16 pre-trained model for transfer learning.
    :param n_classes: int - number of classes required for the classification problem
    :param include_top_vgg: bool - whether or not to include the top of the pre-trained model
    :param n_hidden: int - number of hidden layers to add to the pre-trained model
    :param img_height: int - image height
    :param img_width: int - image width
    :return: tf.keras.Model - final model
    """
    if transfer_learning:
        weights = 'imagenet'
    else:
        weights = None
    
    if base_model == 'vgg16':
        base_model_net = vgg16.VGG16(include_top=False, weights=weights)
        preprocess_layer = vgg16.preprocess_input
        #base_model = vgg16.VGG16(include_top=False, pooling='max', weights=weights)
    if base_model == 'vgg19':
        base_model_net = vgg19.VGG19(include_top=False, weights=weights)
        preprocess_layer = vgg19.preprocess_input
    elif base_model == 'densenet201':
        base_model_net = densenet.DenseNet201(include_top=False, weights=weights)
        preprocess_layer = densenet.preprocess_input
    elif base_model == 'densenet169':
        base_model_net = densenet.DenseNet169(include_top=False, weights=weights)
        preprocess_layer = densenet.preprocess_input
    elif base_model == 'densenet121':
        base_model_net = densenet.DenseNet121(include_top=False, weights=weights)
        preprocess_layer = densenet.preprocess_input
    elif base_model == 'resnet152v2':
        base_model_net = resnet_v2.ResNet152V2(include_top=False, weights=weights)
        preprocess_layer = resnet_v2.preprocess_input
    elif base_model == 'resnet50':
        base_model_net = resnet50.ResNet50(include_top=False, weights=weights)
        preprocess_layer = resnet50.preprocess_input
    elif base_model == 'resnet152':
        base_model_net = resnet.ResNet152(include_top=False, weights=weights)
        preprocess_layer = resnet.preprocess_input 
    elif base_model == 'resnet101':
        base_model_net = resnet.ResNet101(include_top=False, weights=weights)
        preprocess_layer = resnet.preprocess_input
    elif base_model == 'inception_v3':
        base_model_net = inception_v3.InceptionV3(include_top=False, weights=weights)
        preprocess_layer = inception_v3.preprocess_input
    elif base_model == 'classifier':
        return make_classifier(n_classes, n_hidden, img_height, img_width)
    
    data_augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip('horizontal'),
        layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    inputs = layers.Input(shape=(img_height, img_width, 3))
    x = data_augmentation(inputs)
    x = preprocess_layer(x)
    x = base_model_net(x, training=False)
    #x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    #x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    #x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(n_hidden, activation='relu', name='dense_1')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(n_hidden, activation='relu', name='dense_2')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(n_hidden, activation='relu', name='dense_3')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    n_outputs = n_classes if n_classes != 2 else 1  # only one output neuron if it's a binary classification problem
    activation = 'softmax' if n_classes != 2 else 'sigmoid'  # sigmoid if it's a binary classification problem
    outputs = layers.Dense(n_outputs, activation=activation, name='output')(x)
    model = tf.keras.Model(inputs, outputs)

    return model


def make_simple_model(n_classes, include_top_vgg=False, n_hidden=512, img_height=224, img_width=224, transfer_learning=True):
    
    data_augmentation = tf.keras.Sequential([
            layers.experimental.preprocessing.RandomFlip('horizontal'),
            layers.experimental.preprocessing.RandomRotation(0.2),
        ])
    inputs = layers.Input(shape=(img_height, img_width, 3))
    x = data_augmentation(inputs)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(512, activation='relu', name='dense_1')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu', name='dense_2')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    n_outputs = n_classes if n_classes != 2 else 1  # only one output neuron if it's a binary classification problem
    activation = 'softmax' if n_classes != 2 else 'sigmoid'  # sigmoid if it's a binary classification problem
    outputs = layers.Dense(n_outputs, activation=activation, name='output')(x)
    model = tf.keras.Model(inputs, outputs)

    return model


def freeze_all_base_model(model, base_model='vgg16'):
    for layer in model.layers:
        if base_model in layer.name:
            for base_model_layer in layer.layers:
                base_model_layer.trainable = False


def unfreeze_last_base_model(model, which_freeze=15, base_model='vgg16'):
    for layer in model.layers:
        if base_model in layer.name:
            for base_model_layer in layer.layers[:which_freeze]:
                base_model_layer.trainable = False
            for base_model_layer in layer.layers[which_freeze:]:
                base_model_layer.trainable = True


def unfreeze_all_base_model(model, base_model='vgg16'):
    for layer in model.layers:
        if base_model in layer.name:
            for base_model_layer in layer.layers:
                base_model_layer.trainable = True


def print_vgg_trainable(model):
    for layer in model.layers:
        if 'vgg' in layer.name:
            for i, vgg_layer in enumerate(layer.layers):
                print(i, vgg_layer.name, vgg_layer.trainable)


def loss_definition(n_classes):
    return tf.keras.losses.CategoricalCrossentropy() if n_classes > 2 else tf.keras.losses.BinaryCrossentropy()


def initial_model(n_classes, n_hidden=512, img_height=224, img_width=224, seed=None, transfer_learning=True, base_model='vgg16'):
    if seed is not None:
        tf.random.set_seed(seed)

    model = make_model(n_classes=n_classes, n_hidden=n_hidden, img_height=img_height, img_width=img_width, transfer_learning=transfer_learning, base_model=base_model)
    #model = make_simple_model(n_classes=n_classes, n_hidden=n_hidden, img_height=img_height, img_width=img_width,
    #                   transfer_learning=transfer_learning)
    #model = make_classifier(n_classes=n_classes, n_hidden=n_hidden, img_height=img_height, img_width=img_width, img_depth=2048)
    if transfer_learning:
        freeze_all_base_model(model, base_model=base_model)
    else:
        unfreeze_all_base_model(model, base_model=base_model)
    
    return model


def callbacks_definition(log_path=DEFAULT_LOG_PATH, checkpoints_path=DEFAULT_CHECKPOINTS_PATH,
                         streamlit_callbacks=None, base_epochs=30, fine_tuning_epochs=30):
    tb = TensorBoard(log_dir=log_path)
    checkpoint = ModelCheckpoint(checkpoints_path / 'train_{epoch}.tf', verbose=1, save_weights_only=False,
                                 save_best_only=True, monitor='val_loss')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, restore_best_weights=True)
    
    
    callbacks = [tb, checkpoint, reduce_lr, early_stopping]
    #callbacks = [checkpoint, reduce_lr, early_stopping]
    if streamlit_callbacks is not None:  # used only with streamlit web application
        class ProgressBarCallback(tf.keras.callbacks.Callback):
            def __init__(self, base_epochs, fine_tuning_epochs):
                super().__init__()
                self.state = 'base'
                self.base_epochs = base_epochs
                self.actual_base_epochs = 0
                self.fine_tuning_epochs = fine_tuning_epochs
                self.placeholders = []
                self.epoch = 0

            def on_train_begin(self, logs=None):
                streamlit_callbacks[0]("###### STARTED " + self.state.upper() + " TRAINING")
                self.placeholders.append(st.empty())
                with self.placeholders[-1]:
                    streamlit_callbacks[1](0.0)

            def on_train_end(self, logs=None):
                with self.placeholders[-1]:
                    streamlit_callbacks[1](1.0)
                streamlit_callbacks[0]("###### FINISHED " + self.state.upper() + " TRAINING")
                if self.state == 'base':
                    self.state = 'fine_tuning'
                    self.actual_base_epochs = self.epoch + 1  # keep the epoch number even if early stopped

            def on_epoch_end(self, epoch, logs=None):
                self.epoch = epoch
                with self.placeholders[-1]:
                    epoch = epoch if self.state == 'base' else (epoch - self.actual_base_epochs)
                    streamlit_callbacks[1](
                        (epoch + 1) / (self.base_epochs if self.state == 'base' else (self.fine_tuning_epochs +
                                                                                      (self.base_epochs -
                                                                                       self.actual_base_epochs))))
        callbacks = callbacks + [ProgressBarCallback(base_epochs=base_epochs, fine_tuning_epochs=fine_tuning_epochs)]
        #ARRUMAR
        import matplotlib.pyplot as plt
        class timecallback(tf.keras.callbacks.Callback):
            def __init__(self):
                self.times = []
                # use this value as reference to calculate cummulative time taken
                self.timetaken = time.time()
            def on_epoch_end(self,epoch,logs = {}):
                self.times.append((epoch,time.time() - self.timetaken))
            def on_train_end(self,logs = {}):
                plt.xlabel('Epoch')
                plt.ylabel('Total time taken until an epoch in seconds')
                plt.plot(*zip(*self.times))
                plt.show()
        callbacks = callbacks + [timecallback()]
    return callbacks


def get_best_model_name(checkpoints_path=DEFAULT_CHECKPOINTS_PATH):
    # returns the last integer number inside the name of the best model file
    return sorted([(int(x.name.split('_')[1].split('.')[0]), x.name) for x in list(Path(checkpoints_path).iterdir()) if len(x.name.split('_')) > 1])[-1][1]


def load_best_model(checkpoints_path=DEFAULT_CHECKPOINTS_PATH):
    # loads the best model from checkpoints folder
    best_model_name = get_best_model_name(checkpoints_path=checkpoints_path)
    print('USER - Restoring model weights from the end of the best epoch:', best_model_name.split('_')[-1].split('.')[0])
    model = tf.keras.models.load_model(Path(checkpoints_path) / best_model_name)  # loads the best model even if early_stopping does not triggers
    return model


def train(model, train_ds, valid_ds, n_classes, base_epochs=30, fine_tuning_epochs=30, fine_tune_at_layer=15,
          fine_tuning_lr=0.001, callbacks=None, seed=None, transfer_learning=True, base_model='vgg16', checkpoints_path=DEFAULT_CHECKPOINTS_PATH, base_lr=0.001, metrics=['accuracy']):
    if seed is not None:
        tf.random.set_seed(seed)
    
    metrics_ = [tf.keras.metrics.AUC(curve='ROC', name='roc_auc') if metric == 'roc_auc' else metric for metric in metrics]
    metrics_ = [tf.keras.metrics.AUC(curve='PR', name='pr_auc') if metric == 'pr_auc' else metric for metric in metrics_]
    loss = loss_definition(n_classes=n_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_lr), loss=loss, metrics=metrics_)
    history = model.fit(train_ds, epochs=base_epochs, validation_data=valid_ds, callbacks=callbacks)
    #history = model.fit(train_ds, epochs=base_epochs, validation_data=valid_ds, callbacks=callbacks, validation_steps=valid_ds.cardinality())
    
    gc.collect()
    del model
    gc.collect()
    
    clear_session()
    #disable_eager_execution()
    model = load_best_model(checkpoints_path=checkpoints_path)

    if transfer_learning and (fine_tuning_epochs > 0):
        unfreeze_last_base_model(model, which_freeze=fine_tune_at_layer, base_model=base_model)

        total_epochs = base_epochs + fine_tuning_epochs
        loss = loss_definition(n_classes=n_classes)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tuning_lr), loss=loss, metrics=metrics_)

        if seed is not None:
            tf.random.set_seed(seed)

        history = model.fit(train_ds, epochs=total_epochs, validation_data=valid_ds, callbacks=callbacks,
                            initial_epoch=history.epoch[-1] + 1)
        
        gc.collect()
        del model
        gc.collect()
        
        clear_session()
        #disable_eager_execution()
        model = load_best_model(checkpoints_path=checkpoints_path)
    return model, history
