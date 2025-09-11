# SOURCE: Evidential deep learning code comes from NCAR MILES Group: https://miles.ucar.edu/software/milesguess/
# Also from example in docstring of CategoricalDNN from: https://github.com/ai2es/miles-guess/blob/main/mlguess/keras/models.py

# Following from running evid NN from example: https://github.com/ai2es/miles-guess/blob/main/notebooks/classifier_example.ipynb
# Scripts: https://github.com/ai2es/miles-guess/tree/main/mlguess/keras

import yaml
import numpy as np
import pandas as pd
from evid_milesguess_models import CategoricalDNN
import tensorflow as tf
import evid_milesguess_callbacks

# Things to fix before running post downstream model:
# 1) saving model, and don't need to save the training_log.csv (which shows results by epoch)
# 2) track on w&b?
# 3) major updates getting things to loop through and work by tracker (no hyptuning needed, just one off runs)

# YAML-like model config converted into a Python dictionary
# Mimicing off of miles-guess example: https://github.com/ai2es/miles-guess/blob/main/config/ptype/evidential.yml
config = {
    "model": {
        "activation": "leaky_relu",
        "annealing_coeff": 34,
        "batch_size": 1130,
        "dropout_alpha": 0.11676011477923032,
        "epochs": 5,  # update here
        "evidential": True,
        "n_inputs": 84,
        "hidden_layers": 4,
        "hidden_neurons": 212,
        "l2_weight": 0.000881889591229087,
        "loss": "evidential",
        "lr": 0.004800502096767794,
        "n_classes": 4,
        "optimizer": "adam",
        "output_activation": "linear",
        "use_dropout": 1,  # You might want to convert this to True (bool)
        "verbose": 1,
    },
    "callbacks": {
        "CSVLogger": {
            "append": 0,
            "filename": "/home/csutter/DRIVE-clean/Evid/training_log.csv",
            "separator": ",",
        },
        "EarlyStopping": {
            "mode": "max",
            "monitor": "val_ave_acc",
            "patience": 9,
            "restore_best_weights": 1,
            "verbose": 0,
        },
        "ReduceLROnPlateau": {
            "factor": 0.1,
            "min_lr": 1e-15,
            "mode": "max",
            "monitor": "val_ave_acc",
            "patience": 3,
            "verbose": 0,
        },
    },
}

print(config["model"])
print(config["callbacks"])
# ev_mlp = CategoricalDNN(**config["model"])


# # Parameters from your model config
# n_samples = 1000
# n_inputs = 84
# n_classes = 4


# # Generate random features as floats
# # For testing, just random numbers in [0,1)
# x_train = np.random.rand(n_samples, n_inputs).astype(np.float32)

# # Generate random integer class labels in [0, n_classes)
# y_train_int = np.random.randint(low=0, high=n_classes, size=n_samples)

# # If your model expects one-hot encoded labels (likely if loss='categorical_crossentropy'), do this:
# y_train = tf.keras.utils.to_categorical(y_train_int, num_classes=n_classes)

# # Now you can run:
# history = ev_mlp.fit(x_train, y_train)

n_samples = 1000
n_features = 23
n_classes = 5
x_train = np.random.random(size=(n_samples, n_features))
y_train = np.random.randint(low=0, high=n_classes, size=n_samples)
x_val = np.random.random(size=(n_samples, n_features))
y_val = np.random.randint(low=0, high=n_classes, size=n_samples)

# hidden_layers=2,
# evidential=True,
# activation='relu',
# n_classes=n_classes,
# n_inputs=n_features,
# epochs=10,
# annealing_coeff=1.5,
# lr=0.0001,
# verbose = 1

# from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau

# callbacks = [
#     CSVLogger(filename='training_log.csv', append=False, separator=','),
#     EarlyStopping(monitor='val_accuracy', patience=9, mode='max', restore_best_weights=True, verbose=0),
#     ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=3, min_lr=1e-15, mode='max', verbose=0)
# ]

callbacks = evid_milesguess_callbacks.get_callbacks(config["callbacks"])

### Evidential
model = CategoricalDNN(**config["model"], callbacks=callbacks)
hist = model.fit(x_train, y_train, validation_data=(x_val, y_val))
p_with_uncertainty = model.predict(x_train, return_uncertainties=True, batch_size=5000)

# print(p_with_uncertainty[0:10])
