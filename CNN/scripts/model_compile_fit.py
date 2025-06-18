# Portions of this code were writen with the assistance of AI tools (e.g., ChatGPT).

# imports
# System and file handling
import os
from os import listdir, makedirs
import subprocess
import itertools
from glob import glob

# Other packages
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
from matplotlib import pyplot

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.optimizers import SGD, Adam, schedules

# scikit-learn
from sklearn.utils import class_weight
import sklearn.metrics as metrics

# Configure PIL to avoid image errors
ImageFile.LOAD_TRUNCATED_IMAGES = True

import _config as config # holds constants like BATCH_SIZE, cat_num, etc.
import class_weights  # likely defines weight functions
import helper_fns_adhoc  # must define cat_str_ind_dictmap()



def compile_model(model, train_size, batchsize, lr_init, lr_opt=config.lr_opt, lr_after_num_of_epoch =config.lr_after_num_of_epoch, lr_decayrate = config.lr_decayrate, momentum = config.momentum, evid = config.evid, evid_lr_init = config.evid_lr_init):
    """

    This function:
    - Compiles the model using categorical crossentropy loss and SGD optimizer with optional exponential learning rate decay
    - If `evid` is True, compiles the model using a custom evidential loss function with an Adam optimizer
    - Configures learning rate decay using `ExponentialDecay` if `lr_opt` is True
    - Uses predefined config values for learning rate, momentum, decay rate, and evidential loss hyperparameters

    Args:
        model (tf.keras.Model): Uncompiled Keras model.
        train_size (int): Number of training samples, used to calculate decay steps.
        batchsize (int): Batch size used during training.
        lr_init (float): Initial learning rate.
        lr_opt (bool): Whether to apply learning rate decay.
        lr_after_num_of_epoch (int): Number of epochs after which to decay learning rate.
        lr_decayrate (float): Learning rate decay rate.
        momentum (float): Momentum value for SGD optimizer.
        evid (bool): Whether to use evidential deep learning.
        evid_lr_init (float): Initial learning rate for evidential learning.

    Returns:
        model (tf.keras.Model): Compiled model ready for training.
    """

    if evid:
        print("evidential learning, loading in MILES GUESS loss function")
        print(f"using class weights? {classwts_normalized}")
        e = tf.Variable(1)
        report_epoch_callback = MILES_callbacks.ReportEpoch(e)
        print("setting specific evidential optimizer")
        optimizer_evid = keras.optimizers.Adam(
            learning_rate=evid_lr_init, beta_1=0.9, beta_2=0.999, epsilon=1e-7
        )
        model.compile(
            loss=evidential_cat_loss(
                evi_coef=evid_annealing_coeff,
                epoch_callback=report_epoch_callback,
                class_weights=classwts_normalized,# None or classwts_normalized, or [10, 10, 10, 10, 10, 10]  [1, 1, 1, 1, 1, 1], classwts_six_normalized, class_weight_set
                ),
            metrics=["accuracy"],
            optimizer=optimizer_evid, 
                )
        print("evidential model is set")
    else:

        if lr_opt == True:
            num_of_batches = train_size / batchsize  
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=lr_init,
                decay_steps=num_of_batches
                * lr_after_num_of_epoch,  # how it's defined here, decay step will mean update after every X epochs
                decay_rate=lr_decayrate,  # https://stackoverflow.com/questions/65620186/exponentialdecay-learning-rate-schedule-with-staircase-true-changes-the-traini
                staircase=False,  # update every batch, not jump update every epoch
            )
            print(
                f"\n using learning rate optimization with small decrease after every batch, i.e. staircase set False, \n such that the after {lr_after_num_of_epoch} epoch(s), i.e. decay steps = {num_of_batches} batches per epoch x {lr_after_num_of_epoch} epoch, the learning rate has been reduced to {lr_decayrate} of the previous learning rate. \n Initial lr is {lr_init} \n"
            )
            lr_use = lr_schedule
        else:
            print(f"using fixed learning rate of {lr_use}")
            lr_use = 0.01

        optimizer_use = tf.keras.optimizers.SGD(learning_rate=lr_use, momentum = momentum)  # lr_schedule
        
        # sgd_optimizer = tf.keras.optimizers.SGD(learning_rate=lr_init, momentum=momentum)
        model.compile(
            optimizer=optimizer_use, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"]
        )
    print(f"final model summary: {model.summary()}")

    return model

    
def train_fit(
    modelinput,
    traindata,
    valdata,
    callbacks_list,
    class_weights_use,
    evid = config.evid,
    epoch_set = config.epoch_set,
    BATCH_SIZE = config.BATCH_SIZE,
    # wbtable # LATER also figre out how to append # of epochs ran
    # for reading in df
    # append_details, # for w&b
    # wb_config={},
):
    """
    Trains a TF model using the provided training and validation datasets, tracks performance metrics across epochs, and prints summary statistics.

    This function:
    - Trains the model using `model.fit()` with the specified training/validation data, epochs, batch size, and callbacks
    - Logs and prints the maximum training and validation accuracy/loss achieved
    - Prints the number of epochs the model was trained for

    Args:
        modelinput (tf.keras.Model): Compiled Keras model to be trained.
        traindata (tf.data.Dataset): Training dataset.
        valdata (tf.data.Dataset): Validation dataset.
        callbacks_list (list): List of callbacks (e.g., EarlyStopping, ModelCheckpoint).
        epoch_set (int, optional): Number of training epochs (default from config).
        BATCH_SIZE (int, optional): Batch size for training (default from config).

    Returns:
        history (tf.keras.callbacks.History): Training history object containing loss and accuracy per epoch.
    """

    if evid: # class weights already incorporated in the loss function for evidential, so set it to None
        class_weights_use = None 
    
    for x, y in traindata.take(1):
        print("X shape:", x.shape)
        print("Y shape:", y.shape)


    history = modelinput.fit(
        traindata,
        epochs=epoch_set,
        callbacks=callbacks_list,
        validation_data=valdata,
        batch_size=BATCH_SIZE,  # HERE SHOULD BE UPDATED
        class_weight=class_weights_use,
    )

