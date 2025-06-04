# imports
import os
from os import listdir, makedirs

# import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from keras.models import Sequential
from PIL import Image, ImageFile
from sklearn import preprocessing  # for converting class name to value.
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import (  # EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam, schedules

import config
import custom_callbacks

ImageFile.LOAD_TRUNCATED_IMAGES = True
import itertools
from glob import glob

import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from matplotlib import pyplot
from sklearn.utils import class_weight
from tensorflow.keras.optimizers import SGD

from wandb.keras import WandbCallback, WandbMetricsLogger, WandbModelCheckpoint

# tf.config.threading.set_intra_op_parallelism_threads(31)
# tf.config.threading.set_inter_op_parallelism_threads(31)





# split into 3 if want - build, optimizer, compile

# old way
# def optimizer(lr=0.01):
#     return SGD(learning_rate=lr)

# new way w/ learning rate optimizer
def optimizer_fn(
    train_size, batchsize, lr_init=config.lr_init, lr_decr=config.lr_decayrate
):

    if config.lr_opt == True:
        num_of_batches = train_size / batchsize  # config.batch_size_def  # per epoch
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr_init,
            decay_steps=num_of_batches
            * config.lr_after_num_of_epoch,  # how it's defined here, decay step will mean update after every X epochs
            decay_rate=lr_decr,  # https://stackoverflow.com/questions/65620186/exponentialdecay-learning-rate-schedule-with-staircase-true-changes-the-traini
            staircase=False,  # if set to True, then the learning rate will *jump* to 99% of initial learning rate for the 2nd epoch. Set to False meants it's actually updating every batch, jst a tiny amount such that after 2 epochs its been brought down to 99% of inital lr
        )
        lr_use = lr_schedule

        print(
            f"\n using learning rate optimization with small decrease after every batch, i.e. staircase set False, \n such that the after {config.lr_after_num_of_epoch} epoch(s), i.e. decay steps = {num_of_batches} batches per epoch x {config.lr_after_num_of_epoch} epoch, the learning rate has been reduced to {lr_decr} of the previous learning rate. \n Initial lr is {lr_init} \n"
        )
    else:
        lr_use = 0.01
        print(f"using fixed learning rate of {lr_use}")

    optimizer_use = keras.optimizers.SGD(learning_rate=lr_use)  # lr_schedule

    return optimizer_use


# def imsize_model(color=config.colormode):
#     """return image size including channels, as a tuple, used as input shapr for model build. Depends on whether rgb or grayscale set in config"""
#     if config.colormode == "grayscale":
#         inputshape = (config.imheight, config.imwidth, 1)
#         print("using grayscale images with 1 channel")
#     elif config.colormode == "rgb":
#         inputshape = (config.imheight, config.imwidth, 3)
#         print("using rgb images with 3 channels")
#     else:
#         print("something off with colormode set in config. Check config")
#     return inputshape


def model_compile(
    model, loss_fn, performance_metrics, traindata_size, f, lr_init, lr_decr
):

    if config.evid:
        # from MILES_models.py
        print("setting specific evidential optimizer")
        optimizer_use = keras.optimizers.Adam(
            learning_rate=config.evid_lr_init, beta_1=0.9, beta_2=0.999, epsilon=1e-7
        )
    else:
        optimizer_use = optimizer_fn(traindata_size, batchsize, lr_init, lr_decr)
    # 2/13
    sgd_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    # this functin is called in build_compile.py, so that is where we designate the loss function to be the evidential one
    model.compile(
        loss=loss_fn,
        metrics=performance_metrics,
        optimizer=sgd_optimizer,  # optimizer_use
    )
    return model


def create_callbacks_list(savebestweights, earlystop_patience = config.earlystop_patience, evid = config.evid):

    print("started model_fitting")
    print(savebestweights)

    checkpoint = ModelCheckpoint(
        savebestweights,
        monitor="val_accuracy",
        verbose=0,
        save_best_only=True,
        mode="max",
    )  # saves out best model locally


    es = custom_callbacks.EarlyStopping(
        monitor="val_accuracy", mode="max", patience=earlystop_patience
    )

    # wb_metricslog = WandbMetricsLogger(
    #     log_freq="epoch"
    # )  # this plots loss and accuracy curves to w&b UI (under "epochs" dropdown). It also logs learning rate. These are plots in the UI dropdown where tables are, but also in the "Overview" part of that run. Can adjust log freq if want to do by batch rather than epoch
    # wb_checkpoint = WandbModelCheckpoint(filepath = savebestweights, monitor = "val_acc", save_best_only=True, Mode = "max") # this and "checkpoint" above essentially do the same thing but saving out best model also as an artifact on w&b too

    print("got through es")
    callbacks_list = [
        checkpoint,
        es,
        # wb_metricslog,
    ]
    if evid:
        print("evidential learing, setting learning rate reduction on plateau")
        reducelr = ReduceLROnPlateau(
            factor=0.1,
            min_lr=1.0e-15,
            mode="max",
            monitor="val_accuracy",
            patience=3,
            verbose=0,
        )
        # logger.info("... loaded ReduceLROnPlateau")
        callbacks_list.append(reducelr)
        print(
            "added another callback to reduce learning rate on plateau for adam loss function"
        )
    return callbacks_list

# hoping to deprecate this 6/4/2025. Moved the callbacks prep above, and moving the fit call into the main script
def model_fit(modelinput, traindata, epochsnum, valdata, classweights, savebestweights):

    # print("started model_fitting")
    # print(savebestweights)

    # checkpoint = ModelCheckpoint(
    #     savebestweights,
    #     monitor="val_accuracy",
    #     verbose=0,
    #     save_best_only=True,
    #     mode="max",
    # )  # saves out best model locally


    # es = custom_callbacks.EarlyStopping(
    #     monitor="val_accuracy", mode="max", patience=config.earlystop_patience
    # )

    # # wb_metricslog = WandbMetricsLogger(
    # #     log_freq="epoch"
    # # )  # this plots loss and accuracy curves to w&b UI (under "epochs" dropdown). It also logs learning rate. These are plots in the UI dropdown where tables are, but also in the "Overview" part of that run. Can adjust log freq if want to do by batch rather than epoch
    # # wb_checkpoint = WandbModelCheckpoint(filepath = savebestweights, monitor = "val_acc", save_best_only=True, Mode = "max") # this and "checkpoint" above essentially do the same thing but saving out best model also as an artifact on w&b too

    # print("got through es")
    # callbacks_list = [
    #     checkpoint,
    #     es,
    #     # wb_metricslog,
    # ]
    # if config.evid:
    #     print("evidential learing, setting learning rate reduction on plateau")
    #     reducelr = ReduceLROnPlateau(
    #         factor=0.1,
    #         min_lr=1.0e-15,
    #         mode="max",
    #         monitor="val_accuracy",
    #         patience=3,
    #         verbose=0,
    #     )
    #     # logger.info("... loaded ReduceLROnPlateau")
    #     callbacks_list.append(reducelr)
    #     print(
    #         "added another callback to reduce learning rate on plateau for adam loss function"
    #     )

    # , save_freq = "epoch" #checkpoint,  wb_checkpoint
    print("got through callbacks list.. starting history")
    print(classweights)
    print(classweights.keys())
    # for key in classweights:
    #     print(f"Key: {key}, Data Type of Key: {type(key)}")
    classweights = {
        int(k): v for k, v in classweights.items()
    }  # Ensure keys are integers


    print("checking output size")
    print(modelinput.output_shape)

    
    history = modelinput.fit(
        traindata,
        epochs=epochsnum,
        callbacks=callbacks_list,
        validation_data=valdata,
        batch_size=32,  # HERE SHOULD BE UPDATED
        # class_weight=classweights,
    )
    print("got through history")

    return history
