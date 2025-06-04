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
from keras.backend import manual_variable_initialization
from matplotlib import pyplot
from sklearn.utils import class_weight
from tensorflow.keras.optimizers import SGD

from wandb.keras import WandbCallback, WandbMetricsLogger, WandbModelCheckpoint

# tf.config.threading.set_intra_op_parallelism_threads(31)
# tf.config.threading.set_inter_op_parallelism_threads(31)


# ABOUT: used in other script build_compile.py
# Has data loading in here

manual_variable_initialization(True)

pd.set_option("max_colwidth", 800)

#### See build.py for all other functions not touched


# #### To save out examples of augmented images. NEED TO FIX OR REMOVE THIS
# # function inputs the saveims (y/n) as defined in config, and where aug images should be saved to
# # function returns the directory and prefix for saving images
# def vars_savedir(
#     parentdir=config.aug_dir,
#     catdirs=config.category_dirs,
#     saveims=config.save_ims,
#     saveto=config.data_gen,
# ):
#     if saveims == "yes":
#         save_to_dir_set = f"{saveto}train/"
#         save_prefix_set = "aug"
#         # set up directory structure for where the augmented images will go
#         # from the list of all classes, remove the ones we don't want for this model
#         # category_dirs = catdirs
#         makedirs(save_to_dir_set, exist_ok=True)
#     else:
#         save_to_dir_set = None
#         save_prefix_set = ""
#     return save_to_dir_set, save_prefix_set


### define function for converting image to grayscale, function will be used in data loading
# NEED TO ADD THIS BACK IN
def rgbtogray(image):
    tf.image.rgb_to_grayscale(image)
    return image


#### find class weights (i.e. if you have uneven classes), depends on what's set in config file
# note ONLY training data b/c class weights should only be applied to training data not val
def classweights(
    labels_dict,
    wts_use,
    trainlabels,
    balance=True,
    setclassimportance=[],
    num_train_imgs=0,
    train_cat_cts=[],
):
    """
    wts_use (str): if set to "yes", then class weights will be set and returned based on whether set balance var to True or set setweights
    trainlabels: list of labels which will be used to calculate the weights. Used in any scenario where class weights are required
    Note: Either balance OR setclassimportance must be used if using weights
    balance (boolean): if True, will do it based on unequal balance of number imgs per class.
    setclassimportance (list): if balance is False, it will use this list of predefined (predetermined) percentages that add up to 100% representing mportance by class. N. Note, when using balance = True, it's assuming each class is equal, so 1/6 = 16.67% per class. When using this instead, set it so that dry, wet, etc are the percentages we want (i.e. usually to underweight obs).
    num_train_imgs (int): if using setclassimportance, this is needed as it is the total number of images in the training set and used in calculation of weights.
    train_cat_cts (list): count of images in each class, used for weight calculation. Should be alphabetical

    Returns dictionary of each class and its corresponding weight
    """
    # # Initialize the LabelEncoder to convert from string classes to values (0-5 if 6-class)
    # le = preprocessing.LabelEncoder()
    # # Fit the encoder on the unique class labels (it automatically maps strings to numbers)
    # le.fit(trainlabels)
    # # Transform the string labels to numeric values (0-5)
    # labels_ind = le.transform(trainlabels)

    # print(trainlabels)

    labels_ind = [labels_dict[catname] for catname in trainlabels]

    # try doing class weighting on trainlables instead of labels_ind (values)
    if wts_use == "yes":
        if balance:
            print(
                "using class weights based solely on unequal balance of imgs per class, i.e. gives each class equal importance/influence in model build via loss function"
            )
            class_weights = class_weight.compute_class_weight(
                class_weight="balanced",
                classes=np.unique(labels_ind),
                y=labels_ind,
            )
        else:
            print(
                "using class weights as defined by the given importance/influence of each class, which sum to 1 in total. Doing the normal way is 1/6 per class (if 6 classes) but we may want to force severe snow to make up 1/2 of total influence, which we can do using this method. All classes just have to sum to 1. "
            )
            print(setclassimportance)
            class_weights = []
            for i in range(0, len(train_cat_cts)):
                print(num_train_imgs)
                ifeq = setclassimportance[i] * num_train_imgs
                print(ifeq)
                print(train_cat_cts[i])
                wt = ifeq / train_cat_cts[i]
                print(wt)
                class_weights.append(wt)
        class_weight_set = dict(zip(np.unique(labels_ind), class_weights))
    else:
        class_weight_set = None
    print(f"Class weight set is {class_weight_set}")
    print("through new version of class weighting")
    return class_weight_set


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
