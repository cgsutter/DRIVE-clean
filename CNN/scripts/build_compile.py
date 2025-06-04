# imports
import os
import subprocess
from os import listdir, makedirs
import config
import class_weights
import data_input_pipeline
import custom_callbacks
import build_baseline
import build_bn_dropout
import helper_fns_adhoc

import cv2

# import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from keras.models import Sequential
from PIL import Image, ImageFile
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint  # EarlyStopping,
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam, schedules

# from src import data_input_pipeline as data_input_pipeline

ImageFile.LOAD_TRUNCATED_IMAGES = True
import itertools
from glob import glob

import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from keras.backend import manual_variable_initialization
from matplotlib import pyplot
from sklearn.utils import class_weight
from tensorflow.keras.optimizers import SGD

# comment out for now 6/3/25
# import wandb
# from src import MILES_callbacks as MILES_callbacks
# from src.MILES_loss import evidential_cat_loss as evidential_cat_loss
# from wandb.keras import WandbCallback, WandbMetricsLogger, WandbModelCheckpoint


def read_imgs_as_np_array(listims, listlabels):
    """ Note: this docstring is AI assisted
    
    This function reads a list of image file paths and corresponding labels, processes each image, and returns a list of preprocessed image arrays along with their labels.

    Images are:
    - Read using OpenCV
    - Converted from BGR to RGB
    - Cropped (top 20%)
    - Resized to a target size defined in config.TARGET_SIZE
    - Preprocessed using MobileNet's preprocessing function

    Any unreadable or corrupted images are skipped and recorded.

    Args:
        listims (list of str): List of image file paths.
        listlabels (list): Corresponding labels for the images.

    Returns:
        images_pixel (list of np.array): List of preprocessed image arrays.
        labels_imgs (list): Labels corresponding to successfully processed images.
    """
    brokenimgs = []
    images_pixel = []
    labels_imgs = []
    for im_i in range(0, len(listims)):  # len(listims) range(8800, 8900)
        if im_i % 1000 == 0:
            print(im_i)

        try:
            image_array = cv2.imread(listims[im_i], cv2.IMREAD_UNCHANGED)
            # print(image_array)
            # Convert BGR (OpenCV default) to RGB
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

            # Crop 20% from the top)
            h, w, _ = image_array.shape
            crop_height = int(0.2 * h)
            image_array = image_array[crop_height:, :, :]

            image_array = cv2.resize(image_array, config.TARGET_SIZE)  # Resize

            # Apply MobileNet preprocessing
            image_array = preprocess_input(image_array)

            images_pixel.append(image_array)
            labels_imgs.append(listlabels[im_i])

        except:
            brokenimgs.append(listims[im_i])
    # for broken in brokenimgs:
    #     print(f"total number of corrupted images {len(broken)}")
    #     print(f"corrupted image that can't be fixed: {broken}")
    print(f"number of broken imgs is {len(brokenimgs)}")
    return images_pixel, labels_imgs

def create_tf_datasets(tracker,
    cat_num = config.cat_num,
    BATCH_SIZE = config.BATCH_SIZE):
    """ Note: this docstring is AI assisted

    This function:
    - Reads a CSV file specified by `tracker` containing image file paths and label metadata
    - Filters the data into training and validation subsets based on the `innerPhase` column
    - Loads and preprocesses images using `read_imgs_as_np_array` (includes cropping, resizing, and normalization)
    - Encodes class labels to one-hot format using a category-to-index dictionary
    - Converts image and label arrays to TensorFlow datasets
    - Shuffles, batches, and prefetches the datasets for efficient training

    Args:
        tracker (str): Path to CSV file with image paths and labels.
        cat_num (int): Number of output classes (default from config).
        SHUFFLE_BUFFER_SIZE (int): Buffer size for shuffling (default from config).
        BATCH_SIZE (int): Batch size for training (default from config).

    Returns:
        dataset_train (tf.data.Dataset): Preprocessed and batched training dataset.
        dataset_val (tf.data.Dataset): Preprocessed and batched validation dataset.
        train_labels (list): Original labels for training data.
        val_labels (list): Original labels for validation data.
        numims_train (int): Number of training images.
        traincatcounts (pd.Series): Class distribution in the training set.
    """

    ### STEP 1: LOAD IN DATA (TF DATASETS) and PREP WEIGHTS
    df = pd.read_csv(f"{tracker}")
    print(f"using {tracker}")
    df_train = df[df[f"innerPhase"] == "innerTrain"]
    df_val = df[df[f"innerPhase"] == "innerVal"]
        
    # create variable and list that are sometimes used as an input in class weights function

    print(f"NUMBER OF TRAIN:: {len(df_train)}")
    print(f"NUMBER OF VAL:: {len(df_val)}")
    numims_train = len(df_train)  # var used to calc what weights should be
    df_train_size = len(df_train)

    train_images = list(df_train["img_orig"])
    train_labels = list(df_train["img_cat"])


    val_images = list(df_val["img_orig"]) 
    val_labels = list(df_val["img_cat"])


    dftraincatcount = (
        df_train[["img_cat", "img_name"]]
        .groupby(["img_cat"])
        .size()
        .reset_index(name="counts")
        .sort_values(["img_cat"])
    )
    traincatcounts = dftraincatcount[
        "counts"
    ]  # this is a list used to calc what weights should be given the amount that are in each cat

    print(traincatcounts)

    dict_cat_str_ind = helper_fns_adhoc.cat_str_ind_dictmap()

    imgs_train, cats_train = read_imgs_as_np_array(train_images, train_labels)

    imgs_val, cats_val = read_imgs_as_np_array(val_images, val_labels)

    label_encoding_train = [dict_cat_str_ind[catname] for catname in cats_train]
    label_encoding_val = [dict_cat_str_ind[catname] for catname in cats_val]

    train_labels_one_hot = np.eye(cat_num)[label_encoding_train]
    val_labels_one_hot = np.eye(cat_num)[label_encoding_val]

    # Ensure dtype is float32 for ims and int for labels
    images_fortfd_train = np.array(imgs_train, dtype=np.float32)
    labels_fortfd_train = np.array(train_labels_one_hot, dtype=np.int32)
    images_fortfd_val = np.array(imgs_val, dtype=np.float32)
    labels_fortfd_val = np.array(val_labels_one_hot, dtype=np.int32)

    print(
        "part5 data prepped before making it to tensor slices, see if it slows down here and if so then it's just the tf dataset creation that is taking time"
    )
    dataset_train = tf.data.Dataset.from_tensor_slices(
        (images_fortfd_train, labels_fortfd_train)
    )

    dataset_val = tf.data.Dataset.from_tensor_slices(
        (images_fortfd_val, labels_fortfd_val)
    )

    # Count the number of elements (images) in the dataset
    num_images = sum(1 for _ in dataset_val)
    print(f"Number of images in the dataset: {num_images}")

    # Shuffling with buffer size, take buffer size number of images in order, randomly select one from it, and then shift the buffer down one to maintain buffer size, select another, etc. True random shuffling will be complete if buffer >= total number of samples. (see: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle) 
    dataset_train = dataset_train.shuffle(numims_train)  # Shuffle data
    dataset_train = dataset_train.batch(BATCH_SIZE)  # Batch the data
    dataset_train = dataset_train.prefetch(tf.data.AUTOTUNE)  # Optimize pipeline

    dataset_val = dataset_val.batch(BATCH_SIZE)  # Batch the data
    dataset_val = dataset_val.prefetch(tf.data.AUTOTUNE)  # Optimize pipeline

    return dataset_train, dataset_val, train_labels, val_labels, numims_train, traincatcounts

def compile_model(model, train_size, batchsize, lr_init = config.lr_init, lr_opt=config.lr_opt, lr_after_num_of_epoch =config.lr_after_num_of_epoch, lr_decayrate = config.lr_decayrate, momentum = config.momentum, evid = config.evid, evid_lr_init = config.evid_lr_init):
    """ Note: this docstring is AI assisted

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

    
def build(
    modelinput,
    traindata,
    valdata,
    callbacks_list,
    epoch_set = config.epoch_set,
    BATCH_SIZE = config.BATCH_SIZE,
    # wbtable # LATER also figre out how to append # of epochs ran
    # for reading in df
    # append_details, # for w&b
    # wb_config={},
):
    """ Note: this docstring is AI assisted
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
    history = modelinput.fit(
        traindata,
        epochs=epoch_set,
        callbacks=callbacks_list,
        validation_data=valdata,
        batch_size=BATCH_SIZE,  # HERE SHOULD BE UPDATED
        # class_weight=classweights,
    )

    type_history = type(history)
    print(f"TYPE HISTORY IS {type_history}")

    # should remove
    print(history.history.keys())
    # print(history.history.values())
    # print(type(history.history.values()))

    final_val_acc = max(history.history["val_accuracy"])
    final_train_acc = max(history.history["accuracy"])
    final_val_loss = max(history.history["val_loss"])
    final_train_loss = max(history.history["loss"])

    number_of_epochs_it_ran = len(history.history["loss"])
    print(number_of_epochs_it_ran)
    # print(train_accuracy = history.history['accuracy'])
    # print(val_accuracy = history.history['val_accuracy'])

    # print(history.val_acc)

    print(f"moving on to table")

    # run.log(
    #     {f"model_table": wbtable}
    # )  # here, there would only be multiple (bar charts, after joining) if doing hyperparam tuning. Thus, joining by "model" here will help choose the best hyperparam for that specific split

    # print("past save table")

    # run.finish()

    comment = f"val images: {len(valdata)} imgs" f"train images: {len(traindata)} imgs"

    # titleloss_final = f"{titleloss} \n {comment}"
    # titleacc_final = f"{titleacc} \n {comment}"
    # # save model build history by epoch in loss and accuracy curves
    # src.plots_charts.plots_acc_loss(
    #     history, save_curves_to, titleloss_final, titleacc_final
    # )

    # print(type(config.experiment))
