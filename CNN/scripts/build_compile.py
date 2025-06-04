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

# ABOUT: called directly in __main__.
# This whole script is ONE function
# Uses function from build_baseline.py
# Uses function from build_bn_dropout.py
# Uses function from build_transferlearning.py (although not really being used right now, see code below redundant)




def read_imgs_as_np_array(listims, listlabels):
    """
    Read in images as pixels and skip those that are severely corrupted.
    Note the checking for severe corruption really isn't needed for training bc we have a preprocessing step that removes all the poorly corrupted images BUT keeping this code because it will be helpful for inference.
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
    SHUFFLE_BUFFER_SIZE = config.SHUFFLE_BUFFER_SIZE,
    BATCH_SIZE = config.BATCH_SIZE):


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
    train_labels = df_train["img_cat"]  # just for grabbing class weights

    train_images = list(df_train["img_orig"])  # 3/12
    train_labels = list(df_train["img_cat"])
    # train_labels = df_train["img_cat"]#  Replace with numeric or one-hot encoded labels
    print("through train")
    val_images = list(df_val["img_orig"])  # 3/12
    val_labels = list(df_val["img_cat"])
    print("HERE66")
    print(len(val_images))
    print(len(train_images))

    print(df_val.head())

    print("heretoseeifobs")
    print(np.unique(train_labels))

    # for reference print to console
    # df_cts = (
    #     df[["phase", "img_cat", "img_name"]]
    #     .groupby(["phase", "img_cat"])
    #     .size()
    #     .reset_index(name="counts")
    # )
    # print(df_cts)

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
    print("check here traincatcounts!!")
    print(traincatcounts)

    dict_cat_str_ind = helper_fns_adhoc.cat_str_ind_dictmap()

    print("before running")
    print(len(train_images))
    print(len(train_labels))
    imgs_train, cats_train = read_imgs_as_np_array(train_images, train_labels)
    print("here!!")
    print(len(train_images))
    print(len(imgs_train))

    print("before running val")
    print(len(val_images))
    print(len(val_labels))
    imgs_val, cats_val = read_imgs_as_np_array(val_images, val_labels)
    print("here!!")
    print(len(val_images))
    print(len(imgs_val))

    

    print("place2")
    print(type(imgs_train))
    # print(imgs_train[0:5])
    # print(imgs_train[0][0])


    label_encoding_train = [dict_cat_str_ind[catname] for catname in cats_train]
    label_encoding_val = [dict_cat_str_ind[catname] for catname in cats_val]
    # to subset
    # label_encoding_train = [inputdictmap[catname] for catname in train_labels[0:SUBSETNUM]]
    # label_encoding_val = [inputdictmap[catname] for catname in val_labels[0:SUBSETNUM]]

    

    train_labels_one_hot = np.eye(cat_num)[label_encoding_train]
    val_labels_one_hot = np.eye(cat_num)[label_encoding_val]

    print("place4")
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

    print("part6")
    print(type(dataset_val))
    print(type(dataset_train))
    print("PART6VAL")
    print(dataset_val)
    # Count the number of elements (images) in the dataset
    num_images = sum(1 for _ in dataset_val)
    print(f"Number of images in the dataset: {num_images}")
    # Create a TensorFlow dataset from the NumPy array
    # dataset = tf.data.Dataset.from_tensor_slices(image_array)

    # Print dataset structure
    print(dataset_train)

    dataset_train = dataset_train.shuffle(SHUFFLE_BUFFER_SIZE)  # Shuffle data
    dataset_train = dataset_train.batch(BATCH_SIZE)  # Batch the data
    dataset_train = dataset_train.prefetch(tf.data.AUTOTUNE)  # Optimize pipeline

    dataset_val = dataset_val.shuffle(SHUFFLE_BUFFER_SIZE)  # Shuffle data
    print("check val size tf dataset")
    print(dataset_val)
    dataset_val = dataset_val.batch(BATCH_SIZE)  # Batch the data
    dataset_val = dataset_val.prefetch(tf.data.AUTOTUNE)  # Optimize pipeline

    print("part7")
    print(type(dataset_val))
    print(dataset_val)

    return dataset_train, dataset_val, train_labels, val_labels, numims_train, traincatcounts

def compile_model(model, train_size, batchsize, lr_init = config.lr_init, lr_opt=config.lr_opt, lr_after_num_of_epoch =config.lr_after_num_of_epoch, lr_decayrate = config.lr_decayrate, momentum = config.momentum, evid = config.evid, evid_lr_init = config.evid_lr_init ):
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
        # 6/4/2025 -- need to add the learning rate scheduler in here after getting running first


        if lr_opt == True:
            num_of_batches = train_size / batchsize  
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=lr_init,
                decay_steps=num_of_batches
                * lr_after_num_of_epoch,  # how it's defined here, decay step will mean update after every X epochs
                decay_rate=lr_decayrate,  # https://stackoverflow.com/questions/65620186/exponentialdecay-learning-rate-schedule-with-staircase-true-changes-the-traini
                staircase=False,  # if set to True, then the learning rate will *jump* to 99% of initial learning rate for the 2nd epoch. Set to False meants it's actually updating every batch, jst a tiny amount such that after 2 epochs its been brought down to 99% of inital lr
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
