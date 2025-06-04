# System and file handling
import os
from glob import glob

# Image processing
import cv2
import numpy as np
import pandas as pd
from PIL import ImageFile

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import preprocess_input

# Configure PIL to avoid image errors
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Local project files
import _config as config
import helper_fns_adhoc

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