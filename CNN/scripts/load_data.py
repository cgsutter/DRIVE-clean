# Portions of this code were writen with the assistance of AI tools (e.g., ChatGPT).

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
# from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess


# Configure PIL to avoid image errors
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Local project files
import _config as config
import helper_fns_adhoc

def read_imgs_as_np_array(listims, listlabels): # remove arch input
    """
    
    This function reads a list of image file paths and corresponding labels, crops & resizes, checks for broken images, and returns a list of the read-in image arrays along with their labels.

    Images are:
    - Read using OpenCV
    - Converted from BGR to RGB
    - Cropped (top 20%)
    - Resized to a target size defined in config.TARGET_SIZE

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
        if im_i % 1000 == 0: # to show progress as data loads
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

            images_pixel.append(image_array)
            labels_imgs.append(listlabels[im_i])

        except:
            brokenimgs.append(listims[im_i])
    # for broken in brokenimgs:
    #     print(f"total number of corrupted images {len(broken)}")
    #     print(f"corrupted image that can't be fixed: {broken}")
    print(f"number of broken imgs is {len(brokenimgs)}")
    return images_pixel, labels_imgs

def preprocess_and_aug(image_np, arch_for_preprocess, augflag):
    """This function takes one image (np array), adjusts it with random augmentation IF aug flag is True, and then regardless of augmentation or not, preprocesses it according to the architecture used. 
    
    Note that augmentations functions require as inputs images to be a TF tensor that ranges 0 and 1 (divide by 255). Then and after augmentation, which may return values slightly below 0 or above 1, the values need to be converted back to range 0 to 255 for architecture specific preprocessing, so they are again adjusted to be first clipped to 0 and 1 and then multiplied by 255.

    Args:
        image_np (np array): An image
        arch_for_preprocess (str): arhcitecture name
        augflag (bool): 

    Returns:
       Image np array that is preprocessed
    """

    # print("entered img-level fn")
    image_tensor = tf.convert_to_tensor(image_np)
    # print("converted to tensor")
    # Apply augmentation
    if augflag:
        # print("inside augmentation")
        # these augmentations require values ranging from 0 to 1. Immediately following augmentation, it will be preprocessed back according to the architecture chosen
        image_tensor = tf.cast(image_tensor, tf.float32) / 255.0
        image_tensor = tf.image.random_flip_left_right(image_tensor)
        image_tensor = tf.image.random_brightness(image_tensor, max_delta=0.1)
        image_tensor = tf.image.random_contrast(image_tensor, lower=0.8, upper=1.2)

        # After augmentation - need to convert back to 0 to 255, so first clip to be between 0 and 1 then multipy by 255 
        image_tensor = tf.clip_by_value(image_tensor, 0.0, 1.0)
        # Scale back up to 0-255 for preprocessing
        image_tensor = image_tensor * 255.0

    # preprocessing unique to the architecture
    # print("starting preprocessing for arch")
    if arch_for_preprocess == "densenet":
        image_array = densenet_preprocess(image_tensor)
    elif arch_for_preprocess == "incep":
        image_array = inception_preprocess(image_tensor)
    elif arch_for_preprocess == "mobilenet":
        image_array = mobilenet_preprocess(image_tensor)
    elif arch_for_preprocess == "resnet":
        image_array = resnet_preprocess(image_tensor)
    elif arch_for_preprocess == "vgg16":
        image_array = vgg_preprocess(image_tensor)
    elif arch_for_preprocess == "xcep":
        image_array = xception_preprocess(image_tensor)
    
    # print("finsihed preprocessing for arch")

    return image_array

def prepare_tf_dataset(imginput, labelsinput, arch, aug):

    # prepare labels
    dict_catKey_indValue, dict_indKey_catValue = helper_fns_adhoc.cat_str_ind_dictmap()
    label_encoding = [dict_catKey_indValue[catname] for catname in labelsinput]
    labels_one_hot = np.eye(config.cat_num)[label_encoding]
    labels_fortfd = tf.convert_to_tensor(labels_one_hot)

    # prepare images
    imgs_np = np.array(imginput, dtype=np.float32)
    images_prepped = []
    for im_np in imgs_np:
        im_prepped = preprocess_and_aug(image_np = im_np, arch_for_preprocess = arch, augflag = aug)
        images_prepped.append(im_prepped)

    # images_fortfd = tf.convert_to_tensor(images_prepped, dtype=tf.float32)

    # for loading in images into the tf dataset batches, generate them as they're being batched from tf rather than reading all at once, for efficiency. Do this with a generator.
    def image_generator(images_list, labels_list):
        for img, label in zip(images_list, labels_list):
            yield img, label

    dataset = tf.data.Dataset.from_generator(
        lambda: image_generator(images_prepped, labels_fortfd),
        output_signature=(
            tf.TensorSpec(shape=(config.imheight, config.imwidth, 3), dtype=tf.float32), 
            tf.TensorSpec(shape=(5), dtype=tf.int32)
        )
    )
    return dataset



def create_tf_datasets(tracker,
    arch_set = config.arch_set, 
    cat_num = config.cat_num, # remove
    BATCH_SIZE = config.BATCH_SIZE,
    augflag_use = config.aug):
    """

    This function:
    - Reads a CSV file specified by `tracker` containing image file paths and label metadata
    - Filters the data into training and validation subsets based on the `innerPhase` column
    - Loads and preprocesses images using `read_imgs_as_np_array` (includes cropping, resizing, and normalization)
    - Encodes class labels to one-hot format using a category-to-index dictionary
    - Preprocesses image using 'preprocess_and_aug' based on  architecture, and also augments if that flag is set
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

    # dict_catKey_indValue, dict_indKey_catValue = helper_fns_adhoc.cat_str_ind_dictmap() # REMOVE

    imgs_train, cats_train = read_imgs_as_np_array(train_images, train_labels)#, arch_for_preprocess = arch_set)

    imgs_val, cats_val = read_imgs_as_np_array(val_images, val_labels)#, arch_for_preprocess = arch_set)

    # print("inspect images and their values")
    # print(imgs_train[0])
    # print(imgs_train[0][0])
    # print(imgs_train[0][0][0])

    dataset_train = prepare_tf_dataset(imgs_train, cats_train, arch = arch_set, aug = augflag_use)
    dataset_val = prepare_tf_dataset(imgs_val, cats_val, arch = arch_set, aug = augflag_use)

    # # here 1
    # label_encoding_train = [dict_catKey_indValue[catname] for catname in cats_train]
    # label_encoding_val = [dict_catKey_indValue[catname] for catname in cats_val]
    # print("label encoding done")

    # train_labels_one_hot = np.eye(cat_num)[label_encoding_train]
    # val_labels_one_hot = np.eye(cat_num)[label_encoding_val]
    # print("one hot encoding done")

    # # here 2
    # # prep train images
    # imgs_train_np = np.array(imgs_train, dtype=np.float32)
    # labels_fortfd_train = tf.convert_to_tensor(train_labels_one_hot)

    # # prep val images
    # imgs_val_np = np.array(imgs_val, dtype=np.float32)
    # labels_fortfd_val = tf.convert_to_tensor(val_labels_one_hot)

    # # preprocess and augment -- train images
    # images_prepped_train = []
    # for im_np in imgs_train_np:
    #     im_prepped = preprocess_and_aug(image_np = im_np, arch_for_preprocess = arch_set, augflag = augflag_use)
    #     images_prepped_train.append(im_prepped)
    # images_fortfd_train = tf.convert_to_tensor(images_prepped_train, dtype=tf.float32)
    # # print("an example to show aug - switch aug on and off to see this one image print differently with the two run")
    # # print(images_prepped_train[0][0])

    # # preprocess and augment -- val images
    # images_prepped_val = []
    # for im_np in imgs_val_np:
    #     im_prepped = preprocess_and_aug(image_np = im_np, arch_for_preprocess = arch_set, augflag = False)# aug should always be false for validation
    #     images_prepped_val.append(im_prepped)
    # images_fortfd_val= tf.convert_to_tensor(images_prepped_val, dtype=tf.float32)

    # print("done preprocess_and_aug")

    print(
        "Through with data prepped, but prior to dataset creation from tensor slices, so if it slows down here then it's just the tf dataset creation that is taking time"
    )
    # dataset_train = tf.data.Dataset.from_tensor_slices(
    #     (images_fortfd_train, labels_fortfd_train)
    # )

    # dataset_val = tf.data.Dataset.from_tensor_slices(
    #     (images_fortfd_val, labels_fortfd_val)
    # )

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


# information about this second tf datasets function:
# For a given run, evaluate each of the 30 CNNs on the full dataset; which is the same for all 30 models. Each model differed in terms of the data (folds) that was used for training and validation, but evaluation should be run on the full dataset (all folds), which is the same. Thus, to save memory and data loading time, load the full dataset just once, and then evaluate that same dataset on each of the 30 models, rather than loading the same data 30 times.
# Note: We don't need to load the full dataset during each model run, only need train and val for that (i.e. the above loading function), so don't waste resources by loading the full dataset for each run, just do it once when running evaluation, as done in the function below

def create_tf_datasets_for_evaluation(tracker,
    arch_set = config.arch_set,
    cat_num = config.cat_num,
    BATCH_SIZE = config.BATCH_SIZE,
    augflag_use = False):

    """

    Loads and preprocesses a FULL dataset for evaluation, returning a TensorFlow dataset along with original labels and image paths.

    This function:
    - Reads a CSV file specified by `tracker` containing image file paths and label metadata
    - Loads and preprocesses images using `read_imgs_as_np_array` (includes cropping, resizing, and normalization)
    - Encodes class labels to one-hot format using a category-to-index dictionary
    - Preprocesses image using 'preprocess_and_aug' based on  architecture, and also augments if that flag is set
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
    print(f"Total number of images for eval {len(df)}")

    # FOR TESTING! CHANGE THIS!!
    # df = df[0:2000]


    all_images = list(df["img_orig"])
    all_labels = list(df["img_cat"])


    dict_catKey_indValue, dict_indKey_catValue = helper_fns_adhoc.cat_str_ind_dictmap()

    imgs_all, cats_all = read_imgs_as_np_array(all_images, all_labels) #arch_for_preprocess = arch_set

    label_encoding_all = [dict_catKey_indValue[catname] for catname in cats_all]

    all_labels_one_hot = np.eye(cat_num)[label_encoding_all]

    # Ensure dtype is float32 for ims and int for labels
    # images_fortfd_all = np.array(imgs_all, dtype=np.float32)
    # labels_fortfd_all = np.array(all_labels_one_hot, dtype=np.int32)

    # images_fortfd_all = tf.convert_to_tensor(imgs_all, dtype=tf.float32)
    # labels_fortfd_all = tf.convert_to_tensor(all_labels_one_hot, dtype=tf.float32)

    # 6/10 adjust tensor way
    print("starting convert_to_tensor")
    imgs_all_np = np.array(imgs_all, dtype=np.float32)
    
    print("t1")
    labels_fortfd_all = tf.convert_to_tensor(all_labels_one_hot)
    print("t2")

    print("starting preprocess_and_aug")

    images_prepped_all = []
    for im_np in imgs_all_np:
        im_prepped = preprocess_and_aug(image_np = im_np, arch_for_preprocess = arch_set, augflag = augflag_use)
        images_prepped_all.append(im_prepped)

    print("though image preparation for arch")
    print(images_prepped_all[0][0])

    print('types')
    print(type(images_prepped_all))
    print(len(images_prepped_all))
    print(type(images_prepped_all[0]))
    print(type(images_prepped_all[0][0]))
    print(images_prepped_all[0])

    print(type(labels_fortfd_all))
    print(len(labels_fortfd_all))
    print(labels_fortfd_all[0])
    print(type(labels_fortfd_all[0]))
    # print(type(images_prepped_all[0][0]))

    # for loading in 22k images, need to generate them as they're being batched from tf rather than reading all 22k in at once. Do this with a generator.
    def image_generator(images_list, labels_list):
        for img, label in zip(images_list, labels_list):
            yield img, label

    dataset_all = tf.data.Dataset.from_generator(
        lambda: image_generator(images_prepped_all, labels_fortfd_all),
        output_signature=(
            tf.TensorSpec(shape=(config.imheight, config.imwidth, 3), dtype=tf.float32), 
            tf.TensorSpec(shape=(5), dtype=tf.int32)
        )
    )

    # BEGIN OLD WAY
    # # print(type())
    # print("starting conversion to tensor")
    # print(type(images_prepped_all))
    # images_fortfd_all = tf.convert_to_tensor(np.array(images_prepped_all), dtype=tf.float32)
    # print("through conversion to tf dataset")

    # # print("an example to show aug - switch aug on and off to see this one image print differently with the two run")
    # # # print(images_prepped_all[0][0])


    # print(
    #     "time lag check: data prepped before making it to tensor slices, see if it slows down here and if so then it's just the tf dataset creation that is taking time"
    # )

    # # try immediately converting from list
    # print("try immediately converting to dataset rather than converting to tensor first")
    # dataset_all = tf.data.Dataset.from_tensor_slices(
    #     (images_fortfd_all, labels_fortfd_all)
    # )

    # print("through the time lag part")
    # END OLD WAY

    # Count the number of elements (images) in the dataset
    num_images = sum(1 for _ in dataset_all)
    print(f"Number of images in the dataset for evaluation: {num_images}")


    dataset_all = dataset_all.batch(BATCH_SIZE)  # Batch the data
    dataset_all = dataset_all.prefetch(tf.data.AUTOTUNE)  # Optimize pipeline

    for x, y in dataset_all.take(1):
        print("X shape:", x.shape)
        print("Y shape:", y.shape)

    print(type(dataset_all))
    # print(all_labels[0:4])
    # print(all_images[0:4])
    print("through data loading of full dataset for evaluation")


    # this code also returns the image names, which are needed for pairing with the corresponding preds, and merging into the specific tracker for when saving out evaluation csvs
    return dataset_all, all_labels, all_images
