# Portions of this code were writen with the assistance of AI tools (e.g., ChatGPT).

# Process used for training and validation datasets can be done with loading all images to memory (eagerly), and that way has it's benefits because we can track things like number of corrupt images, etc. But for evaluation where all 22k images need to be read in (compared to 7k for training, 3k for validation) -- these methods are bottlenecks. Need to make use of TF's parallelization and loading images in batches. The tracking of broken images (which is 0, but helpful for confirmation that all data is loaded) is not used in this code, only in the training and validation one, skipping that here in this code. 

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


def grab_img_paths_labels(tracker, phase= ""):


    ### STEP 1: LOAD IN DATA (TF DATASETS) and PREP WEIGHTS
    df = pd.read_csv(f"{tracker}")

    print(f"using {tracker}")
        
    # create variable and list that are sometimes used as an input in class weights function

    if phase != "":
        print("filtering by phase")
        df = df[df[f"innerPhase"] == phase]
    
    # grab catcount information 
    dfcatcount = (
        df[["img_cat", "img_name"]]
        .groupby(["img_cat"])
        .size()
        .reset_index(name="counts")
        .sort_values(["img_cat"])
    )
    catcounts = dfcatcount[
        "counts"
    ]  # this is a list used to calc what weights should be given the amount that are in each cat

    # print(catcounts)

    images_list = list(df["img_orig"])
    labels_list = list(df["img_cat"])
    images_names = list(df["img_name"])

    # encode the label
    dict_catKey_indValue, dict_indKey_catValue = helper_fns_adhoc.cat_str_ind_dictmap()
    label_encoding = [dict_catKey_indValue[catname] for catname in labels_list]

    return images_list, label_encoding, catcounts, images_names # labels_list



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
    image_tensor = tf.cast(image_tensor, tf.float32)
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
        # Assuming image_tensor is a tf.Tensor here
        # tf.print("Inside preprocess_and_aug, image_tensor shape before resnet_preprocess:", tf.shape(image_tensor))
        image_array = resnet_preprocess(image_tensor)
        # tf.print("Inside preprocess_and_aug, image_array shape after resnet_preprocess:", tf.shape(image_array))
    elif arch_for_preprocess == "vgg16":
        image_array = vgg_preprocess(image_tensor)
    elif arch_for_preprocess == "xcep":
        image_array = xception_preprocess(image_tensor)
    
    # print("finsihed preprocessing for arch")

    return image_array

def load_and_ready_image(image_path, arch_str, aug_bool):
    # NEED TO MAKE INPUTS TF STRINGS?
    # image_path will be a TensorFlow string tensor.
    # We need to convert it to a Python string to use with cv2.
    image_path_format = image_path.numpy().decode('utf-8')
    arch_str_format = arch_str.numpy().decode('utf-8') # Decode string tensor
    aug_bool_format = aug_bool.numpy().item() # Convert boolean tensor to Python bool

    try: 
        image_array = cv2.imread(image_path_format, cv2.IMREAD_UNCHANGED)
        # print(image_array)
        # Convert BGR (OpenCV default) to RGB
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        # Crop 20% from the top)
        h, w, _ = image_array.shape
        crop_height = int(0.2 * h)
        image_array = image_array[crop_height:, :, :]

        image_array = cv2.resize(image_array, config.TARGET_SIZE)  # Resize
    except:
    
        print(f"Warning: Could not read image {image_path_format}. Returning dummy array.")
        image_array = np.zeros(config.TARGET_SIZE, dtype=np.float32)



    # images_pixel.append(image_array)
    # labels_imgs.append(listlabels[im_i])
    # print()

    # Assuming image_array is a tf.Tensor here
    # tf.print("Inside load_and_ready_image, image_array shape before preprocess_and_aug:", tf.shape(image_array))

    img_processed = preprocess_and_aug(image_np = image_array, arch_for_preprocess = arch_str_format, augflag = aug_bool_format) 

    # tf.print("Inside load_and_ready_image, img_processed shape after preprocess_and_aug:", tf.shape(img_processed))
    
    return img_processed.numpy() # return numpy object so that when called on in tf.py_function, which automatically converts it to a tf tensor, it won't already be a tf object (which can cause problems)


def tf_load_and_preprocess_with_label(image_path, label, arch_for_preprocess_tensor, augflag_tensor):
    # image_path and label are already tf.Tensor objects (from dataset elements)
    # arch_for_preprocess_tensor and augflag_tensor are passed in as explicit tf.Tensor constants (see map call below)

    processed_image = tf.py_function(
        func=load_and_ready_image,
        inp=[image_path, arch_for_preprocess_tensor, augflag_tensor], # Pass all four tf.Tensor inputs
        Tout=tf.float32 # Only the image (as a NumPy array) is returned by load_and_preprocess_image
    )
    processed_image.set_shape([config.imheight, config.imwidth, 3])



    processed_label = tf.cast(label, tf.int32) #tf.uint8
    processed_label = tf.one_hot(processed_label, depth=config.cat_num) 
    # Add this line
    # dict_catKey_indValue, dict_indKey_catValue = helper_fns_adhoc.cat_str_ind_dictmap()
    # label_encoding = [dict_catKey_indValue[catname] for catname in label]
    # label_one_hot = np.eye(config.cat_num)[label_encoding]
    # label_fortfd = tf.convert_to_tensor(label_one_hot)


    return processed_image, processed_label #processed_label

# --- Create the tf.data.Dataset pipeline ---

# Get the paths and labels

def load_data(trackerinput, phaseinput, archinput, auginput):


    # read in dataframe and grab lists of paths and labels

    all_evaluation_image_paths, all_evaluation_labels, catcounts, imgnames = grab_img_paths_labels(tracker = trackerinput, phase= phaseinput)
    
    numims = len(all_evaluation_image_paths)

    print(f"Found {len(all_evaluation_image_paths)} images with corresponding labels for evaluation.")

    # Create a dataset from image paths AND labels
    path_and_label_dataset = tf.data.Dataset.from_tensor_slices((all_evaluation_image_paths, all_evaluation_labels))

    # --- Define the specific arch and augflag values for THIS pipeline run ---
    # These must be tf.Tensor constants
    arch_tensor_constant = tf.constant(archinput, dtype=tf.string) # Example architecture
    augflag_tensor_constant = tf.constant(auginput, dtype=tf.bool) # For evaluation, typically False
    print("Augmentation set as: ")
    print(augflag_tensor_constant)

    # 4. Map the preprocessing function
    # Use a lambda to pass the specific arch and augflag constants to your function.
    mapped_dataset = path_and_label_dataset.map(
        lambda img_path, lbl: tf_load_and_preprocess_with_label(
            img_path, lbl, arch_tensor_constant, augflag_tensor_constant
        ),
        num_parallel_calls=tf.data.AUTOTUNE # Allows parallel data processing
    )


    if phaseinput == "innerTrain":
        print(f"Shuffling individual images for training...")
        # numims is the total number of images. A shuffle buffer size larger than
        # the number of examples ensures a perfect shuffle. For large datasets,
        # a buffer size around 1000-10000 or a percentage of numims is common.
        # Using numims is safe but consumes memory if numims is very large.
        # If numims is > 20,000, consider a smaller, fixed buffer like 10000.
        final_dataset = mapped_dataset.shuffle(buffer_size=numims)
    else:
        # For evaluation, we do NOT shuffle to maintain order for matching predictions
        print(f"Not shuffling images for evaluation (phase: {phaseinput}).")
        final_dataset = mapped_dataset # No shuffle applied

    # 5. Batch the images and labels
    # Apply batching *after* shuffling (for training) or no shuffling (for eval)
    final_dataset = final_dataset.batch(config.BATCH_SIZE, drop_remainder=False)

    # 6. Prefetch data to overlap data loading/preprocessing with model execution
    final_dataset = final_dataset.prefetch(tf.data.AUTOTUNE)

    print("got through data loading")
    print(f"Dataset element spec: {final_dataset.element_spec}")

    # print("\nInspecting first batch:")
    # for i, (images_batch, labels_batch) in enumerate(final_dataset.take(1)): # Take only 1 batch
    #     print(f"  Batch {i+1}:")
    #     print(f"    Images batch shape: {images_batch.shape}")
    #     print(f"    Labels batch shape: {labels_batch.shape}")
    #     print(f"    Images batch dtype: {images_batch.dtype}")
    #     print(f"    Labels batch dtype: {labels_batch.dtype}")
    #     print(f"    First 5 labels in batch: {labels_batch[:5].numpy()}")

    # returning labels, numims, and catcounts, because those are needed for calculating class weights off of training data. Convenient to return these metrics in this function (but besides calculating class weights in the call_model_train script, these aren't needed) 
    return final_dataset, imgnames, all_evaluation_labels, numims, catcounts
