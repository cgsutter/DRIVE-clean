import logging
import math
import os
from struct import unpack

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image  # For image verification
from PIL import ImageFile
from tensorflow import keras

import config

ImageFile.LOAD_TRUNCATED_IMAGES = True

def tf_process_image(img_path, label):
    """Loads, decodes, resizes, and normalizes an image using TensorFlow ops."""
    try:
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)  # Or tf.image.decode_png
        image = tf.image.resize(image, (224, 224))
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        return image, label, img_path  # Return image, label, AND path

    except tf.errors.InvalidArgumentError:  # Handle decoding errors
        print(f"Error decoding image: {img_path}")
        return (
            tf.zeros([224, 224, 3]),
            label,
            img_path,
        )  # Dummy image, keep label and path

    except Exception as e:  # Catch any other exceptions
        print(f"Error processing image {img_path}: {e}")
        return (
            tf.zeros([224, 224, 3]),
            label,
            img_path,
        )  # Dummy image, keep label and path


def prepare_dataset(image_paths, labels):

    print("HERE2!!!")
    print(type(image_paths))
    image_paths = list(image_paths)
    print(type(labels))

    print(labels[2:5])
    labels = tf.cast(labels, tf.int32)  # CRUCIAL: Cast to tf.int32 here!

    image_ds = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )  # Paths only listed ONCE

    dataset = image_ds.map(tf_process_image, num_parallel_calls=tf.data.AUTOTUNE)

    def one_hot_encode(image, label, path):  # Added path here
        label = tf.squeeze(label)  # Remove extra dimensions if present
        label = tf.one_hot(label, depth=6)  # num_classes must be defined
        return image, label, path  # Added path here

    dataset = dataset.map(one_hot_encode)
    dataset = dataset.batch(128)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    print("here3")
    for image, label, path in dataset.take(1):  # Take one element to inspect
        print("Image shape:", image.shape)
        print("Label shape:", label.shape)
        print("Label dtype:", label.dtype)  # Should be tf.int32
        print("Path shape:", path.shape)
        print("Path dtype:", path.dtype)  # Should be tf.strin

    return dataset


# add a function that returns a fixed dictionary of classes in case loading does it differently each time


def create_label_dict(labels=config.category_dirs):

    cats_alphabetical = sorted(labels)

    # make a list of values 0 through 5 for 6-cats. Will be dynamic if cat length changes
    cat_inds = [i for i in range(0, len(cats_alphabetical))]

    label_dict_keyIs_catNumber = dict(zip(cat_inds, cats_alphabetical))

    label_dict_keyIs_catString = dict(zip(cats_alphabetical, cat_inds))

    return label_dict_keyIs_catNumber, label_dict_keyIs_catString
