# pandas
# config
import os

# UPDATE TO BETH IS WHEN IN __MAIN
# from src  import config as config
import _config as config
import helper_fns_adhoc

# tensor flow for data load and mobilenet preprocessing
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
# import trackers_to_run as trackers_to_run
from tensorflow import keras
from tensorflow.keras.applications.mobilenet import preprocess_input


# For a given run, evaluate each of the 30 CNNs on the full dataset; which is the same for all 30 models. Each model differed in terms of the data (folds) that was used for training and validation, but evaluation should be run on the full dataset (all folds), which is the same. Thus, to save memory and data loading time, load the full dataset just once, and then evaluate that same dataset on each of the 30 models, rather than loading the same data 30 times.


def evaluate(modeldir, dataset, imgnames, savepreds, trackerinput):
    
    print(modeldir)
    model = tf.keras.models.load_model(modeldir, compile=False)
    print("got through model load!!") # note: this is not set up right now for evid, which requires loading of the custom loss function to deserialize (and that requires class weights which are unique to each of the 30 datasets, come back to this..



    for x, y in dataset.take(1):
        print("X shape:", x.shape)
        print("Y shape:", y.shape)



    # results = model.evaluate(dataset) # metrics of model skill, like loss and acc
    p2 = model.predict(dataset)
    c2 = np.argmax(p2, axis=1)

    dict_catKey_indValue, dict_indKey_catValue= helper_fns_adhoc.cat_str_ind_dictmap()

    predicted_classname = [dict_indKey_catValue[i] for i in c2]

    print(len(predicted_classname))

    print(predicted_classname[0:5])

    # just a df of results will concat these COLS to the end of the tracker df

    # print("major check here")
    # print(np.unique(p2[0]))
    # print(len(p2[0]))
    # columns = [f"prob_{dict_indKey_catValue[i]}" for i in range(len(dict_indKey_catValue))]
    # print(np.unique(columns))

    df_results = pd.DataFrame(
        p2, columns=[f"prob_{dict_indKey_catValue[i]}" for i in range(len(dict_indKey_catValue))]
    )

    df_results["model_pred"] = predicted_classname
    df_results["img_orig"] = imgnames

    print(df_results[0:10])
    print(df_results.columns)
    print(df_results["img_orig"][0])

    # # connect to the unique tracker (unique for each of the 30 datasets)
    df_all = pd.read_csv(trackerinput)
    df_final = df_all.merge(df_results, how = "inner", on = "img_orig")

    # print(df_final[0:10])
    # print(df_final.columns)
    print(len(df_final))


    df_final.to_csv(savepreds)
