# Portions of this code were writen with the assistance of AI tools (e.g., ChatGPT).

import os

# UPDATE TO BETH IS WHEN IN __MAIN
# from src  import config as config
import _config as config
import helper_fns_adhoc
import model_build

# tensor flow for data load and mobilenet preprocessing
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
# import trackers_to_run as trackers_to_run
from tensorflow import keras
from tensorflow.keras.applications.mobilenet import preprocess_input



def evaluate(modeldir, dataset, imgnames, trackerinput, saveflag = False, saveto = "", run_arch = config.arch_set,run_trle = config.transfer_learning, run_ast = config.ast, run_dr = config.dr_set, run_l2 = config.l2_set ):
    """For a given run, evaluate each of the 30 CNNs on the full dataset; which is the same for all 30 models. Each model differed in terms of the data (folds) that was used for training and validation, but evaluation should be run on the full dataset (all folds), which is the same. Thus, to save memory and data loading time, load the full dataset just once, and then evaluate that same dataset on each of the 30 models, rather than loading the same data 30 times.

    Args:
        modeldir (str): path to one model
        dataset (tf dataset): data that is alrealy prepared for evaluation on model
        imgnames (list): list of image names, which is used for connecting preds to the full tracker that has all info by observation
        trackerinput (str): path to original tracker
        saveflag (bool, optional): whether to save prediction csvs out (usually only set to True for one-off models, don't want to save 30x csvs for each BL and hyperparam model). Defaults to False.
        saveto (str, optional): path to save csv to (if saveflag is True). Defaults to "".

    Returns:
        Pandas dataframe: model predictions for each observation
        String: tracker name
    """
    
    # print(modeldir)

    print(f"Recreate model architecture")

    model = model_build.model_baseline(
        # one_off = config.one_off,
        # hyp_run = config.hyp_run,
        evid = config.evid,
        num_classes = config.cat_num,
        input_shape = (config.imheight, config.imwidth, 3),
        arch = run_arch,
        transfer_learning = run_trle,
        ast = run_ast,
        dropout_rate = run_dr,
        l2weight =run_l2,
        activation_layer_def = config.activation_layer_def,
        activation_output_def = config.activation_output_def
        )

    print(f"Load model weights {modeldir}")

    # Find the path to the latest checkpoint file within that directory
    # tf.train.latest_checkpoint() will return the base name of the checkpoint (e.g., "best_weights")
    # prefixed with the directory path.
    latest_checkpoint_path = tf.train.latest_checkpoint(modeldir)
    
    #  Load the weights into the recreated model
    model.load_weights(latest_checkpoint_path)
    print(f"Weights loaded successfully from: {latest_checkpoint_path}")

    p2 = model.predict(dataset)
    c2 = np.argmax(p2, axis=1)

    # note: this is not set up right now for evid, which requires loading of the custom loss function to deserialize (and that requires class weights which are unique to each of the 30 datasets, come back to this..



    # model = tf.keras.models.load_model(modeldir, compile=False)
    # print("got through model load!!") 

    print("Starting evaluate() in results_predictions.py")

    p2 = model.predict(dataset)
    c2 = np.argmax(p2, axis=1)

    dict_catKey_indValue, dict_indKey_catValue= helper_fns_adhoc.cat_str_ind_dictmap()

    predicted_classname = [dict_indKey_catValue[i] for i in c2]

    df_results = pd.DataFrame(
        p2, columns=[f"prob_{dict_indKey_catValue[i]}" for i in range(len(dict_indKey_catValue))]
    )

    df_results["model_pred"] = predicted_classname
    df_results["img_name"] = imgnames

    print(df_results[0:10])
    print(df_results.columns)
    print(df_results["img_name"][0])

    # # connect to the unique tracker (unique for each of the 30 datasets)
    df_all = pd.read_csv(trackerinput)
    print(df_all.columns)
    df_final = df_all.merge(df_results, how = "inner", on = "img_name")

    print(len(df_final))

    tracker_ident = helper_fns_adhoc.tracker_differentiator(trackerpath = trackerinput)

    df_final["tracker"] = tracker_ident

    t_name = os.path.basename(trackerinput)[:-4]

    if saveflag:
        # predssaveto = f"{config.preds_path}/{t_name}_{rundetails}.csv"
        df_final.to_csv(saveto) # saving the preds df to csv for one-off runs    
        print("saved to")
        print(saveto)

    return df_final, t_name

