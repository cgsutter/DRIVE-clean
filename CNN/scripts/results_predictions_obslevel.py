# Portions of this code were writen with the assistance of AI tools (e.g., ChatGPT).

import _config as config
import helper_fns_adhoc
import model_build
import load_dataset
import pandas as pd

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf # need?
import os



def make_preds(run_tracker = config.trackers_list[0], tracker_rundetails = "", wandblog = "", run_arch = config.arch_set, run_trle = config.transfer_learning, run_ast = config.ast, run_l2 =  config.l2_set, run_dr = config.dr_set, run_aug = config.aug, saveflag = False, phaseuse = ""):

    """
    Loads a trained model and validation dataset, generates predictions, and returns a 
    merged DataFrame of results.

    Parameters:
        run_tracker (str): Path to the tracker CSV file defining the dataset.
        tracker_rundetails (str): Suffix for identifying the specific model run.
        wandblog (str): Placeholder for W&B logging (currently unused).
        run_arch (str): Model architecture to use.
        run_trle (bool): Whether transfer learning was used.
        run_ast (bool): Whether to include architecture-specific top layers.
        run_l2 (float): L2 regularization weight.
        run_dr (float): Dropout rate.
        run_aug (bool): Whether data augmentation was used (should be False for val data).
        saveflag (bool): Whether to save the output predictions to CSV.

    Returns:
        pd.DataFrame: Merged DataFrame containing original tracker data, model predictions, predicted class labels, and associated image names.
    """

    print(f"Running Evaluation {run_tracker} using architecture {run_arch}. Transfer learning {run_trle}, arch-specific top {run_ast}. Dropout is {run_dr} and l2 weight is {run_l2}.")

    #### load data

    tf_ds_val, val_imgnames, labels_val, numims_val, valcatcounts = load_dataset.load_data(trackerinput = run_tracker, phaseinput = phaseuse, archinput = run_arch, auginput = False) # should always be false for val data


    print("validation data")
    print(type(tf_ds_val))
    print(numims_val)
    print(valcatcounts)
    print(val_imgnames[0:4])


    #### read model

    # This is only unique to an experiment & hyperparams NOT tracker, could move this outside the function
    tracker_filebase = helper_fns_adhoc.prep_basefile_str(tracker_designated = run_tracker)

    modeldir_set = f"{config.model_path}/{tracker_filebase}_{tracker_rundetails}"

    print("through here")
    print(tracker_filebase)
    print(tracker_rundetails)
    print(f"Model to read in {modeldir_set}")

    print(f"Recreate model architecture")

    model = model_build.model_baseline(
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

    print(f"Load model weights {modeldir_set}")

    latest_checkpoint_path = tf.train.latest_checkpoint(modeldir_set)
    
    #  Load the weights into the recreated model
    model.load_weights(latest_checkpoint_path)
    print(f"Weights loaded successfully from: {latest_checkpoint_path}")

    print("PRINT before predict")

    #### make preds

    dataset_for_prediction = tf_ds_val.map(lambda x, y: x)
    print(dataset_for_prediction.element_spec)

    p2 = model.predict(dataset_for_prediction)
    
    print("PRINT after predict")
    c2 = np.argmax(p2, axis=1)

    print("Complete with evaluate() in results_predictions.py")

    # note: this is not set up right now for evid, which requires loading of the custom loss function to deserialize (and that requires class weights which are unique to each of the 30 datasets, come back to this..

    dict_catKey_indValue, dict_indKey_catValue= helper_fns_adhoc.cat_str_ind_dictmap()

    predicted_classname = [dict_indKey_catValue[i] for i in c2]

    df_results = pd.DataFrame(
        p2, columns=[f"prob_{dict_indKey_catValue[i]}" for i in range(len(dict_indKey_catValue))]
    )

    df_results["model_pred"] = predicted_classname
    df_results["img_name"] = val_imgnames

    print(df_results[0:10])
    print(df_results.columns)
    print(df_results["img_name"][0])

    # # connect to the unique tracker (unique for each of the 30 datasets)
    df_all = pd.read_csv(run_tracker)
    print(df_all.columns)
    df_final = df_all.merge(df_results, how = "inner", on = "img_name")

    print(len(df_final))

    tracker_ident = helper_fns_adhoc.tracker_differentiator(trackerpath = run_tracker)

    df_final["tracker"] = tracker_ident

    cats = config.category_dirs
    cats_alphabetical = sorted(cats)
    cols_for_prob_cats = [f"prob_{c}" for c in cats_alphabetical]
    print("preparing model_prob col in make_preds function" )
    if saveflag:
        # add col that includes model prob (max across the predicted classes)
        df_final["model_prob"] = df_final[cols_for_prob_cats].max(axis=1)
        # same naming as with saved models: {tracker_filebase}_{tracker_rundetails}
        predssaveto = f"{config.preds_path}/{tracker_filebase}_{tracker_rundetails}.csv"
        df_final.to_csv(predssaveto) # saving the preds df to csv for one-off runs    
        print("saved to")
        print(predssaveto)
    else:
        print("not saving preds")

    return df_final