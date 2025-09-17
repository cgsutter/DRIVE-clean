# This code is adjusted from /home/csutter/DRIVE-clean/CNN/scripts/results_predictions_obslevel.py for use in inference

# Portions of this code were writen with the assistance of AI tools (e.g., ChatGPT).

import sys

sys.path.append("/home/csutter/DRIVE-clean/CNN/scripts")  # Replace with the actual path

import model_build
import load_dataset
import helper_fns_adhoc

import pandas as pd
import tensorflow as tf
import gc  # Import garbage collection module
import os


import cv2
import numpy as np


# Set model to use for inference
# Model details of final selected model from which to use
# Later, make this simpler by just calling m0 through m30. Have work to do anyway to adjust this to be like 5 models rather than 30, and probably remove the ensembling step in general.
# can always keep ensembling (w 5-fold rather than the 6 folds, dont need test anymore, just need val) and prove the value of ensembling with another dataset. It doesnt even need to be labeled, can just evaluate on new cases and do the statistics from cam work meeting to prove the validity of ensembling (which didnt rely on label/performance)

inference_run_tracker = (
    "/home/csutter/DRIVE-clean/operational_inference/data_1_images/example_small.csv"
)

dir_of_models = "/home/csutter/DRIVE-clean/operational_inference/trainedModels_1_cnn"
model_nums = ["m0","m1","m2"] #,"m3","m4"
model_paths = [f"{dir_of_models}/cnn_{i}" for i in model_nums]

print("HERE:A")
print(inference_run_tracker)
print(model_nums)
print(model_paths)

dir_tosave_preds = "/home/csutter/DRIVE-clean/operational_inference/data_2_cnnpreds"

print("HERE:B")
print(dir_tosave_preds)

inference_arch = "resnet"  # config.arch_set
inference_epoch = 75  # config.epoch_set
inference_l2 = 0.1  # config.l2_set
inference_dr = 0.2  # config.dr_set
inference_transferLearning = True  # config.transfer_learning
inference_ast = True  # config.ast
inference_evid = False  # config.evid
inference_cat_num = 5
inference_cats = [
    "wet",
    "dry",
    "snow",
    "snow_severe",
    "poor_viz",
]
inference_imheight = 224
inference_imwidth = 224
inference_aug = False
inference_activation_layer_def = "relu"
inference_activation_output_def = "softmax"

print("HERE:C")

#####################


def make_preds(
    run_tr_filename=inference_run_tracker,
    models_to_run=model_paths,
    tracker_rundetails="",
    wandblog="",
    run_arch=inference_arch,
    run_trle=inference_transferLearning,
    run_ast=inference_ast,
    run_l2=inference_l2,
    run_dr=inference_dr,
    run_aug=inference_aug,
    saveflag=True,
    phaseuse="",
    inference_otherdata="",
):  # check saveflag
    """
    Loads a trained model and validation dataset, generates predictions, and returns a
    merged DataFrame of results.

    Parameters:
        run_tracker (str): Path to the tracker CSV file defining the dataset.
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

    # print(f"Running Evaluation {run_tracker} using architecture {run_arch}. Transfer learning {run_trle}, arch-specific top {run_ast}. Dropout is {run_dr} and l2 weight is {run_l2}.")

    # For eventual merging of model preds to this tracker csv which has other observation-level meta data
    df_all = pd.read_csv(run_tr_filename)
    print(run_tr_filename)
    print(df_all.columns)

    #### load data
    # Note that the data loading code takes in a csv path and reads in the data in a format needed for model inference. (dont need to pass the loaded df)

    tf_ds_val, val_imgnames, labels_val, numims_val, valcatcounts = (
        load_dataset.load_data(
            trackerinput=run_tr_filename,
            phaseinput="",
            archinput=run_arch,
            auginput=run_aug,
        )
    )

    print("validation data")
    print(type(tf_ds_val))
    print(numims_val)
    print(valcatcounts)
    print(val_imgnames[0:4])

    #### prep model arch (same for each of the model in an ensemble)

    print(f"Recreate model architecture")

    model = model_build.model_baseline(
        evid=inference_evid,  # global
        num_classes=inference_cat_num,  # global
        input_shape=(inference_imheight, inference_imwidth, 3),  # global
        arch=inference_arch,
        transfer_learning=run_trle,
        ast=run_ast,
        dropout_rate=run_dr,
        l2weight=run_l2,
        activation_layer_def=inference_activation_layer_def,  # global
        activation_output_def=inference_activation_output_def,  # global
    )

    #### get preds for each model
    for mrun in models_to_run:

        print("beginning step in loop...")

        modelname = os.path.basename(mrun)
        print(f"Model name: {modelname}")
        print(f"Model to read in {mrun}")

        print(f"Load model weights {mrun}")

        latest_checkpoint_path = tf.train.latest_checkpoint(mrun)

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

        dict_catKey_indValue, dict_indKey_catValue = (
            helper_fns_adhoc.cat_str_ind_dictmap()
        )

        predicted_classname = [dict_indKey_catValue[i] for i in c2]

        df_results = pd.DataFrame(
            p2,
            columns=[
                f"prob_{dict_indKey_catValue[i]}"
                for i in range(len(dict_indKey_catValue))
            ],
        )

        df_results["model_pred"] = predicted_classname
        df_results["img_name"] = val_imgnames

        print(df_results[0:10])
        print(df_results.columns)
        print(df_results["img_name"][0])

        # df_all was loaded in the beginning
        # merge to get all meta data observation level
        print("Inside important check!!")
        print(len(df_all))
        print(len(df_results))
        df_final = df_all.merge(df_results, how="inner", on="img_name")

        print(len(df_final))

        df_final["tracker"] = modelname

        cats = inference_cats  # global
        cats_alphabetical = sorted(cats)
        cols_for_prob_cats = [f"prob_{c}" for c in cats_alphabetical]
        print("preparing model_prob col in make_preds function")

        df_final["model_prob"] = df_final[cols_for_prob_cats].max(axis=1)

        # running inference on other data so need to name predictions csvs accordingly
        # grab filename from the inference data, will use this prepended to save out accordingly
        beg = inference_otherdata.rfind("/")
        infdata_name = inference_otherdata[beg + 1 : -4]
        # remove csv
        predssaveto = f"{dir_tosave_preds}/{modelname}.csv"

        print(predssaveto)

        df_final.to_csv(predssaveto)  # saving the preds df to csv for one-off runs
        print("saved to")
        print(predssaveto)


make_preds()

# Note on tensorflow warnings: - may get some warnings about checkpoints and unrestored values at the end... these are safe to ignore if only doing inference (predicting). It only matters if we care about resuming the exact training state, o/w, can ignore this warning.
