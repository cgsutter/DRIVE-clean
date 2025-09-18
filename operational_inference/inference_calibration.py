# Portions of this code were writen with the assistance of AI tools (e.g., ChatGPT)

import sys

sys.path.append("/home/csutter/DRIVE-clean/calibration/scripts")  

import calib
# import config
import os
import pandas as pd
import joblib
import numpy as np


model_nums = ["m0","m1","m2","m3","m4"] #

classif_model = "downstream"  #"CNN" or "downstream" or "fcstOnly" # HERE!!

if classif_model == "CNN":
    dir_of_uncalib_preds = "/home/csutter/DRIVE-clean/operational_inference/data_2_cnnpreds"
    datafiles = [f"{dir_of_uncalib_preds}/cnn_{i}.csv" for i in model_nums]
    print(datafiles)

    dir_of_calib_models = "/home/csutter/DRIVE-clean/operational_inference/trainedModels_2_calib_cnn"
    modelfiles = [f"{dir_of_calib_models}/calib_{i}.pkl" for i in model_nums]
    print(modelfiles)

    saveto_noext = f"/home/csutter/DRIVE-clean/operational_inference/data_3_cnncalib/cnncalib_"

elif classif_model == "downstream":
    dir_of_uncalib_preds = "/home/csutter/DRIVE-clean/operational_inference/data_4_downstream"
    datafiles = [f"{dir_of_uncalib_preds}/downstream_{i}.csv" for i in model_nums]
    print(datafiles)

    dir_of_calib_models = "/home/csutter/DRIVE-clean/operational_inference/trainedModels_4_calib_downstream"
    modelfiles = [f"{dir_of_calib_models}/calib_{i}.pkl" for i in model_nums]
    print(modelfiles)

    saveto_noext = f"/home/csutter/DRIVE-clean/operational_inference/data_5_downstreamcalib/downstreamcalib_"

for i in range(0,len(model_nums)):
    # runname = f[:-4] # remove the .csv

    csv = datafiles[i]
    print(f"reading in uncalibrated predictions from {csv}")
    # read in data
    dfread = pd.read_csv(csv)

    # prep column names
    t_all = calib.rename_cols_for_calibration_consistency(
        dfinput=dfread, classification_model=classif_model
    )

    # print(t_all.columns)

    # add classifier col of 0s and 1s if model predicted that cat
    t_all["classifier_TF"] = t_all["img_cat"] == t_all["o_pred"]
    t_all["classifier_01"] = t_all["classifier_TF"].astype(int)

    # prep the data for training/eval
    X_eval_unshaped = np.array(t_all["o_prob"])
    X_eval = X_eval_unshaped.reshape(-1, 1)

    # load in calibration model
    m = modelfiles[i]
    print(f"loading model {m}")
    model = joblib.load(m)

    # transform call evaluates the isotonic model on X_eval data to get the calibrated probabilities
    evaldata_output_predProb = model.transform(X_eval) # Calibrate the probabilities
    # print(X_eval[0:3])

    # add columns with calibrated probs
    t_all["o_prob_calib"] = evaldata_output_predProb

    t_all[
        [
            "calib_prob_dry",
            "calib_prob_poor_viz",
            "calib_prob_snow",
            "calib_prob_snow_severe",
            "calib_prob_wet",
        ]
    ] = t_all.apply(calib.rowfn_normalize_remaining, axis=1, result_type="expand")


    # Look at all the final probability columns to find the highest one now, and parse out the string cat name from the column which was highest. This *should* be the same as the original model's pred, but theoretically there could be some borderline cases where the predicted highest orig probability was calibrated downward so that one of the other classes surpassed it. Should be rare if at all, but need to account for this case.

    t_all["calib_pred"] = (
        t_all[
            [
                "calib_prob_dry",
                "calib_prob_poor_viz",
                "calib_prob_snow",
                "calib_prob_snow_severe",
                "calib_prob_wet",
            ]
        ]
        .idxmax(axis=1)
        .str.replace("calib_prob_", "", regex=False)
    )

    # note that this has to be done after calib AND normalizing bc sometimes the highest prob can change
    t_all["calib_prob"] = t_all[
        [
            "calib_prob_dry",
            "calib_prob_poor_viz",
            "calib_prob_snow",
            "calib_prob_snow_severe",
            "calib_prob_wet",
        ]
    ].max(axis=1)

    # print(t_all.columns)
    # print(t_all[0:3])

    saveto = f"{saveto_noext}m{i}.csv"
    t_all.to_csv(saveto)
    print(saveto)