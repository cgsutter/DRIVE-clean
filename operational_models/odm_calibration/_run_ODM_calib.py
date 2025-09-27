# Portions of this code were writen with the assistance of AI tools (e.g., ChatGPT)

import odm_calib
import os
import pandas as pd

# Remember for operational models, ODM CNNs are not retrained, so just take models and its preds (below) from original development dir. What does change about operations is that we are calibrating the CNN preds (which is what is in this script) and then ensembling on the calib preds. 

datapaths = ["/home/csutter/DRIVE-clean/ODM/data_preds/ynobs_A_split0__A_resnet_TRLETrue_ASTTrue_L20_1_DR0_2_E75_AugTrue.csv",
"/home/csutter/DRIVE-clean/ODM/data_preds/ynobs_A_split1__A_resnet_TRLETrue_ASTTrue_L20_1_DR0_2_E75_AugTrue.csv",
"/home/csutter/DRIVE-clean/ODM/data_preds/ynobs_B_split0__A_resnet_TRLETrue_ASTTrue_L20_1_DR0_2_E75_AugTrue.csv",
"/home/csutter/DRIVE-clean/ODM/data_preds/ynobs_B_split1__A_resnet_TRLETrue_ASTTrue_L20_1_DR0_2_E75_AugTrue.csv"]

datafiles = [os.path.basename(f) for f in datapaths]

for f in datafiles:
    # runname = f[:-4] # remove the .csv

    # basename = os.path.basename(f)
    # print(basename)

    csv = f"/home/csutter/DRIVE-clean/ODM/data_preds/{f}"
    print(f"reading in {csv}")
    # read in data
    dfread = pd.read_csv(csv)

    print(dfread.columns)

    # prep column names
    t_all = odm_calib.rename_cols_for_calibration_consistency(
        dfinput=dfread, classification_model="CNN"
    )

    print(t_all.columns)

    # add classifier col of 0s and 1s if model predicted that cat
    t_all["classifier_TF"] = t_all["img_cat"] == t_all["o_pred"]
    t_all["classifier_01"] = t_all["classifier_TF"].astype(int)

    # training calib model on validation data
    t_val = t_all[t_all["innerPhase"] == "innerVal"]

    # where to save out the model
    modeldir = f"/home/csutter/DRIVE-clean/operational_models/odm_calibration/data_models/calib_CNN_model"
    # important, note that this is a DIRECTORY being created with calib_CNN_model, for example
    os.makedirs(modeldir, exist_ok=True)
    model_savename = f"{modeldir}/{f[:-4]}_trainedOnVal.pkl"

    # where to save out calibrated data results
    datadir = f"/home/csutter/DRIVE-clean/operational_models/odm_calibration/data_preds/calib_CNN_data"
    # important, note that this is a DIRECTORY being created with calib_CNN_data, for example
    os.makedirs(datadir, exist_ok=True)
    calib_tracker_savename = f"{datadir}/{f}"

    v1, t1 = odm_calib.calibrate_and_normalize_all_cats_PredOnly(
        t_val,
        t_all,
        calib_model_type_input="isotonic",
        modelsavename=model_savename,
    )

    print(len(v1))
    print(len(t1))
    print(t1.columns)
    t1.to_csv(calib_tracker_savename)
