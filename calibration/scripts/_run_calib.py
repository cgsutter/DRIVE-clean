import calib
import config
import os
import pandas as pd

# This code is set up to train a calibration model, and evaluate on all predictions, where the data is organized in a csv. This code points to a directory (as designated in config), and this calibration process will be complete for each csv file of data within that directory. Each csv represents data that is used to train/eval the calibration. The csvs stem from previous classification model outputs, so they are the predictions from multiple splits and models. Each one require calibration (as a new calib model).

# Given the note above, there is no need to tie in specific tracker lists, as they would have been included in the csvs in the directory that is being pointed to

datafiles = os.listdir(config.dir_of_datacsvs)

# subset datafiles

if config.subset_files_torun:
    datafiles.sort()
    print(datafiles[0:4])
    datafiles = [x for x in datafiles if config.subset_string in x]
    print(f"subsetted datafiles on {config.subset_string}")
    print(len(datafiles))


for f in datafiles:
    # runname = f[:-4] # remove the .csv

    print(f"reading in {f}")
    csv = f"{config.dir_of_datacsvs}/{f}"

    # read in data
    dfread = pd.read_csv(csv)

    print(dfread.columns)

    # prep column names
    t_all = calib.rename_cols_for_calibration_consistency(dfinput = dfread, classification_model = config.classif_model)

    # add classifier col of 0s and 1s if model predicted that cat
    t_all["classifier_TF"] = (
        t_all["img_cat"] == t_all["o_pred"]
        )
    t_all["classifier_01"] = t_all["classifier_TF"].astype(int)

    # training calib model on validation data
    t_val = t_all[t_all["innerPhase"] == "innerVal"] 

    # where to save out the model
    modeldir = f"/home/csutter/DRIVE-clean/calibration/calib_{config.classif_model}_model"
    os.makedirs(modeldir, exist_ok=True)
    model_savename = f"{modeldir}/{f[:-4]}_trainedOnVal.pkl" 

    # where to save out calibrated data results
    datadir = f"/home/csutter/DRIVE-clean/calibration/calib_{config.classif_model}_data"
    os.makedirs(datadir, exist_ok=True)
    calib_tracker_savename = f"{datadir}/{f}"

    v1, t1 = calib.calibrate_and_normalize_all_cats_PredOnly(
        t_val,
        t_all,
        calib_model_type_input=config.calib_model_type,
        modelsavename=model_savename,
    )

    print(len(v1))
    print(len(t1))
    print(t1.columns)
    t1.to_csv(calib_tracker_savename)

