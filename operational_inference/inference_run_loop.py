# See inference_run.py for full code. This script uses that code, but set up to run through a loop of given dates, e.g. case study


import grab_time
import grab_imgs
import grab_hrrr
import inference_calibration
import inference_cnn
import inference_downstream
import inference_ensemble

import os
from datetime import datetime

import pandas as pd

datestorun = pd.read_csv("/home/csutter/DRIVE-clean/operational_runs/set3_20250924/dates.csv") #HERE!!

dates = list(datestorun["date"])

# TO DEFINE -- where all the data prediction dirs should be nested within
parentdir = "/home/csutter/DRIVE-clean/operational_runs/set3_20250924" #HERE!!
# "/home/csutter/DRIVE-clean/operational_inference" # for one-off runs, e.g. testing code

for i in dates:

    ####### Define time of run and prepare corresponding filenames

    START = datetime.now()

    # B: for providing a given time (for case studies/past dates). Should be in UTC (+4EDT, +5EST)
    y,m,d,ymd,hr,hr_int,min,min_int,dirstructure  = grab_time.time_given( timestampstring = i)

    ## Prep dir and file names for data (based on date info above)

    imgcsv_save = f"{parentdir}/data_1_images/{dirstructure}/step1_imgfiles.csv"
    hrrrcsv_save = f"{parentdir}/data_1b_hrrr/{dirstructure}/step1b_hrrr.csv"
    cnn_preds = f"{parentdir}/data_2_cnnpreds/{dirstructure}"
    cnncalib_preds = f"{parentdir}/data_3_cnncalib/{dirstructure}"
    downstream_preds = f"{parentdir}/data_4_downstream/{dirstructure}"
    downstreamcalib_preds = f"{parentdir}/data_5_downstreamcalib/{dirstructure}"
    final_ensemble_preds = f"{parentdir}/data_6_ensembling/{dirstructure}"
    odm_cnn_preds = f"{parentdir}/data_odm_1_cnnpreds/{dirstructure}"
    odm_final_ensemble_preds = f"{parentdir}/data_odm_2_ensembling/{dirstructure}"


    ####### Grab data for the instance run

    # Grab image data 
    grab_imgs.step1_fn(
        rundate=ymd,
        runhour=hr_int,  # this is an int. only need hour as int, dont need minute as int which is why we just have minute as string below
        saveimgcsv = imgcsv_save,
        y=y,
        m=m,
        d=d,
        hour_str=hr,
        min_str=min,
    )

    IMG = datetime.now()

    # Grab HRRR data
    grab_hrrr.runall(fhnum_input = 2, imgdatacsv = imgcsv_save, hrrrdatacsv_tosave = hrrrcsv_save)

    HRRR = datetime.now()


    ###### Surface condition model (SCM)

    # Set model info related to SCM
    model_nums = ["m0","m1","m2","m3","m4"] #
    cats = [
            "wet",
            "dry",
            "snow",
            "snow_severe",
            "poor_viz",
        ]

    # # ## Step A - CNN
    cnnmodels_dir = "/home/csutter/DRIVE-clean/operational_inference/trainedModels_1_cnn"
    inference_cnn.cnn_run(inference_run_tracker = imgcsv_save, model_nums = model_nums, dir_tosave_preds = cnn_preds,dir_of_models = cnnmodels_dir, catnum = 5, catlist = cats)

    CNN = datetime.now()

    # # ## Step B - CNN CALIB

    inference_calibration.calib_run(model_nums = model_nums, dir_of_uncalib_preds = cnn_preds, classif_model = "CNN", saveto_dir = cnncalib_preds)

    # ## Step C - DOWNSTREAM

    inference_downstream.downstream_run(model_nums = model_nums, dir_of_cnncalib_preds = cnncalib_preds, hrrrdatapath = hrrrcsv_save, downstream_preds_dir = downstream_preds)

    # # ## Step D - DOWNSTREAM CALIB

    inference_calibration.calib_run(model_nums = model_nums, dir_of_uncalib_preds = downstream_preds, classif_model = "downstream", saveto_dir = downstreamcalib_preds)

    # # ## Step E - ENSEMBLING

    inference_ensemble.ensemble_run(modeltype = "SCM", model_nums = model_nums, dir_modelpreds = downstreamcalib_preds,dir_save_finalpreds = final_ensemble_preds, catsuse= cats)

    FINISH = datetime.now()
    print(f"TIME CHECK DONE START: {START}")
    print(f"TIME CHECK DONE IMG: {IMG}")
    print(f"TIME CHECK DONE HRRR: {HRRR}")
    print(f"TIME CHECK DONE CNN: {CNN}")
    print(f"TIME CHECK DONE EVERYTHING ELSE: {FINISH}")


    ####### Obstruction detection model (ODM)

    model_nums_ODM = ["m0","m1","m2","m3"] #,"m4","m5"
    catslist_ODM = ["nonobs","obs"]
    odm_cnnmodels_dir = "/home/csutter/DRIVE-clean/operational_inference/trainedModels_odm_CNN"

    inference_cnn.cnn_run(inference_run_tracker = imgcsv_save, model_nums = model_nums_ODM, dir_tosave_preds = odm_cnn_preds, dir_of_models = odm_cnnmodels_dir, catnum = 2, catlist = catslist_ODM)

    inference_ensemble.ensemble_run(modeltype = "ODM", model_nums = model_nums_ODM, dir_modelpreds = odm_cnn_preds,dir_save_finalpreds = odm_final_ensemble_preds, catsuse = ["nonobs","obs"])