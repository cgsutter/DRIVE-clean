# Portions of this code were writen with the assistance of AI tools (e.g., ChatGPT)

import os
import pandas as pd

import joblib
import sys

sys.path.append("/home/csutter/DRIVE-clean/downstream/scripts")
sys.path.append("/home/csutter/DRIVE-clean/CNN/scripts")
import results_summaries
import helper_fns_adhoc
import downstream_model_train


model_nums = ["m0","m1","m2","m3","m4"]

dir_of_cnncalib_preds = "/home/csutter/DRIVE-clean/operational_inference/data_3_cnncalib"
datafiles = [f"{dir_of_cnncalib_preds}/cnncalib_{i}.csv" for i in model_nums]
print(datafiles)

dir_of_downstream_models = "/home/csutter/DRIVE-clean/operational_inference/trainedModels_3_downstream"
modelfiles = [f"{dir_of_downstream_models}/models/downstream_{i}.pkl" for i in model_nums]
scalerfiles = [f"{dir_of_downstream_models}/scaler/scaler_{i}.pkl" for i in model_nums]
print(modelfiles)
print(scalerfiles)

# Dir to save the output predicted downstream csv
downstream_preds_dir = "/home/csutter/DRIVE-clean/operational_inference/data_4_downstream"

# Where the hrrr data lives
hrrr_data_csv = "/home/csutter/DRIVE/weather_img_concatmodels/cnn_hrrr_fcsthr2/nestedcv_imgname_hrrrdata_fcsthr2.csv"

cols_from_cnn = [
    "innerPhase",
    "outerPhase",
    "img_name",
    "img_orig",
    "site",
    "img_cat",
    "foldnum",
    "timeofday",
    "timeofevent",
    "calib_prob_dry",
    "calib_prob_poor_viz",
    "calib_prob_snow",
    "calib_prob_snow_severe",
    "calib_prob_wet",
    "calib_prob",
]

features = [
    "t2m", 
    "r2", 
    "uavg", 
    "asnow",  
    "tp",  
    "tcc", 
    "calib_prob_dry",
    "calib_prob_poor_viz",
    "calib_prob_snow",
    "calib_prob_snow_severe",
    "calib_prob_wet",
]






for i in range(0, len(datafiles)):

    ### Grab file and model paths
    csv = datafiles[i]
    downstream_model = modelfiles[i]
    scaler_model = scalerfiles[i]

    #### Load data
    cnndata = pd.read_csv(csv)

    # load in weather data
    hrrr = downstream_model_train.hrrr_data_load_prep(hrrr_data_csv)

    # merge data
    cnn_and_weather_data = downstream_model_train.merge_cnn_and_weather_data(
        cnndata_df=cnndata,
        weatherdata_df=hrrr,
        cols_to_keep_cnn=cols_from_cnn,
    )

    # Prepped data length
    print(len(cnn_and_weather_data))

    all_input_data = cnn_and_weather_data[features]
    all_output_data = cnn_and_weather_data["img_cat"] # need?
    all_imgname = cnn_and_weather_data["img_name"] # need?

    ### Load and run models

    # Scaler model
    scalerload = joblib.load(scaler_model)
    all_input_scaled = scalerload.transform(all_input_data)

    # Downstream model
    model = joblib.load(downstream_model)
    y_proball = model.predict_proba(all_input_scaled)
    y_predall = model.predict(all_input_scaled)
    ytrueall = all_output_data

    # Clean up predictions, add as cols to df, save out as csv
    print("check lengths of output")

    print(len(cnn_and_weather_data))
    print(len(y_proball))
    print(len(y_predall))
    print(len(ytrueall))

    dict_catKey_indValue, dict_indKey_catValue = (
        helper_fns_adhoc.cat_str_ind_dictmap()
    )

    df_results = pd.DataFrame(
        y_proball,
        columns=[
            f"ds_prob_{dict_indKey_catValue[i]}"
            for i in range(len(dict_indKey_catValue))
        ],
    )

    df_results["ds_pred"] = y_predall

    df_results["ds_prob"] = df_results[
        [
            "ds_prob_dry",
            "ds_prob_poor_viz",
            "ds_prob_snow",
            "ds_prob_snow_severe",
            "ds_prob_wet",
        ]
    ].max(axis=1)

    df_final = pd.concat([cnn_and_weather_data, df_results], axis=1)

    print(len(df_final))

    # print results - quick check, if there is something wrong in the ordering of predictions such that merging it with the full dataframe doesn't match, the results will be poor, and can check that with quick accuracy check below.
    correcty = sum(df_final["ds_pred"] == df_final["img_cat"])
    print(correcty / len(df_final))

    df_final.to_csv(f"{downstream_preds_dir}/downstream_m{i}.csv")

