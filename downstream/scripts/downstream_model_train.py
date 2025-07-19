import os
import pickle

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump, load
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import class_weight
from tensorflow.keras import regularizers

import config
import csv

## this script does not same models/trackers of results
## it does hyp tuning and saves one line of results summaries to a csv based on the one set at HERE!!

# define fn for dnn training


def dnn(hypsuse, traindata_input, traindata_output, valdata_input, valdata_output):

    weightsuse = class_weight.compute_class_weight(
        "balanced", classes=np.unique(traindata_output), y=traindata_output
    )
    class_weights_use = dict(enumerate(weightsuse))

    # run model

    # print(hypsuse)
    # print(hypsuse["hidden_units"])
    # print(type(hypsuse["hidden_units"]))

    hidden_units = eval(hypsuse["hidden_units"])

    model = Sequential()
    for i in range(hypsuse["hidden_layers"]):
        # print(i)
        if i == 0:
            model.add(
                Dense(
                    hidden_units[i],
                    input_shape=(traindata_input.shape[1],),
                    activation="relu",
                    kernel_regularizer=regularizers.l2(hypsuse["l2_reg"]),
                )
            )
        else:
            # print(hidden_units[i])
            model.add(
                Dense(
                    hidden_units[i],
                    activation="relu",
                    kernel_regularizer=regularizers.l2(hypsuse["l2_reg"]),
                )
            )
        model.add(Dropout(hypsuse["dropout"]))
    # if len(classes_to_exclude) == 0:
    #     model.add(Dense(6, activation="softmax"))
    # else:
    model.add(Dense(5, activation="softmax"))
    # print(model.summary())
    model.compile(
        loss="categorical_crossentropy",
        optimizer="sgd",
        metrics=["accuracy"],
    )

    # Encode labels
    le = LabelEncoder()
    traindata_output = le.fit_transform(traindata_output)
    valdata_output = le.transform(valdata_output)  # Use transform, not fit_transform

    # Convert labels to categorical format
    traindata_output = to_categorical(traindata_output)
    valdata_output_categorical = to_categorical(valdata_output)

    # Train model
    model.fit(
        traindata_input,
        traindata_output,
        epochs=30,
        batch_size=128,
        verbose=0,
        class_weight=class_weights_use,
    )

    # Evaluate accuracy
    accuracy = model.evaluate(valdata_input, valdata_output_categorical, verbose=0)[1]

    # Get predicted categories
    y_prob = model.predict(valdata_input)
    y_pred_ind = np.argmax(y_prob, axis=1)
    y_pred = le.inverse_transform(y_pred_ind)

    ytrue = le.inverse_transform(valdata_output)

    # list1 = calcstats_onefold(ytrue, y_pred) #7/19 remove this
    # print(list1)

    return ytrue, y_pred


def dnn_model_trainfinal(
    hypsuse, traindata_input, traindata_output, valdata_input, valdata_output
):

    weightsuse = class_weight.compute_class_weight(
        "balanced", classes=np.unique(traindata_output), y=traindata_output
    )
    class_weights_use = dict(enumerate(weightsuse))

    # run model

    # print(hypsuse)
    # print(hypsuse["hidden_units"])
    # print(type(hypsuse["hidden_units"]))

    hidden_units = eval(hypsuse["hidden_units"])

    model = Sequential()
    for i in range(hypsuse["hidden_layers"]):
        # print(i)
        if i == 0:
            model.add(
                Dense(
                    hidden_units[i],
                    input_shape=(traindata_input.shape[1],),
                    activation="relu",
                    kernel_regularizer=regularizers.l2(hypsuse["l2_reg"]),
                )
            )
        else:
            # print(hidden_units[i])
            model.add(
                Dense(
                    hidden_units[i],
                    activation="relu",
                    kernel_regularizer=regularizers.l2(hypsuse["l2_reg"]),
                )
            )
        model.add(Dropout(hypsuse["dropout"]))
    # if len(classes_to_exclude) == 0:
    #     model.add(Dense(6, activation="softmax"))
    # else:
    model.add(Dense(5, activation="softmax"))
    # print(model.summary())
    model.compile(
        loss="categorical_crossentropy",
        optimizer="sgd",
        metrics=["accuracy"],
    )

    # Encode labels
    le = LabelEncoder()
    traindata_output = le.fit_transform(traindata_output)
    valdata_output = le.transform(valdata_output)  # Use transform, not fit_transform

    # Convert labels to categorical format
    traindata_output = to_categorical(traindata_output)
    valdata_output_categorical = to_categorical(valdata_output)

    # Train model
    model.fit(
        traindata_input,
        traindata_output,
        epochs=30,
        batch_size=128,
        verbose=0,
        class_weight=class_weights_use,
    )

    # Evaluate accuracy
    accuracy = model.evaluate(valdata_input, valdata_output_categorical, verbose=0)[1]

    # Get predicted categories
    y_prob = model.predict(valdata_input)
    y_pred_ind = np.argmax(y_prob, axis=1)
    y_pred = le.inverse_transform(y_pred_ind)

    ytrue = le.inverse_transform(valdata_output)

    return y_pred, y_prob, model


def multinomial_logistic_reg(
    hypsuse, traindata_input, traindata_output, valdata_input, valdata_output
):

    model = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",  # solver for multinomial for multi class model
        max_iter=hypsuse["max_iter"],  # increase if convergence warning
        C=hypsuse["C"],  # regularization strength (can tune later)
        random_state=42,  # for reproducibility
    )

    model.fit(traindata_input, traindata_output)

    # Predict on validation data
    y_prob = model.predict_proba(valdata_input)
    y_pred = model.predict(valdata_input)

    # list1 = calcstats_onefold(valdata_output, y_pred)  # (ytrue, y_pred)

    return valdata_output, y_pred


def svm_model(
    hypsuse, traindata_input, traindata_output, valdata_input, valdata_output
):
    svm_classifier = svm.SVC(
        **hypsuse, class_weight="balanced", probability=True, verbose=1
    )

    # pick up here below, run for either method
    svm_classifier.fit(traindata_input, traindata_output)

    # Evaluate the classifier on ALL data and save out tracker

    # Evaluate the classifier on the validation data
    y_pred = svm_classifier.predict(valdata_input)
    # y_prob = svm_classifier.predict_proba(vali) # see /home/csutter/DRIVE/weather_img_concatmodels/cnn_hrrr_fcsthr2/nested_train_rf.py for how to actually get probs for svm model. Dont need it rn for hyptuning

    # list1 = calcstats_onefold(valdata_output, y_pred)  # (ytrue, y_pred)

    return valdata_output, y_pred


def gnb_model(
    hypsuse, traindata_input, traindata_output, valdata_input, valdata_output
):

    # from dnn, but for gnb need a weight by class not a dictionary, so use just weightsuse
    weightsuse = class_weight.compute_class_weight(
        "balanced", classes=np.unique(traindata_output), y=traindata_output
    )
    # to return dict with class names rather than ints, and then use that to make weights by sample
    weightsuse2 = [[np.unique(traindata_output)[i], weightsuse[i]] for i in range(0, 5)]
    print(weightsuse2)
    weightsuse3 = dict(weightsuse2)
    print(weightsuse3)
    # Assign weight to each sample based on its class
    sample_weights_use = np.array(
        [weightsuse3[class_label] for class_label in traindata_output]
    )
    print(sample_weights_use)

    nb = GaussianNB(**hypsuse)
    nb.fit(traindata_input, traindata_output, sample_weight=sample_weights_use)

    # Evaluate the classifier on the validation data
    y_pred = nb.predict(valdata_input)
    y_prob = nb.predict_proba(valdata_input)

    # list1 = calcstats_onefold(valdata_output, y_pred)  # (ytrue, y_pred)

    return valdata_output, y_pred


def rf_model(hypsuse, traindata_input, traindata_output, valdata_input, valdata_output):

    model = RandomForestClassifier(**hypsuse, class_weight="balanced")

    model.fit(traindata_input, traindata_output)

    # predict on data used to train the model (which was val)
    y_pred = model.predict(valdata_input)

    # list1 = calcstats_onefold(valdata_output, y_pred)  # (ytrue, y_pred)

    return valdata_output, y_pred


def rf_model_trainfinal(
    hypsuse, traindata_input, traindata_output, valdata_input, valdata_output
):

    # 5/8 add other class weighting to add more weight to PV
    weightsuse = class_weight.compute_class_weight(
        class_weight="balanced", classes=np.unique(traindata_output), y=traindata_output
    )

    print(weightsuse)
    print(weightsuse[2])
    # Convert to dict: {0: weight0, 1: weight1, ...}
    print("CLASS WEIGHTS")
    class_weights_use = {
        "dry": weightsuse[0],
        "poor_viz": weightsuse[1],
        "snow": weightsuse[2],
        "snow_severe": weightsuse[3],
        "wet": weightsuse[4],
    }
    print(class_weights_use)

    # if want to give PV class a bit of a boost (*1.5) or decrease (*.5)
    # class_weights_use["poor_viz"] *= 2  # 5/12
    # class_weights_use["wet"] *= 1.2 # 1.5 or 1.2 was best from 5/10
    # class_weights_use["dry"] *= .85 # 5/12
    print(class_weights_use)

    model = RandomForestClassifier(
        **hypsuse, class_weight=class_weights_use
    )  # class_weight="balanced"

    model.fit(traindata_input, traindata_output)

    # predict on data used to train the model (which was val)
    y_pred = model.predict(valdata_input)
    y_prob = model.predict_proba(valdata_input)
    # list1 = calcstats_onefold(valdata_output, y_pred)  # (ytrue, y_pred)

    return y_pred, y_prob, model


# define function that returns stats about performance
def calcstats_onefold(ytrueinput, ypredinput):
    dfforcalc = pd.DataFrame({"img_cat": ytrueinput, "pred": ypredinput})

    splitspecific_list = []

    dfforcalc["correct_flag"] = ytrueinput == ypredinput
    splitspecific_list.append(len(dfforcalc))
    splitspecific_list.append(sum(dfforcalc["correct_flag"]))

    # by class
    for c in [
        "snow_severe",
        "snow",
        "wet",
        "dry",
        "poor_viz",
        # "obs",
    ]:  # added obs. Note that this was not included in original merging method w hrrr RF + merge algo logic
        sub = dfforcalc[dfforcalc["img_cat"] == c]
        cat_total = len(sub)
        cat_correct = len(sub[sub["correct_flag"] == True])
        splitspecific_list.append(cat_total)
        splitspecific_list.append(cat_correct)

    print("end and printing splitspecific_list")
    print(splitspecific_list)
    return splitspecific_list


# define function that runs all the above, just have to call it at the end

# added 7/19- load and prep HRRR data
def hrrr_data_load_prep(hrrr_data_path_csv):

    hrrr = pd.read_csv(hrrr_data_path_csv)

    hrrr = hrrr.dropna()  # ADDED 1231 bc starting w FH9 had an issue

    def addimgnamecol(row):
        elem = row["img_orig"].rfind("/")
        name = row["img_orig"][elem + 1 :]
        return name

    hrrr["img_name"] = hrrr.apply(addimgnamecol, axis=1)
    hrrr = hrrr.rename(columns={"img_orig": "img_orig_hrrr"})

    print(len(hrrr))

    # add average wind
    uavg = np.sqrt(hrrr["u10"] ** 2 + hrrr["v10"] ** 2)
    hrrr["uavg"] = uavg

    # hrrr.columns

    return hrrr

def merge_cnn_and_weather_data(cnndata_df, weatherdata_df,cols_to_keep_cnn):
    cnndata_df = cnndata_df[cols_to_keep_cnn]  # HERE!! if using calibrated probabilities cols will need to be different 'prob_dry', 'prob_poor_viz', 'prob_snow', 'prob_snow_severe', 'prob_wet','model_pred'  OR 'norm_calib_dry', 'norm_calib_poor_viz','norm_calib_snow', 'norm_calib_snow_severe', 'norm_calib_wet', 'norm_calib_cat'
    # print(len(c1))

    # merge the datasets dealing with multiple of the same cols, etc

    # already in cnn df don't need to carry it over from hrrr dataset
    hrrr_full1 = weatherdata_df.drop(
        columns=["img_cat", "site"]
    )  # drop this bc want to use img_cat from the labeled dataset

    print("len cnn df")
    print(len(cnndata_df))

    # note that hrrr has more data than labeled dataset bc it used to have obstructions in it, and also was before relabeling. Must left merge into the labeled_dataset to use the right data
    hrrr_full = cnndata_df.merge(hrrr_full1, how="left", on="img_name")
    # print(len(hrrr_full))
    # print(hrrr_full.columns)

    print("len all data for training downstream")
    print(len(hrrr_full))
    return hrrr_full

def prepare_data_fortraining(dfinput, features = config.features_for_training):
    ## prepare data

    # HERE!! Update if want to train on train or train on val (and if this is the case then also be sure to set val equal to test, accordingly)
    train = dfinput[dfinput["innerPhase"] == "innerTest"]  # flip 5/8/25
    train_input_data = train[features]
    train_output_data = train["img_cat"]

    val = dfinput[dfinput["innerPhase"] == "innerVal"]  # flip 5/8/25
    val_input_data = val[features]
    val_output_data = val["img_cat"]

    # transform data, normalize and standardize

    # standardize data based on training data, then apply that transformation on all data (dont make a different standardization for val, test, etc. )

    # Step 1: Initialize the scaler
    scaler = StandardScaler()

    # Step 2: Fit the scaler using only the training data (or whatever data will be used to train the model)
    scaler.fit(train_input_data)  # Learns mean and std from training data

    # Step 3: Transform each dataset using the same scaler
    train_input_scaled = scaler.transform(
        train_input_data
    )  # Applies learned mean/std
    val_input_scaled = scaler.transform(val_input_data)  # Applies learned mean/std
    # test_input_scaled = scaler.transform(test_input_data) # Uses same mean/std as training

    return train_input_scaled, train_output_data, val_input_scaled, val_output_data

#7/19 look for HT way here!!
def grab_alg_hyps(alg):
    if alg == "DNN":
        hyperparams = config.dnn_HT

    elif alg == "logistic":
        hyperparams = config.logistic_HT

    elif alg == "svm":
        hyperparams = config.svm_HT
            
    elif alg == "gnb":
        
        hyperparams = config.gnb_HT
           
    elif alg == "rf":
        
        hyperparams = config.rf_HT
        
    else:
        print("issue with algorithm input")

    return hyperparams



def run_training(alg, hypselected,train_input_scaled, train_output_data,val_input_scaled,val_output_data):
    if alg == "DNN":
        ytrue, ypred = dnn(
            hypsuse=hypselected,
            traindata_input=train_input_scaled,
            traindata_output=train_output_data,
            valdata_input=val_input_scaled,
            valdata_output=val_output_data,
        )

    elif alg == "logistic":
        ytrue, ypred = multinomial_logistic_reg(
            hypsuse=hypselected,
            traindata_input=train_input_scaled,
            traindata_output=train_output_data,
            valdata_input=val_input_scaled,
            valdata_output=val_output_data,
        )

    elif (
        alg == "svm"
    ):  # from here /home/csutter/DRIVE/weather_img_concatmodels/cnn_hrrr_fcsthr2/train_blended.py
        
        ytrue, ypred = svm_model(
            hypsuse=hypselected,
            traindata_input=train_input_scaled,
            traindata_output=train_output_data,
            valdata_input=val_input_scaled,
            valdata_output=val_output_data,
        )
            
    elif alg == "gnb":
        
        ytrue, ypred = gnb_model(
            hypsuse=hypselected,
            traindata_input=train_input_scaled,
            traindata_output=train_output_data,
            valdata_input=val_input_scaled,
            valdata_output=val_output_data,
        )
           
    elif alg == "rf":
        
        ytrue, ypred = rf_model(
            hypsuse=hypselected,
            traindata_input=train_input_scaled,
            traindata_output=train_output_data,
            valdata_input=val_input_scaled,
            valdata_output=val_output_data,
        )
        
    else:
        print("issue with algorithm input")
        ytrue = "issue"
        ypred = "issue"
    
    return ytrue, ypred

def track_results_main_file(dict_results, mainresultsfile = config.file_collect_results):
    # save out to a main file that will collect all model results, even hyptuning

    file_exists = os.path.isfile(mainresultsfile)

    with open(mainresultsfile, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=dict_results.keys())
        if not file_exists:
            writer.writeheader() # write header only once
        writer.writerow(dict_results) # always write the data row

    
# ###################################
# def run_HT(alg):
#     """
#     alg: string of "DNN" or "logistic" or "svm" to say which hyperparams and model function to run

#     """
#     # load CNN output dataset

#     OTnum = [0, 1, 2, 3, 4, 5]
#     modelnum = [0, 1, 2, 3, 4]

#     directory = "/home/csutter/DRIVE/dot/model_calibration/nestedcv/_twotrain_CNN_calibPredOnly"  # HERE!!
#     # "/home/csutter/DRIVE/dot/model_calibration/nestedcv/_twotrain_CNN_calibPredOnly"
#     # "/home/csutter/DRIVE/dot/model_calibration/nestedcv/_final_CNN_calibPredOnly"
#     # "/home/csutter/DRIVE/dot/model_trackpaths_results"  if want to use raw cnn output or calibrated, will be a different dir
#     # "/home/csutter/DRIVE/dot/model_calibration/nestedcv/CNN"
#     csv_files = [
#         os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".csv")
#     ]

#     csv_files = sorted(csv_files)

#     alldescs = []

#     for i in OTnum:
#         # print(i)
#         for j in modelnum:
#             # print(j)
#             alldescs.append(f"OT{i}_m{j}")

#     print(len(alldescs))

#     # load in weather data
#     hrrr = pd.read_csv(
#         "/home/csutter/DRIVE/weather_img_concatmodels/cnn_hrrr_fcsthr2/nestedcv_imgname_hrrrdata_fcsthr2.csv"
#     )

#     hrrr = hrrr.dropna()  # ADDED 1231 bc starting w FH9 had an issue

#     def addimgnamecol(row):
#         elem = row["img_orig"].rfind("/")
#         name = row["img_orig"][elem + 1 :]
#         return name

#     hrrr["img_name"] = hrrr.apply(addimgnamecol, axis=1)
#     hrrr = hrrr.rename(columns={"img_orig": "img_orig_hrrr"})

#     print(len(hrrr))

#     # add average wind
#     uavg = np.sqrt(hrrr["u10"] ** 2 + hrrr["v10"] ** 2)
#     hrrr["uavg"] = uavg

#     hrrr.columns

#     # hyp_example = param_combinations[5]
#     # print(hyp_example)
#     # print(str(hyp_example))

#     # loop through each model (30) and each hyp tune (100 for dnn)

#     # will end up looping through to run for each model, but for now just grab one to work with

#     # hrrr is already loaded as a df, no need to load it multiple times

#     # loop through and run for all 30 models x all hyperparameters and save out results by appending to df in directory

#     for mi in range(0, 30):  # range(0,30):
#         desc = alldescs[mi]
#         cnnfile = [x for x in csv_files if desc in x][
#             0
#         ]  # grab the file from the list of files by finding the one that contains teh desc string
#         # print(cnnfile)
#         c1 = pd.read_csv(cnnfile)
#         # print(c1.columns)

#         c1 = c1[
#             [
#                 "innerPhase",
#                 "outerPhase",
#                 "img_name",
#                 "img_orig",
#                 "site",
#                 "img_cat",
#                 "foldnum",
#                 "timeofday",
#                 "timeofevent",
#                 "norm_calib_dry",
#                 "norm_calib_poor_viz",
#                 "norm_calib_snow",
#                 "norm_calib_snow_severe",
#                 "norm_calib_wet",
#                 "norm_calib_pred",  # used to be norm_calib_cat for some older versions
#                 "norm_calib_prob",
#             ]
#         ]  # HERE!! if using calibrated probabilities cols will need to be different 'prob_dry', 'prob_poor_viz', 'prob_snow', 'prob_snow_severe', 'prob_wet','model_pred'  OR 'norm_calib_dry', 'norm_calib_poor_viz','norm_calib_snow', 'norm_calib_snow_severe', 'norm_calib_wet', 'norm_calib_cat'
#         # print(len(c1))

#         # merge the datasets dealing with multiple of the same cols, etc

#         # already in cnn df don't need to carry it over from hrrr dataset
#         hrrr_full1 = hrrr.drop(
#             columns=["img_cat", "site"]
#         )  # drop this bc want to use img_cat from the labeled dataset

#         # print(len(c1))

#         # note that hrrr has more data than labeled dataset bc it used to have obstructions in it, and also was before relabeling. Must left merge into the labeled_dataset to use the right data
#         hrrr_full = c1.merge(hrrr_full1, how="left", on="img_name")
#         # print(len(hrrr_full))
#         # print(hrrr_full.columns)

#         ## prepare data

#         features = [
#             "t2m",  # X
#             "r2",  # X
#             "uavg",  # added 1/4 calculated above
#             "asnow",  # X
#             "tp",  # X
#             "tcc",  # 5/10
#             # "dswrf", #5/10,
#             # "dlwrf", #5/10,
#             # "orog", #5/10,
#             # "mslma", #5/10,
#             # "si10", #5/10,
#             # "sh2", #5/10,
#             # "d2m", #5/10,
#             # HERE!! Depending on which cols to use calib or not
#             # "prob_dry",
#             # "prob_poor_viz",
#             # "prob_snow",
#             # "prob_snow_severe",
#             # "prob_wet",
#             "norm_calib_dry",
#             "norm_calib_poor_viz",
#             "norm_calib_snow",
#             "norm_calib_snow_severe",
#             "norm_calib_wet",
#         ]

#         # HERE!! Update if want to train on train or train on val (and if this is the case then also be sure to set val equal to test, accordingly)
#         train = hrrr_full[hrrr_full["innerPhase"] == "innerTest"]  # flip 5/8/25
#         train_input_data = train[features]
#         train_output_data = train["img_cat"]

#         val = hrrr_full[hrrr_full["innerPhase"] == "innerVal"]  # flip 5/8/25
#         val_input_data = val[features]
#         val_output_data = val["img_cat"]

#         # not doing anything with testing rn
#         # test = hrrr_full[hrrr_full["innerPhase"]=="innerTest"]
#         # test_input_data = test[features]
#         # test_output_data = test["img_cat"]

#         # transform data, normalize and standardize

#         # standardize data based on training data, then apply that transformation on all data (dont make a different standardization for val, test, etc. )

#         # Step 1: Initialize the scaler
#         scaler = StandardScaler()

#         # Step 2: Fit the scaler using only the training data (or whatever data will be used to train the model)
#         scaler.fit(train_input_data)  # Learns mean and std from training data

#         # Step 3: Transform each dataset using the same scaler
#         train_input_scaled = scaler.transform(
#             train_input_data
#         )  # Applies learned mean/std
#         val_input_scaled = scaler.transform(val_input_data)  # Applies learned mean/std
#         # test_input_scaled = scaler.transform(test_input_data) # Uses same mean/std as training

#         if alg == "DNN":

#             hyps_path = "/home/csutter/DRIVE/dot/models_concatdata/nowcast/features12a/sitesplit/hypgrid_dnn.csv"
#             hypdf = pd.read_csv(hyps_path)
#             print(len(hypdf))

#             hypdf = hypdf.replace({np.nan: None})
#             print(len(hypdf))  # run them all for dnn not as many as rf
#             hypdf = hypdf.drop(columns=["index"])
#             # this param_combinations is normally where the long list  of hyperparams would be
#             param_combinations = hypdf.to_dict(orient="records")

#             for hypselected in param_combinations:
#                 statsy = dnn(
#                     hypsuse=hypselected,
#                     traindata_input=train_input_scaled,
#                     traindata_output=train_output_data,
#                     valdata_input=val_input_scaled,
#                     valdata_output=val_output_data,
#                 )

#                 statsy.append(hypselected)
#                 statsy.append(desc)

#                 # Example list to append (make sure this matches the number of columns in the existing CSV)
#                 # new_data = [0.88, 0.85, 150, 120]  # Replace with your actual data

#                 # Path to the existing CSV file for which we're appending new rows for every new model that's run
#                 csv_file_path = "/home/csutter/DRIVE/dot/models_streamline/HT/streamlined_dnn_trainOnVal_calibCNN_twotrain.csv"  # HERE!!
#                 # _calibCNN_twotrain
#                 # _calibCNN_nowind
#                 # Make sure this csv exists with column names. Make a new one for each new time we call run(), see others to grab col names

#                 # Convert the list to a DataFrame (no column names)
#                 df_new = pd.DataFrame(
#                     [statsy]
#                 )  # List is passed as a row in the DataFrame

#                 # Append the new data to the existing CSV file
#                 df_new.to_csv(csv_file_path, mode="a", header=False, index=False)
#         elif alg == "logistic":
#             param_combinations = [
#                 {"max_iter": max_iter, "C": C}
#                 for max_iter in [100, 300, 500]
#                 for C in [0.01, 0.1, 1.0, 10.0, 100.0]
#             ]
#             for hypselected in param_combinations:
#                 statsy = multinomial_logistic_reg(
#                     hypsuse=hypselected,
#                     traindata_input=train_input_scaled,
#                     traindata_output=train_output_data,
#                     valdata_input=val_input_scaled,
#                     valdata_output=val_output_data,
#                 )

#                 statsy.append(hypselected)
#                 statsy.append(desc)

#                 # Path to the existing CSV file for which we're appending new rows for every new model that's run
#                 csv_file_path = "/home/csutter/DRIVE/dot/models_streamline/HT/streamlined_logistic_trainOnVal_calibCNN_twotrain.csv"  # HERE!! Make sure this csv exists with column names. Make a new one for each new time we call run(), see others to grab col names
#                 # _calibCNN_twotrain
#                 # _calibCNN_nowind

#                 # Convert the list to a DataFrame (no column names)
#                 df_new = pd.DataFrame(
#                     [statsy]
#                 )  # List is passed as a row in the DataFrame

#                 # Append the new data to the existing CSV file
#                 df_new.to_csv(csv_file_path, mode="a", header=False, index=False)
#         elif (
#             alg == "svm"
#         ):  # from here /home/csutter/DRIVE/weather_img_concatmodels/cnn_hrrr_fcsthr2/train_blended.py
#             param_combinations = [
#                 {"kernel": "rbf", "C": 1, "gamma": "scale"},
#                 {"kernel": "rbf", "C": 1, "gamma": 1e-5},
#                 {"kernel": "rbf", "C": 1, "gamma": 1e-4},
#                 {"kernel": "rbf", "C": 1, "gamma": 1e-3},
#                 {"kernel": "rbf", "C": 1, "gamma": 1e-2},
#                 {"kernel": "rbf", "C": 1, "gamma": 0.1},
#                 {"kernel": "rbf", "C": 1, "gamma": 1},
#                 {"kernel": "rbf", "C": 1, "gamma": 10},
#                 {"kernel": "rbf", "C": 0.1, "gamma": "scale"},
#                 {"kernel": "rbf", "C": 0.1, "gamma": 1e-5},
#                 {"kernel": "rbf", "C": 0.1, "gamma": 1e-4},
#                 {"kernel": "rbf", "C": 0.1, "gamma": 1e-3},
#                 {"kernel": "rbf", "C": 0.1, "gamma": 1e-2},
#                 {"kernel": "rbf", "C": 0.1, "gamma": 0.1},
#                 {"kernel": "rbf", "C": 0.1, "gamma": 1},
#                 {"kernel": "rbf", "C": 0.1, "gamma": 10},
#                 {"kernel": "rbf", "C": 1e-2, "gamma": "scale"},
#                 {"kernel": "rbf", "C": 1e-2, "gamma": 1e-5},
#                 {"kernel": "rbf", "C": 1e-2, "gamma": 1e-4},
#                 {"kernel": "rbf", "C": 1e-2, "gamma": 1e-3},
#                 {"kernel": "rbf", "C": 1e-2, "gamma": 1e-2},
#                 {"kernel": "rbf", "C": 1e-2, "gamma": 0.1},
#                 {"kernel": "rbf", "C": 1e-2, "gamma": 1},
#                 {"kernel": "rbf", "C": 1e-2, "gamma": 10},
#                 {"kernel": "rbf", "C": 1e-3, "gamma": "scale"},
#                 {"kernel": "rbf", "C": 1e-3, "gamma": 1e-5},
#                 {"kernel": "rbf", "C": 1e-3, "gamma": 1e-4},
#                 {"kernel": "rbf", "C": 1e-3, "gamma": 1e-3},
#                 {"kernel": "rbf", "C": 1e-3, "gamma": 1e-2},
#                 {"kernel": "rbf", "C": 1e-3, "gamma": 0.1},
#                 {"kernel": "rbf", "C": 1e-3, "gamma": 1},
#                 {"kernel": "rbf", "C": 1e-3, "gamma": 10},
#                 {"kernel": "rbf", "C": 10, "gamma": "scale"},
#                 {"kernel": "rbf", "C": 10, "gamma": 1e-5},
#                 {"kernel": "rbf", "C": 10, "gamma": 1e-4},
#                 {"kernel": "rbf", "C": 10, "gamma": 1e-3},
#                 {"kernel": "rbf", "C": 10, "gamma": 1e-2},
#                 {"kernel": "rbf", "C": 10, "gamma": 0.1},
#                 {"kernel": "rbf", "C": 10, "gamma": 1},
#                 {"kernel": "rbf", "C": 10, "gamma": 10},
#                 {"kernel": "rbf", "C": 100, "gamma": "scale"},
#                 {"kernel": "rbf", "C": 100, "gamma": 1e-5},
#                 {"kernel": "rbf", "C": 100, "gamma": 1e-4},
#                 {"kernel": "rbf", "C": 100, "gamma": 1e-3},
#                 {"kernel": "rbf", "C": 100, "gamma": 1e-2},
#                 {"kernel": "rbf", "C": 100, "gamma": 0.1},
#                 {"kernel": "rbf", "C": 100, "gamma": 1},
#                 {"kernel": "rbf", "C": 100, "gamma": 10},
#             ]
#             for hypselected in param_combinations:
#                 statsy = svm_model(
#                     hypsuse=hypselected,
#                     traindata_input=train_input_scaled,
#                     traindata_output=train_output_data,
#                     valdata_input=val_input_scaled,
#                     valdata_output=val_output_data,
#                 )
#                 statsy.append(hypselected)
#                 statsy.append(desc)

#                 # Path to the existing CSV file for which we're appending new rows for every new model that's run
#                 csv_file_path = "/home/csutter/DRIVE/dot/models_streamline/HT/streamlined_svm_trainOnVal_calibCNN_twotrain.csv"  # HERE!! Make sure this csv exists with column names. Make a new one for each new time we call run(), see others to grab col names
#                 # _calibCNN_nowind
#                 # _calibCNN_twotrain

#                 # Convert the list to a DataFrame (no column names)
#                 df_new = pd.DataFrame(
#                     [statsy]
#                 )  # List is passed as a row in the DataFrame

#                 # Append the new data to the existing CSV file
#                 df_new.to_csv(csv_file_path, mode="a", header=False, index=False)
#         elif alg == "gnb":
#             param_combinations = [
#                 {"var_smoothing": 1e-12},
#                 {"var_smoothing": 1e-11},
#                 {"var_smoothing": 1e-10},
#                 {"var_smoothing": 1e-10},
#                 {"var_smoothing": 1e-8},
#                 {"var_smoothing": 1e-7},
#                 {"var_smoothing": 1e-6},
#             ]
#             for hypselected in param_combinations:
#                 statsy = gnb_model(
#                     hypsuse=hypselected,
#                     traindata_input=train_input_scaled,
#                     traindata_output=train_output_data,
#                     valdata_input=val_input_scaled,
#                     valdata_output=val_output_data,
#                 )
#                 statsy.append(hypselected)
#                 statsy.append(desc)

#                 # Path to the existing CSV file for which we're appending new rows for every new model that's run
#                 csv_file_path = "/home/csutter/DRIVE/dot/models_streamline/HT/streamlined_gnb_trainOnVal_calibCNN_twotrain.csv"  # HERE!! Make sure this csv exists with column names. Make a new one for each new time we call run(), see others to grab col names
#                 # _calibCNN_nowind
#                 # _calibCNN_twotrain

#                 # Convert the list to a DataFrame (no column names)
#                 df_new = pd.DataFrame(
#                     [statsy]
#                 )  # List is passed as a row in the DataFrame

#                 # Append the new data to the existing CSV file
#                 df_new.to_csv(csv_file_path, mode="a", header=False, index=False)
#         elif alg == "rf":
#             hyps_path = "/home/csutter/DRIVE/dot/models_streamline/HT/hypgrid_rf_11feat_grid288.csv"
#             hypdf = pd.read_csv(hyps_path)
#             print(len(hypdf))

#             # hypdf = hypdf.replace({np.nan: None})
#             print(len(hypdf))  # run them all for dnn not as many as rf
#             hypdf = hypdf.drop(
#                 columns=[col for col in hypdf.columns if "Unnamed" in col], axis=1
#             )
#             print(hypdf)

#             param_combinations = []

#             # for ds in [regular,nomaxdepth,nomaxsamples,nomaxdepth_nomaxsamples]:
#             for _, row in hypdf.iterrows():
#                 row_dict = {}

#                 # Skip max_depth if it's 999
#                 if row["max_depth"] != 999:
#                     row_dict["max_depth"] = int(
#                         row["max_depth"]
#                     )  # make sure it's int, not float

#                 # Skip max_samples if it's 999
#                 if row["max_samples"] != 999.00:
#                     row_dict["max_samples"] = row["max_samples"]

#                 # Include all other parameters
#                 for col in hypdf.columns:
#                     if col not in ["max_depth", "max_samples"]:
#                         row_dict[col] = row[col]

#                 param_combinations.append(row_dict)

#             for hypselected in param_combinations:
#                 statsy = rf_model(
#                     hypsuse=hypselected,
#                     traindata_input=train_input_scaled,
#                     traindata_output=train_output_data,
#                     valdata_input=val_input_scaled,
#                     valdata_output=val_output_data,
#                 )
#                 statsy.append(hypselected)
#                 statsy.append(desc)

#                 # Path to the existing CSV file for which we're appending new rows for every new model that's run
#                 csv_file_path = "/home/csutter/DRIVE/dot/models_streamline/HT/streamlined_rf_trainOnVal_calibCNN_twotrain.csv"  # HERE!! Make sure this csv exists with column names. Make a new one for each new time we call run(), see others to grab col names 5/9/25 here
#                 # _calibCNN_twotrain
#                 # _calibCNN_nowind

#                 # Convert the list to a DataFrame (no column names)
#                 df_new = pd.DataFrame(
#                     [statsy]
#                 )  # List is passed as a row in the DataFrame

#                 # Append the new data to the existing CSV file
#                 df_new.to_csv(csv_file_path, mode="a", header=False, index=False)
#         else:
#             print("issue with algorithm input")


# ###################### For training and saving out final models
# # find best models from HT: /home/csutter/DRIVE/weather_img_concatmodels/cnn_hrrr_fcsthr2/nested_skill_streamline_vs_blending.ipynb


# def run_final():

#     # load CNN output dataset

#     OTnum = [0, 1, 2, 3, 4, 5]
#     modelnum = [0, 1, 2, 3, 4]

#     directory = "/home/csutter/DRIVE/dot/model_calibration/nestedcv/shuffle_CNN"
#     # "/home/csutter/DRIVE/dot/model_calibration/nestedcv/_twotrain_CNN_calibPredOnly"
#     # "/home/csutter/DRIVE/dot/model_calibration/nestedcv/_final_CNN_calibPredOnly" 5/27 -- this was the dir listed
#     # 5/27 experiments: /halved_CNN/ or /halvedShuffle_CNN/ or /onetrain_fourfolds_CNN/ or /shuffle_CNN/
#     csv_files = [
#         os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".csv")
#     ]

#     csv_files = sorted(csv_files)

#     alldescs = []

#     for c in csv_files:
#         # print(i)
#         beg = c.rfind("_OT") + 1
#         des = c[beg : beg + 11]
#         alldescs.append(des)

#     print(len(alldescs))

#     # load in weather data
#     hrrr = pd.read_csv(
#         "/home/csutter/DRIVE/weather_img_concatmodels/cnn_hrrr_fcsthr2/nestedcv_imgname_hrrrdata_fcsthr2.csv"
#     )

#     hrrr = hrrr.dropna()  # ADDED 1231 bc starting w FH9 had an issue

#     def addimgnamecol(row):
#         elem = row["img_orig"].rfind("/")
#         name = row["img_orig"][elem + 1 :]
#         return name

#     hrrr["img_name"] = hrrr.apply(addimgnamecol, axis=1)
#     hrrr = hrrr.rename(columns={"img_orig": "img_orig_hrrr"})

#     print(len(hrrr))

#     # add average wind
#     uavg = np.sqrt(hrrr["u10"] ** 2 + hrrr["v10"] ** 2)
#     hrrr["uavg"] = uavg

#     hrrr.columns

#     # hyp_example = param_combinations[5]
#     # print(hyp_example)
#     # print(str(hyp_example))

#     # loop through each model (30) and each hyp tune (100 for dnn)

#     # will end up looping through to run for each model, but for now just grab one to work with

#     # hrrr is already loaded as a df, no need to load it multiple times

#     # loop through and run for all 30 models x all hyperparameters and save out results by appending to df in directory

#     for mi in range(0, 30):  # range(0,30):
#         desc = alldescs[mi]
#         cnnfile = [x for x in csv_files if desc in x][
#             0
#         ]  # grab the file from the list of files by finding the one that contains teh desc string
#         # print(cnnfile)
#         c1 = pd.read_csv(cnnfile)
#         print("check length here")
#         print(len(c1))
#         print(c1.columns)

#         c1 = c1[
#             [
#                 "innerPhase",
#                 "outerPhase",
#                 "img_name",
#                 "img_orig",
#                 "site",
#                 "img_cat",
#                 "foldnum",
#                 "timeofday",
#                 "timeofevent",
#                 "norm_calib_dry",
#                 "norm_calib_poor_viz",
#                 "norm_calib_snow",
#                 "norm_calib_snow_severe",
#                 "norm_calib_wet",
#                 "norm_calib_pred",  # used to be norm_calib_cat for some older versions
#                 "norm_calib_prob",
#             ]
#         ]  # HERE!! if using calibrated probabilities cols will need to be different 'prob_dry', 'prob_poor_viz', 'prob_snow', 'prob_snow_severe', 'prob_wet','model_pred'  OR 'norm_calib_dry', 'norm_calib_poor_viz','norm_calib_snow', 'norm_calib_snow_severe', 'norm_calib_wet', 'norm_calib_cat'
#         # print(len(c1))

#         # merge the datasets dealing with multiple of the same cols, etc

#         # already in cnn df don't need to carry it over from hrrr dataset
#         hrrr_full1 = hrrr.drop(
#             columns=["img_cat", "site"]
#         )  # drop this bc want to use img_cat from the labeled dataset

#         # print(len(c1))

#         # note that hrrr has more data than labeled dataset bc it used to have obstructions in it, and also was before relabeling. Must left merge into the labeled_dataset to use the right data
#         hrrr_full = c1.merge(hrrr_full1, how="left", on="img_name")
#         print("check length 2")
#         print(len(hrrr_full))
#         # print(hrrr_full.columns)

#         ## prepare data

#         features = [
#             "t2m",  # X
#             "r2",  # X
#             "uavg",  # added 1/4 calculated above
#             "asnow",  # X
#             "tp",  # X
#             "tcc",  # 5/10
#             # "dswrf", #5/10,
#             # "dlwrf", #5/10,
#             # "orog", #5/10,
#             # "mslma", #5/10,
#             # "si10", #5/10,
#             # "sh2", #5/10,
#             # "d2m", #5/10,
#             # HERE!! Depending on which cols to use calib or not
#             # "prob_dry",
#             # "prob_poor_viz",
#             # "prob_snow",
#             # "prob_snow_severe",
#             # "prob_wet",
#             "norm_calib_dry",
#             "norm_calib_poor_viz",
#             "norm_calib_snow",
#             "norm_calib_snow_severe",
#             "norm_calib_wet",
#         ]

#         # Update if want to train on train or train on val (and if this is the case then also be sure to set val equal to test, accordingly)

#         train = hrrr_full[
#             hrrr_full["innerPhase"]
#             == "innerTest"  # here!! important note for the onetrain_fourfolds experiment, this needs to be innerTrain, o/w for ALL other experiments default is innerTest!!
#         ]  # HERE!! innerTrain or innerVal
#         print("check length 3")
#         print(len(train))
#         train_input_data = train[features]
#         print(len(train_input_data))
#         train_output_data = train["img_cat"]

#         # for all data
#         all = hrrr_full
#         all_input_data = all[features]
#         all_output_data = all["img_cat"]

#         print("Important check training data amount")
#         print(len(train_input_data))

#         # transform data, normalize and standardize

#         # standardize data based on training data, then apply that transformation on all data (dont make a different standardization for val, test, etc. )

#         # Step 1: Initialize the scaler
#         scaler = StandardScaler()

#         # Step 2: Fit the scaler using only the training data (or whatever data will be used to train the model)
#         scaler.fit(train_input_data)  # Learns mean and std from training data

#         scalersave = (
#             f"/home/csutter/DRIVE/dot/models_streamline/shuffle/{desc}_scaler.pkl"
#         )
#         # f"/home/csutter/DRIVE/dot/models_streamline/_finalscaler_twotrain/{desc}_scaler.pkl"
#         # f"/home/csutter/DRIVE/dot/models_streamline/_finalscaler/{desc}_scaler.pkl"
#         # 5/27 experiments: /halved/ or /halvedShuffle/ or /onetrain_fourfolds/ or /shuffle/
#         joblib.dump(scaler, scalersave)  # HERE!!

#         # Step 3: Transform each dataset using the same scaler
#         train_input_scaled = scaler.transform(
#             train_input_data
#         )  # Applies learned mean/std
#         all_input_scaled = scaler.transform(all_input_data)  # Applies learned mean/std
#         # test_input_scaled = scaler.transform(test_input_data) # Uses same mean/std as training

#         # for debugging inference
#         # just for debugging if saving out scaled input data
#         # print("debugging!!!")
#         # print(type(all_input_scaled))
#         # print((all_input_scaled.shape))
#         # print(all_input_scaled[0:2])
#         # l_inputs = all_input_scaled.tolist()
#         # print(len(l_inputs))
#         # print(l_inputs[0:2])

#         # HERE!! Uncomment the algorithm and its hyps to use
#         ### for RF

#         # selected hyps for train on test (5/8/25 way for prper nested cv based on paper figure)
#         hypselected = {
#             "max_depth": 10,
#             "max_samples": 0.75,
#             "n_estimators": 100,
#             "max_features": 3,
#             "min_samples_leaf": 5,
#             "bootstrap": True,
#         }

#         # two train - final human-derived
#         # {'max_depth': 10, 'max_samples': 0.75, 'n_estimators': 100, 'max_features': 3, 'min_samples_leaf': 5, 'bootstrap': True}

#         # 3-1 split - final human-derived
#         # {'max_depth': 5, 'max_samples': 0.5, 'n_estimators': 300, 'max_features': 3, 'min_samples_leaf': 5, 'bootstrap': True}

#         y_pred, y_prob, model = rf_model_trainfinal(
#             hypsuse=hypselected,
#             traindata_input=train_input_scaled,
#             traindata_output=train_output_data,
#             valdata_input=all_input_scaled,
#             valdata_output=all_output_data,
#         )

#         ### for DNN
#         # hypselected = {'hidden_layers': 1, 'hidden_units': '[16]', 'dropout': 0.8, 'l2_reg': 0.0}
#         # {
#         #     "hidden_layers": 5,
#         #     "hidden_units": "[256, 128, 64, 32, 16]",
#         #     "dropout": 0.0,
#         #     "l2_reg": 0.001,
#         # }
#         # y_pred, y_prob, model = dnn_model_trainfinal(
#         #     hypsuse=hypselected,
#         #     traindata_input=train_input_scaled,
#         #     traindata_output=train_output_data,
#         #     valdata_input=all_input_scaled,
#         #     valdata_output=all_output_data,
#         # )

#         modelname = f"/home/csutter/DRIVE/dot/models_streamline/shuffle/{desc}_model.pkl"  # HERE!! update the last subdir which the files will be saved in
#         # f"/home/csutter/DRIVE/dot/models_streamline/_final_twotrain/{desc}_model.pkl"
#         # f"/home/csutter/DRIVE/dot/models_streamline/_final/{desc}_model.pkl"
#         # 5/27 experiments: /halved/ or /halvedShuffle/ or /onetrain_fourfolds/ or /shuffle/

#         dump(model, modelname)

#         all["finalprediction"] = y_pred
#         all["finalprob"] = [max(y_prob[i]) for i in range(0, len(y_prob))]
#         all["dry_finalprob"] = [y_prob[i][0] for i in range(0, len(y_prob))]

#         all["poor_viz_finalprob"] = [y_prob[i][1] for i in range(0, len(y_prob))]
#         all["snow_finalprob"] = [y_prob[i][2] for i in range(0, len(y_prob))]
#         all["snow_severe_finalprob"] = [y_prob[i][3] for i in range(0, len(y_prob))]
#         all["wet_finalprob"] = [y_prob[i][4] for i in range(0, len(y_prob))]

#         all["finalpredictionCorrect"] = all["finalprediction"] == all["img_cat"]
#         # Path to the existing CSV file

#         # all["scaledinputs"] = l_inputs # just for debugging if saving out scaled input data

#         print(all[10:20])
#         all.to_csv(
#             f"/home/csutter/DRIVE/dot/models_streamline/shuffle/{desc}_streamline.csv"  # HERE!! update the last subdir which the files will be saved in
#         )
#         # f"/home/csutter/DRIVE/dot/models_streamline/_final_twotrain/{desc}_streamline.csv"
#         # f"/home/csutter/DRIVE/dot/models_streamline/_final/{desc}_streamline.csv"
#         # 5/27 experiments: /halved/ or /halvedShuffle/ or /onetrain_fourfolds/ or /shuffle/


# def run_TOV_finetune():

#     # load CNN output dataset

#     OTnum = [0, 1, 2, 3, 4, 5]
#     modelnum = [0, 1, 2, 3, 4]

#     directory = "/home/csutter/DRIVE/dot/models_streamline/final_trainOnTrain"  # HERE!!
#     csv_files = [
#         os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".csv")
#     ]

#     csv_files = sorted(csv_files)

#     alldescs = []

#     for i in OTnum:
#         # print(i)
#         for j in modelnum:
#             # print(j)
#             alldescs.append(f"OT{i}_m{j}")

#     print(len(alldescs))

#     for mi in range(0, 30):  # range(0,30):
#         desc = alldescs[mi]
#         cnnfile = [x for x in csv_files if desc in x][
#             0
#         ]  # grab the file from the list of files by finding the one that contains teh desc string
#         # print(cnnfile)
#         c1 = pd.read_csv(cnnfile)
#         print(c1.columns)

#         c1 = c1[
#             [
#                 "innerPhase",
#                 "outerPhase",
#                 "img_name",
#                 "img_orig",
#                 "site",
#                 "img_cat",
#                 "foldnum",
#                 "timeofday",
#                 "timeofevent",
#                 "finalprediction",
#                 "finalprob",
#                 "dry_finalprob",
#                 "poor_viz_finalprob",
#                 "snow_finalprob",
#                 "snow_severe_finalprob",
#                 "wet_finalprob",
#             ]
#         ]  # HERE!!

#         ## prepare data

#         features = [
#             "dry_finalprob",
#             "poor_viz_finalprob",
#             "snow_finalprob",
#             "snow_severe_finalprob",
#             "wet_finalprob",
#         ]

#         # HERE!! Update if want to train on train or train on val (and if this is the case then also be sure to set val equal to test, accordingly)
#         train = c1[c1["innerPhase"] == "innerVal"]
#         train_input_data = train[features]
#         train_output_data = train["img_cat"]

#         # for all data
#         all = c1
#         all_input_data = all[features]
#         all_output_data = all["img_cat"]

#         # transform data, normalize and standardize

#         # standardize data based on training data, then apply that transformation on all data (dont make a different standardization for val, test, etc. )

#         # Step 1: Initialize the scaler
#         scaler = StandardScaler()

#         # Step 2: Fit the scaler using only the training data (or whatever data will be used to train the model)
#         scaler.fit(train_input_data)  # Learns mean and std from training data

#         # Step 3: Transform each dataset using the same scaler
#         train_input_scaled = scaler.transform(
#             train_input_data
#         )  # Applies learned mean/std
#         all_input_scaled = scaler.transform(all_input_data)  # Applies learned mean/std
#         # test_input_scaled = scaler.transform(test_input_data) # Uses same mean/std as training

#         # HERE!! Uncomment the algorithm and its hyps to use
#         # for RF
#         hypselected = {
#             "max_depth": 5,
#             "n_estimators": 100,
#             "max_features": 3,
#             "min_samples_leaf": 5,
#             "bootstrap": False,
#         }
#         y_pred, y_prob, model = rf_model_trainfinal(
#             hypsuse=hypselected,
#             traindata_input=train_input_scaled,
#             traindata_output=train_output_data,
#             valdata_input=all_input_scaled,
#             valdata_output=all_output_data,
#         )

#         # # for DNN
#         # hypselected = {
#         #     "hidden_layers": 5,
#         #     "hidden_units": "[256, 128, 64, 32, 16]",
#         #     "dropout": 0.0,
#         #     "l2_reg": 0.001,
#         # }
#         # y_pred, y_prob, model = dnn_model_trainfinal(
#         #     hypsuse=hypselected,
#         #     traindata_input=train_input_scaled,
#         #     traindata_output=train_output_data,
#         #     valdata_input=all_input_scaled,
#         #     valdata_output=all_output_data,
#         # )

#         modelname = f"/home/csutter/DRIVE/dot/models_TOV_finetune/base_streamline_RF_TOT/{desc}_model.pkl"  # HERE!! update the last subdir which the files will be saved in

#         dump(model, modelname)

#         all["finalprediction"] = y_pred
#         all["finalprob"] = [max(y_prob[i]) for i in range(0, len(y_prob))]
#         all["dry_finalprob"] = [y_prob[i][0] for i in range(0, len(y_prob))]

#         all["poor_viz_finalprob"] = [y_prob[i][1] for i in range(0, len(y_prob))]
#         all["snow_finalprob"] = [y_prob[i][2] for i in range(0, len(y_prob))]
#         all["snow_severe_finalprob"] = [y_prob[i][3] for i in range(0, len(y_prob))]
#         all["wet_finalprob"] = [y_prob[i][4] for i in range(0, len(y_prob))]

#         all["finalpredictionCorrect"] = all["finalprediction"] == all["img_cat"]
#         # Path to the existing CSV file

#         print(all[10:20])
#         all.to_csv(
#             f"/home/csutter/DRIVE/dot/models_TOV_finetune/base_streamline_RF_TOT/{desc}_streamline.csv"  # HERE!! update the last subdir which the files will be saved in
#         )


# ###################### TO call on which function to run
# # run_HT(alg="logistic") # for hyptuning
# run_final()  # for one (final) model run and eval on all data
# # run_TOV_finetune()
