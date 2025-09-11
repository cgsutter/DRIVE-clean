# Portions of this code were writen with the assistance of AI tools (e.g., ChatGPT)

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


####### Data-related functions
# load and prep HRRR data
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


def merge_cnn_and_weather_data(cnndata_df, weatherdata_df, cols_to_keep_cnn):
    cnndata_df = cnndata_df[
        cols_to_keep_cnn
    ]  # HERE!! if using calibrated probabilities cols will need to be different 'prob_dry', 'prob_poor_viz', 'prob_snow', 'prob_snow_severe', 'prob_wet','model_pred'  OR 'norm_calib_dry', 'norm_calib_poor_viz','norm_calib_snow', 'norm_calib_snow_severe', 'norm_calib_wet', 'norm_calib_cat'
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


def prepare_data_fortraining(dfinput, features=config.features_for_training):
    ## prepare data
    print("size of data inside prepare_data_fortraining function")

    print(len(dfinput))
    # HERE!! Update if want to train on train or train on val (and if this is the case then also be sure to set val equal to test, accordingly)
    train = dfinput[
        dfinput["innerPhase"] == "innerTest"
    ]  # should say innerTest. But for side experiment it's alt innerTrain
    train_input_data = train[features]
    train_output_data = train["img_cat"]

    print(len(train_input_data))

    val = dfinput[dfinput["innerPhase"] == "innerVal"]  # flip 5/8/25
    val_input_data = val[features]
    val_output_data = val["img_cat"]

    # also need to return the full df data, which is used for the final model when we make predictions on all observations
    all_input_data = dfinput[features]
    all_output_data = dfinput["img_cat"]
    all_imgname = dfinput["img_name"]

    # transform data, normalize and standardize

    # standardize data based on training data, then apply that transformation on all data (dont make a different standardization for val, test, etc. )

    # Step 1: Initialize the scaler
    scaler = StandardScaler()

    # Step 2: Fit the scaler using only the training data (or whatever data will be used to train the model)
    scaler.fit(train_input_data)  # Learns mean and std from training data

    # Step 3: Transform each dataset using the same scaler
    train_input_scaled = scaler.transform(train_input_data)  # Applies learned mean/std
    val_input_scaled = scaler.transform(val_input_data)  # Applies learned mean/std
    # test_input_scaled = scaler.transform(test_input_data) # Uses same mean/std as training

    all_input_scaled = scaler.transform(all_input_data)

    return (
        train_input_scaled,
        train_output_data,
        val_input_scaled,
        val_output_data,
        all_input_scaled,
        all_output_data,
        all_imgname,
        scaler,
    )


# ML model-related functions
# define fn for dnn training
def dnn(
    hypsuse,
    traindata_input,
    traindata_output,
    valdata_input,
    valdata_output,
    makepreds=False,
    alldata_input=None,
    alldataoutput=None,
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

    # list1 = calcstats_onefold(ytrue, y_pred) #7/19 remove this
    # print(list1)

    # for model predictions for final run, run pred and probs on all examples
    if makepreds:
        y_proball = model.predict(alldata_input)
        y_predall_ind = np.argmax(y_proball, axis=1)
        y_predall = le.inverse_transform(y_predall_ind)
        ytrueall = le.inverse_transform(alldataoutput)

    else:
        y_proball = None
        y_predall = None
        ytrueall = None

    return ytrue, y_pred, model, y_proball, y_predall, ytrueall


def multinomial_logistic_reg(
    hypsuse,
    traindata_input,
    traindata_output,
    valdata_input,
    valdata_output,
    makepreds=False,
    alldata_input=None,
    alldataoutput=None,
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

    # for model predictions for final run, run pred and probs on all examples
    if makepreds:
        y_proball = model.predict_proba(alldata_input)
        y_predall = model.predict(alldata_input)
        ytrueall = alldataoutput

    else:
        y_proball = None
        y_predall = None
        ytrueall = None

    return valdata_output, y_pred, model, y_proball, y_predall, ytrueall


def svm_model(
    hypsuse,
    traindata_input,
    traindata_output,
    valdata_input,
    valdata_output,
    makepreds=False,
    alldata_input=None,
    alldataoutput=None,
):
    svm_classifier = svm.SVC(
        **hypsuse, class_weight="balanced", probability=True, verbose=1
    )

    # pick up here below, run for either method
    svm_classifier.fit(traindata_input, traindata_output)

    # Evaluate the classifier on ALL data and save out tracker

    # Evaluate the classifier on the validation data
    y_pred = svm_classifier.predict(valdata_input)

    # for model predictions for final run, run pred and probs on all examples
    if makepreds:
        y_predall = svm_classifier.predict(alldata_input)

        # full_prob = modelload.predict_proba(alli) # REMOVED! Don't use this method... Dont use predict_proba for SVM decision boundary model. It does something w platt scaling and essentially there will be some cases where the predicted class does not align with the maximum from predict proba. SOLUTION: Need to use  the raw scores decision boundary and manually calculate the softmax from that. The max from that will align with the .predict class.
        # manually calculate probabilities due predict_proba notes abovey_pred = svm_classifier.predict(valdata_input)
        raw_scores = svm_classifier.decision_function(
            alldata_input
        )  # Get raw SVM decision scores
        y_proball = softmax(raw_scores)  # Convert to probabilities
        ytrueall = alldataoutput

    else:
        y_proball = None
        y_predall = None
        ytrueall = None

    return valdata_output, y_pred, svm_classifier, y_proball, y_predall, ytrueall


def gnb_model(
    hypsuse,
    traindata_input,
    traindata_output,
    valdata_input,
    valdata_output,
    makepreds=False,
    alldata_input=None,
    alldataoutput=None,
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

    if makepreds:
        y_proball = nb.predict_proba(alldata_input)
        y_predall = nb.predict(alldata_input)
        ytrueall = alldataoutput

    else:
        y_proball = None
        y_predall = None
        ytrueall = None

    return valdata_output, y_pred, nb, y_proball, y_predall, ytrueall


def rf_model(
    hypsuse,
    traindata_input,
    traindata_output,
    valdata_input,
    valdata_output,
    makepreds=False,
    alldata_input=None,
    alldataoutput=None,
):

    model = RandomForestClassifier(**hypsuse, class_weight="balanced")

    model.fit(traindata_input, traindata_output)

    # predict on data used to train the model (which was val)
    y_pred = model.predict(valdata_input)

    if makepreds:
        y_proball = model.predict_proba(alldata_input)
        y_predall = model.predict(alldata_input)
        ytrueall = alldataoutput
    else:
        y_proball = None
        y_predall = None
        ytrueall = None

    return valdata_output, y_pred, model, y_proball, y_predall, ytrueall


#### Ad-hoc, grab the right set of hyperparams from config based on the set model
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


####### Evaluate model training functions
def run_training(
    alg,
    hypselected,
    train_input_scaled,
    train_output_data,
    val_input_scaled,
    val_output_data,
    makepreds_flag=False,
    alldata_input_use=None,
    alldata_output_use=None,
):

    if alg == "DNN":
        ytrue, y_pred, model, y_proball, y_predall, ytrueall = dnn(
            hypsuse=hypselected,
            traindata_input=train_input_scaled,
            traindata_output=train_output_data,
            valdata_input=val_input_scaled,
            valdata_output=val_output_data,
            makepreds=makepreds_flag,
            alldata_input=alldata_input_use,
            alldataoutput=alldata_output_use,
        )

    elif alg == "logistic":
        ytrue, y_pred, model, y_proball, y_predall, ytrueall = multinomial_logistic_reg(
            hypsuse=hypselected,
            traindata_input=train_input_scaled,
            traindata_output=train_output_data,
            valdata_input=val_input_scaled,
            valdata_output=val_output_data,
            makepreds=makepreds_flag,
            alldata_input=alldata_input_use,
            alldataoutput=alldata_output_use,
        )

    elif (
        alg == "svm"
    ):  # from here /home/csutter/DRIVE/weather_img_concatmodels/cnn_hrrr_fcsthr2/train_blended.py

        ytrue, y_pred, model, y_proball, y_predall, ytrueall = svm_model(
            hypsuse=hypselected,
            traindata_input=train_input_scaled,
            traindata_output=train_output_data,
            valdata_input=val_input_scaled,
            valdata_output=val_output_data,
            makepreds=makepreds_flag,
            alldata_input=alldata_input_use,
            alldataoutput=alldata_output_use,
        )

    elif alg == "gnb":

        ytrue, y_pred, model, y_proball, y_predall, ytrueall = gnb_model(
            hypsuse=hypselected,
            traindata_input=train_input_scaled,
            traindata_output=train_output_data,
            valdata_input=val_input_scaled,
            valdata_output=val_output_data,
            makepreds=makepreds_flag,
            alldata_input=alldata_input_use,
            alldataoutput=alldata_output_use,
        )

    elif alg == "rf":

        ytrue, y_pred, model, y_proball, y_predall, ytrueall = rf_model(
            hypsuse=hypselected,
            traindata_input=train_input_scaled,
            traindata_output=train_output_data,
            valdata_input=val_input_scaled,
            valdata_output=val_output_data,
            makepreds=makepreds_flag,
            alldata_input=alldata_input_use,
            alldataoutput=alldata_output_use,
        )

    else:
        print("issue with algorithm input")
        ytrue = "issue"
        ypred = "issue"

    return ytrue, y_pred, model, y_proball, y_predall, ytrueall


##### Results functions (for hyperparameter tuning logging performance)
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


def track_results_main_file(dict_results, mainresultsfile=config.file_collect_results):
    # save out to a main file that will collect all model results, even hyptuning

    file_exists = os.path.isfile(mainresultsfile)

    with open(mainresultsfile, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=dict_results.keys())
        if not file_exists:
            writer.writeheader()  # write header only once
        writer.writerow(dict_results)  # always write the data row
