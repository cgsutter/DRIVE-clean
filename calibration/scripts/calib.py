import config as config
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import LabelBinarizer

# Setting up first to work for calibrating CNN. After getting that working, will make adjustments to also get it working for streamlined (prior to ensembling), and then also ensembling (to get final calibrated probabilities for the end user). Code should be set up to be re-usable between each of the three calibration steps. Maybe just col names in pred dfs will be different.


# Rather than needing to col on dynamic col names depending on the classification algo, which gets tedious for each of these steps require for calibration, instead, change col names temporarily for these calibration fns so that all of these fns work no matter what step of the modeling process it's being applied on. Only in the last step before saving csv do we then rename the column names st that are distinct for the saved out csv to keep all information from all modeling steps.
def rename_cols_for_calibration_consistency(dfinput, classification_model="CNN"):

    # use the same col names for all classif model steps
    renamedcols = [
        "o_prob_dry",
        "o_prob_poor_viz",
        "o_prob_snow",
        "o_prob_snow_severe",
        "o_prob_wet",
        "o_pred",
        "o_prob",
    ]

    if "CNN" in classification_model:  # updated to accommodate for experiment models
        origcols = [
            "prob_dry",
            "prob_poor_viz",
            "prob_snow",
            "prob_snow_severe",
            "prob_wet",
            "model_pred",
            "model_prob",
        ]
        dict_for_rename = dict(zip(origcols, renamedcols))
        dfoutput = dfinput.rename(columns=dict_for_rename)
    elif classification_model == "fcstOnly":
        # NEED TO FILL IN
        origcols = [
            "prob_dry",
            "prob_poor_viz",
            "prob_snow",
            "prob_snow_severe",
            "prob_wet",
            "model_pred",
            "model_prob",
        ]
        dict_for_rename = dict(zip(origcols, renamedcols))
        dfoutput = dfinput.rename(columns=dict_for_rename)
    elif "downstream" in classification_model:
        print("running downstream!")
        print(dfinput.columns)
        origcols = [
            "ds_prob_dry",
            "ds_prob_poor_viz",
            "ds_prob_snow",
            "ds_prob_snow_severe",
            "ds_prob_wet",
            "ds_pred",
            "ds_prob",
        ]
        dict_for_rename = dict(zip(origcols, renamedcols))
        dfoutput = dfinput.rename(columns=dict_for_rename)
    else:
        print("issue with classification_model input")

    return dfoutput


# Function that trains and evaluates a calibration model on the predicted class only


def prep_data_for_calib_model(dftrain, dfeval):

    colprob = "o_prob"
    colpred = "classifier_01"

    X_unshaped = np.array(dftrain[colprob])
    X = X_unshaped.reshape(-1, 1)

    y = np.array(dftrain[colpred])

    X_eval_unshaped = np.array(dfeval[colprob])
    X_eval = X_eval_unshaped.reshape(-1, 1)

    return X, y, X_eval


def build_eval_logistic_PredOnly(
    traindata_input,
    traindata_output,
    evaldata_input,
    modeltype="isotonic",
    model_savename="",
):
    """
    dftrain is the df of val data
    dfeval is the tdf est data
    cat: string, which category to subset and train logistic model on.
    returngs: list of calibrated probabilities for both the val data (which was trained on) and the test data (to see results of calib on another dataset)
    """
    print("entering FIRST pred only")

    if modeltype == "logistic":
        # Create and train a logistic regression model
        model = LogisticRegression()
        model.fit(traindata_input, traindata_output)
        # Make predictions. Note that just model.predict will output 1s and 0s, but what we really want is the probability of 1, which corresponds to the new calibrated probability.
        # return the probabilities of 1s
        traindata_output_predProb = model.predict_proba(traindata_input)[:, 1]
        # evaluate test dataset
        evaldata_output_predProb = model.predict_proba(evaldata_input)[:, 1]

        # Save the trained model to a file
        joblib.dump(model, model_savename)

    elif modeltype == "isotonic":
        # Apply Isotonic Regression to calibrate the probabilities
        iso_reg = IsotonicRegression(out_of_bounds="clip")
        print("inside isot")
        print(traindata_input[0:3])
        print(traindata_output[0:3])

        iso_reg.fit(
            traindata_input, traindata_output
        )  # Fit isotonic regression on the probabilities
        # Use the calibrated probabilities to make predictions
        traindata_output_predProb = iso_reg.transform(
            traindata_input
        )  # Calibrate the probabilities
        evaldata_output_predProb = iso_reg.transform(
            evaldata_input
        )  # Calibrate the probabilities
        joblib.dump(iso_reg, model_savename)

    else:
        "try a isotonic or logistic as modeltype"

    print(f"amount of data used in training is {len(traindata_output_predProb)}")

    return evaldata_output_predProb


# normalize the remaining 4 cats based on the new calibrated predicted cat so that they all still sum to 1. Column names differ depending on classification model used, so account for this


def rowfn_normalize_remaining(row):
    allcats = ["dry", "poor_viz", "snow", "snow_severe", "wet"]

    predcat = str(row[f"o_pred"])
    predprob = row[
        f"o_prob_calib"
    ]  ## this is the predicted class' calibrated prob which was added from the other predOnly function

    # list the cols to grab that need to be normalized, which are the ones that aren't calibrated bc they werent the max
    # print(predcat)
    cats_requiring_calibration = [cat for cat in allcats if cat != predcat]
    # print(cats_requiring_calibration)

    catprobs_NOTnormalized = []

    for c in cats_requiring_calibration:
        coltoconsider = f"o_prob_{c}"
        catprobs_NOTnormalized.append(row[coltoconsider])

    # print(type(catprobs_NOTnormalized))
    # print(catprobs_NOTnormalized)
    NOTnormalized_sum = sum(catprobs_NOTnormalized)
    target_sum = 1 - predprob  # 1 minus predicted cat

    if NOTnormalized_sum == 0:
        # if the 4 non predicted cats are all prob 0, then the normaliztion calculation scaling won't work bc dividing by 0. And the original (un calibrated) prob of the predicted would have been 100. But after calibration it may have changed, so to distribute the remaining leftover, just divide it evenly amongst the 4 non pred classes
        distrib = predprob / 4
        catprobs_normalized = [distrib, distrib, distrib, distrib]
    else:
        catprobs_normalized = [
            x * target_sum / NOTnormalized_sum for x in catprobs_NOTnormalized
        ]

    # make a dictionary of all the cats and their probabilities

    catprobs_final_dict = {}
    catprobs_final_dict[predcat] = predprob
    for i in range(0, 4):
        catprobs_final_dict[cats_requiring_calibration[i]] = catprobs_normalized[i]
    # re order so that the cats are alphabetical
    finaldict_ordered = dict(sorted(catprobs_final_dict.items()))

    return tuple(finaldict_ordered.values())


def calibrate_and_normalize_all_cats_PredOnly(
    dftotrain, dftoeval, calib_model_type_input, modelsavename
):

    print("inside calibrate_and_normalize_all_cats_PredOnly")
    print(len(dftoeval))

    # prep the data for training/eval
    X, y, X_eval = prep_data_for_calib_model(dftotrain, dftoeval)
    print(len(X_eval))

    o_prob_calib = build_eval_logistic_PredOnly(
        X, y, X_eval, modeltype=calib_model_type_input, model_savename=modelsavename
    )
    print(len(o_prob_calib))
    print(type(o_prob_calib))

    # add columns with calibrated probs
    # 7/18 WAS calib_prob
    dftoeval["o_prob_calib"] = o_prob_calib

    print("dftoeval")
    print(len(dftoeval))
    print(dftoeval.columns)

    # since we only calibrated the highest class (predicted class) we need to use that new probability for the highest class, and then set the remaining class probabilities st they sum to 1. Normalize them based on the new max prob for the pred class.
    # this row function applied to all rows to return final relevant cols

    dftoeval[
        [
            "calib_prob_dry",
            "calib_prob_poor_viz",
            "calib_prob_snow",
            "calib_prob_snow_severe",
            "calib_prob_wet",
        ]
    ] = dftoeval.apply(rowfn_normalize_remaining, axis=1, result_type="expand")

    print("dftoeval -- TWO")
    print(len(dftoeval))
    print(dftoeval.columns)

    # Look at all the final probability columns to find the highest one now, and parse out the string cat name from the column which was highest. This *should* be the same as the original model's pred, but theoretically there could be some borderline cases where the predicted highest orig probability was calibrated downward so that one of the other classes surpassed it. Should be rare if at all, but need to account for this case.

    dftoeval["calib_pred"] = (
        dftoeval[
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
    dftoeval["calib_prob"] = dftoeval[
        [
            "calib_prob_dry",
            "calib_prob_poor_viz",
            "calib_prob_snow",
            "calib_prob_snow_severe",
            "calib_prob_wet",
        ]
    ].max(axis=1)

    return dftotrain, dftoeval


# 7/17 NEED TO DO: make this cleaner, dont need all this stuff can just parse out model/file name from tracker

# l2use_desc = str(config.l2_weight).replace(".", "_")
# dropoutuse_desc = str(config.dropout_rate).replace(".", "_")
# batchuse_desc = str(config.batch_size_def).replace(".", "_")
# lrinituse_desc = str(config.lr_init).replace(".", "_")
# lrdecruse_desc = str(config.lr_decayrate).replace(".", "_")
# configdetails = f"arch_{config.arch}_l2{l2use_desc}_dr{dropoutuse_desc}_b{batchuse_desc}_lr{lrinituse_desc}_lrdecr{lrdecruse_desc}_e{config.epoch_def}_es{config.earlystop_patience}_emin{config.min_epochs_before_es}"

# b = t.rfind("/")
# runname = t[b + 1 : -4]

#### Calibrate CNN probabilities -- COMMENT OUT if want to run calibration for fcstOnly whihc is below. O/w can have both uncommented and both running
# note on how to calibrate cnns: although for the blended model we may just predict 5 classes, we still input all 6-class probabilities from the cnn into the blending model (see https://docs.google.com/presentation/d/11bQ3ZyP2vQF8i0vClw8KwakR65KyR6FcrTFbEmXoSl0/edit?slide=id.g2d93f546e4c_0_0#slide=id.g2d93f546e4c_0_0 ) Thus, we still need to do the calibration across all 6 classes.

# # define variables

# #1-load in csv

# t_path = f"/home/csutter/DRIVE/dot/model_trackpaths_results/shuffle/{runname}_{configdetails}.csv"  # here!!
# f"/home/csutter/DRIVE/dot/model_trackpaths_results/twotrain_evensplit/{runname}_{configdetails}.csv"
# f"/home/csutter/DRIVE/dot/model_trackpaths_results/{runname}_{configdetails}.csv"
# 5/27 different experiments CNN models: /halved/ or /halvedShuffle/ or /onetrain_fourfolds/ or /shuffle/

# dfread = pd.read_csv(t_path)

# 7/17 added
# t_all = rename_cols_for_calibration_consistency(dfinput = dfread, classification_model = "CNN")

# # NEED TO UPDATE
# model_savename = f"/home/csutter/DRIVE/dot/model_calibration/nestedcv/shuffle_CNN/{calib_model_type}_{runname}"  # here!!
# f"/home/csutter/DRIVE/dot/model_calibration/nestedcv/_twotrain_CNN_calibPredOnly/{calib_model_type}_{runname}"
# f"/home/csutter/DRIVE/dot/model_calibration/nestedcv/_final_CNN_calibPredOnly/{calib_model_type}_{runname}"
# 5/27 different experiments to save out calib: /halved_CNN/ or /halvedShuffle_CNN/ or /onetrain_fourfolds_CNN/ or /shuffle_CNN/

# add classifier col which is a list of 0s and 1s if the model predicted THAT CLASS. Has nothing to do whether the model got it right or wrong, just a matter of
# t_all["classifier_TF"] = (
#     t_all["img_cat"] == t_all["o_pred"]
# )
# t_all["classifier_01"] = t_all["classifier_TF"].astype(int)

# t_val = t_all[t_all["innerPhase"] == "innerVal"]

# only need to train the calibration of the validation data, o/w at the end just apply it to everything as a whole tracker


# if calibrate_predclass_ONLY:
#     v1, t1 = calibrate_and_normalize_all_cats_PredOnly(
#         t_val,
#         t_all,
#         calib_model_type_input=calib_model_type,
#         modelsavename=model_savename,
#     )
#
# calib_tracker_savename = f"/home/csutter/DRIVE/dot/model_calibration/nestedcv/shuffle_CNN/{calib_model_type}_{runname}_trainedOnVal.csv"  # HERE!!
# f"/home/csutter/DRIVE/dot/model_calibration/nestedcv/_twotrain_CNN_calibPredOnly/{calib_model_type}_{runname}_trainedOnVal.csv"
# f"/home/csutter/DRIVE/dot/model_calibration/nestedcv/_final_CNN_calibPredOnly/{calib_model_type}_{runname}_trainedOnVal.csv"
# 5/27 different experiments to save out calib: /halved_CNN/ or /halvedShuffle_CNN/ or /onetrain_fourfolds_CNN/ or /shuffle_CNN/


# 7/17 need to add: Something at end here to differentiate col names based on the classifications step, since streamlined them all to be the same to run through now....
# t1.to_csv(calib_tracker_savename)

# print(len(v1))
# print(len(t1))
