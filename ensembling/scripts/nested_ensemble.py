import os
import sys

import numpy as np
import pandas as pd

# for each OTnum, grab the predictions from the 5 models, and then apply ensmebling techniques (trying more than one) and save out this df with all of those pieces.

# read in the dfs
# rename the cols since the 5 models will have 5 duplicate col names


def outsideTest_onedf_all5models(splitnum, parentdirofmodels):

    """
    splitnum: str, e.g. "2"
    """

    dir = parentdirofmodels

    files = os.listdir(dir)
    # print(files)

    print(f"_OT{splitnum}_")
    files2 = [f for f in files if f"OT{splitnum}_" in f]

    # print(files2)
    trackers_forOuterTest = [f for f in files2 if ".csv" in f]

    trackers_forOuterTest = sorted(trackers_forOuterTest)
    print("fileslist")
    print(len(trackers_forOuterTest))

    print(
        f"found {len(trackers_forOuterTest)} many relevant trackers for outer test split {splitnum}"
    )
    # print(trackers_forOuterTest)

    dfs_individ = []

    for t in trackers_forOuterTest:
        # grab tracker name from which we will parse out info we need, with the config, to grab the final nowcast results (which was created in the nestedcv_train_blending model)
        # p1 = trackers_forOuterTest[0]
        r = pd.read_csv(f"{dir}/{t}")

        # subset to just outerTest observations. O/w any observation that is kept in from say innerTest from OT0_m1 will be used as training or val in OT0_m2, so we can't really do this ensembling with anything besides OT

        r = r[r["outerPhase"] == "outerTest"].reset_index()
        # print(r.columns)
        print(len(r))

        r = r.drop(columns=["Unnamed: 0"])

        # grab info from file name which will be used to make new df
        b2 = t.rfind(f"_OT{splitnum}_")
        innermodelnum = t[b2 + 5 : b2 + 7]
        # used for renaming cols for each of the 5 inner models
        print(f"inner model number {innermodelnum}")

        desc_beg = t.rfind(f"OT")
        desc_end = desc_beg + 6
        desc = t[desc_beg:desc_end]

        # will read in all 5 results dfs for ensembling but all hve same col names so to work in a df (concatting all the columns) will want to rename the columns so not to have overlap
        cols_to_rename = [
            "innerPhase",  # must be renamed bc innerPhases are different for each of the 5 models inside this outerTest
            "calib_prob_dry",
            "calib_prob_poor_viz",
            "calib_prob_snow",
            "calib_prob_snow_severe",
            "calib_prob_wet",
            "calib_pred",
            "calib_prob", #finalprob
        ]

        r = r.rename(columns={col: f"{innermodelnum}_{col}" for col in cols_to_rename})

        # print(r.columns)

        r = r.sort_values(
            by="img_name"
        )  # not needed since using merge and keeping img_name column

        if innermodelnum != "m0":
            # drop all redundant identifyer cols except img_name
            # Since we will need to mergs all dfs, only keep the identifyer cols from the first df, and then merge all the rest of the dfs with just the model related columns.

            # only keep the columns needed for everything except the first
            r = r[
                [
                    "img_name",
                    f"{innermodelnum}_calib_prob_dry",
                    f"{innermodelnum}_calib_prob_poor_viz",
                    f"{innermodelnum}_calib_prob_snow",
                    f"{innermodelnum}_calib_prob_snow_severe",
                    f"{innermodelnum}_calib_prob_wet",
                    f"{innermodelnum}_calib_pred",
                    f"{innermodelnum}_calib_prob",
                ]
            ]
        # r = r.drop(columns = ["innerPhase"])
        dfs_individ.append(r)

    # print(dfs_individ)
    # print(type(dfs_individ))
    # print(type(dfs_individ[0]))
    # print(dfs_individ[0])
    print(len(dfs_individ))
    merged_df = dfs_individ[0]  # grab the first df which has all the identifyer cols
    for df in dfs_individ[1:]:  # add on by merging each df onto it, in a loop
        merged_df = merged_df.merge(df, on="img_name")

    print(len(merged_df))
    # for i in merged_df.columns:
    #     print(i)

    print("merged df here!")
    # print(merged_df.columns)
    # for c in merged_df.columns:
    #     print(c)
    return merged_df


def dffn_combine_cols_tolist(df, columns):
    """
    Make a new column that contains a list of all the 5 model's predictions
    """
    list_colvalues = df[columns].values.tolist()
    return list_colvalues


def reverse_dict(input_dict):
    reversed_dict = {}
    for key, value in input_dict.items():
        if value not in reversed_dict:
            reversed_dict[value] = [key]
        else:
            reversed_dict[value].append(key)
    return reversed_dict


def rowfn_dict_calcs_from_5preds(row):
    """
    A function to be applied to rows of a df.
    Returns arrays of dictionaries (which will be added as cols to the dataframe).
    For each observation, make two dictionaries. One dictionary with the class as key, and the list of models whose prediction was that class as the values. E.g. {'dry': ['m0', 'm4'], 'wet': ['m1', 'm2', 'm3']}. The second dictionary with the count of models who predicted that cat, e.g. {'dry': 2, 'wet': 3}. The third dict is the average probability value from the predicted classes, e.g. {'dry': 0.4718, 'wet': 0.5170}.
    """
    list_5preds = row["list_5preds"]
    dict_modelkeys_predcatvalues = {}
    for i in range(0, 5):
        # add key value pair
        dict_modelkeys_predcatvalues[f"m{i}"] = list_5preds[i]
    dict_catAsKeys_modelAsValues = reverse_dict(dict_modelkeys_predcatvalues)
    dict_catAsKeys_countAsValues = {}
    dict_catAsKeys_probsAsValues = {}
    # these two lists are for finding the most confident
    probs5 = []
    preds5 = []
    for predcat in dict_catAsKeys_modelAsValues.keys():
        models_w_predcat = dict_catAsKeys_modelAsValues[predcat]
        # add key value pair
        dict_catAsKeys_countAsValues[predcat] = len(models_w_predcat)
        models_w_predcat_probs = []
        for m in models_w_predcat:
            models_w_predcat_probs.append(row[f"{m}_calib_prob"])
            probs5.append(row[f"{m}_calib_prob"])
            preds5.append(row[f"{m}_calib_pred"])
        avg = np.mean(models_w_predcat_probs)
        # add key value pair
        dict_catAsKeys_probsAsValues[predcat] = avg

    # grab the 1 most confident prediction out of the 5 as a dictionary with single value
    dict_mostConfident_singleModel = {}
    for i in range(0, 5):
        if probs5[i] == max(probs5):
            # add key value pair
            dict_mostConfident_singleModel[f"{preds5[i]}"] = probs5[i]
    return (
        dict_catAsKeys_modelAsValues,
        dict_catAsKeys_countAsValues,
        dict_catAsKeys_probsAsValues,
        dict_mostConfident_singleModel,
    )


# Dicionary cols made in above function sets up for doing Method 1: mode, and method 3: max prob. See below:


def rowfn_grab_mode(row):
    """
    Apply row wise to a df
    Returns array of the mode prediction based on the dictionary column that has the counts for each pred cat in it
    """
    dict_catAsKeys_countAsValues = row["dict_catAsKeys_countAsValues"]

    max_key = max(dict_catAsKeys_countAsValues, key=dict_catAsKeys_countAsValues.get)

    max_value = max(dict_catAsKeys_countAsValues.values())
    max_keys = [
        key for key, value in dict_catAsKeys_countAsValues.items() if value == max_value
    ]

    if len(max_keys) > 1:
        # if "poor_viz" in max_keys:
        #     maxcat = "poor_viz"
        # elif "snow_severe" in max_keys:
        #     maxcat = "snow_severe"
        if "snow_severe" in max_keys:
            maxcat = "snow_severe"
        elif "poor_viz" in max_keys:
            maxcat = "poor_viz"
        elif "snow" in max_keys:
            maxcat = "snow"
        elif "wet" in max_keys:
            maxcat = "wet"
        elif "dry" in max_keys:
            maxcat = "dry"
        else:
            print("issue with max_keys")
    else:
        maxcat = max_keys[0]

    return maxcat


def rowfn_grab_max_confidence(row):

    maxconf = list(row["dict_mostConfident_singleModel"].keys())
    maxconf = maxconf[0]
    return maxconf


# Method 2: average probabilities
def dffn_return_avg_cols(dfinput):

    """Function on a df, return the updated df"""

    dcols = [
        "m0_calib_prob_dry",
        "m1_calib_prob_dry",
        "m2_calib_prob_dry",
        "m3_calib_prob_dry",
        "m4_calib_prob_dry",
    ]
    scols = [
        "m0_calib_prob_snow",
        "m1_calib_prob_snow",
        "m2_calib_prob_snow",
        "m3_calib_prob_snow",
        "m4_calib_prob_snow",
    ]
    sscols = [
        "m0_calib_prob_snow_severe",
        "m1_calib_prob_snow_severe",
        "m2_calib_prob_snow_severe",
        "m3_calib_prob_snow_severe",
        "m4_calib_prob_snow_severe",
    ]

    pvcols = [
        "m0_calib_prob_poor_viz",
        "m1_calib_prob_poor_viz",
        "m2_calib_prob_poor_viz",
        "m3_calib_prob_poor_viz",
        "m4_calib_prob_poor_viz",
    ]
    wcols = [
        "m0_calib_prob_wet",
        "m1_calib_prob_wet",
        "m2_calib_prob_wet",
        "m3_calib_prob_wet",
        "m4_calib_prob_wet",
    ]

    dfinput["ensembleAvg_dry"] = dfinput[dcols].mean(axis=1)
    dfinput["ensembleAvg_snow"] = dfinput[scols].mean(axis=1)
    dfinput["ensembleAvg_snow_severe"] = dfinput[sscols].mean(axis=1)
    dfinput["ensembleAvg_wet"] = dfinput[wcols].mean(axis=1)
    dfinput["ensembleAvg_poor_viz"] = dfinput[pvcols].mean(axis=1)
    # dfinput["ensembleAvg_obs"] = dfinput[ocols].mean(axis=1)

    print("through here before!!")

    # print(type(ensembleAvg_dry))
    # print(ensembleAvg_dry[0:4])

    # grab the column of probs that was the highest and parse out to just get the cat name from the col name that was max

    # make into a temporary df just to pull the max

    dfinput["ensembleAvg_pred"] = (
        dfinput[
            [
                "ensembleAvg_dry",
                "ensembleAvg_snow",
                "ensembleAvg_snow_severe",
                "ensembleAvg_wet",
                "ensembleAvg_poor_viz",
                # "ensembleAvg_obs",
            ]
        ]
        .idxmax(axis=1)
        .str[12:]
    )

    print("through here!!")

    # grab the max probability to retrun as predprob

    dfinput["ensembleAvg_predprob"] = dfinput[
        [
            "ensembleAvg_dry",
            "ensembleAvg_snow",
            "ensembleAvg_snow_severe",
            "ensembleAvg_wet",
            "ensembleAvg_poor_viz",
            # "ensembleAvg_obs",
        ]
    ].max(axis=1)

    return dfinput


# if 1 and 2 dont align, then look whether 3 aligns with either of them, and if it does, use that one. If it doesnt align, then we have 3 distinct model preds! Use the highest severity one


def rowfn_select_from_ensembles(row):
    a = row["ensembleAvg_pred"]
    m = row["ensembleMode_pred"]
    c = row["ensembleMaxConf_pred"]

    # if "snow_severe" in [a, m]:

    if a == m:
        select = a
        qualit = "align_avg_mode"
    elif a == c:
        select = a
        qualit = "align_avg_maxConf"
    elif m == c:
        select = m
        qualit = "align_mode_maxConf"
    else:  # it's a tie! Select the most severe
        qualit = "tie_use_most_severe"
        # if "poor_viz" in [a, m, c]: # PV priority
        #     select = "poor_viz"
        # elif "snow_severe" in [a, m, c]:
        #     select = "snow_severe"
        if "snow_severe" in [a, m, c]:  # PV priority
            select = "snow_severe"
        elif "poor_viz" in [a, m, c]:
            select = "poor_viz"
        elif "snow" in [a, m, c]:
            select = "snow"
        elif "wet" in [a, m, c]:
            select = "wet"
        elif "dry" in [a, m, c]:
            select = "dry"
    return select, qualit


# Add cols to categorize correct and oks


def dffn_addcols_correct_oks(dfinput):
    dfinput["select_correct"] = dfinput["select"] == dfinput["img_cat"]

    oks = []
    print("heref6")
    print(len(dfinput))
    for i in range(0, len(dfinput)):
        if dfinput.iloc[i]["select_correct"] == True:
            ok = True
        elif (dfinput.iloc[i]["img_cat"] == "snow_severe") & (
            dfinput.iloc[i]["select"] == "snow"
        ):
            ok = True
        elif (dfinput.iloc[i]["img_cat"] == "snow") & (
            (dfinput.iloc[i]["select"] == "snow_severe")
            | (dfinput.iloc[i]["select"] == "wet")
        ):
            ok = True
        elif (dfinput.iloc[i]["img_cat"] == "wet") & (
            (dfinput.iloc[i]["select"] == "snow") | (dfinput.iloc[i]["select"] == "dry")
        ):
            ok = True
        elif (dfinput.iloc[i]["img_cat"] == "dry") & (
            dfinput.iloc[i]["select"] == "wet"
        ):
            ok = True
        else:
            ok = False
        oks.append(ok)
    dfinput["ok_select"] = oks

    return dfinput


# Through this point, Done with all the ensembling work up through this point, now it's just aggregating the results across the 6 OT models into one final summary csv
# previously this was done in a separate notebook
# these two functions below are run after the 6 OT csvs are made


def calcstats(pred_df, predictioncolname):
    """
    val is input dataframe
    y_pred is a global variable of predictions
    """
    # print("inside calcstats function")
    # print(pred_df.columns)
    splitspecific_list = []

    # pred_df = pd.DataFrame()

    # pred_df["img_model_cnn"] = dfinput["img_model_cnn"]
    # pred_df["img_model_hrrr"] = dfinput["img_model_hrrr"]
    # pred_df["img_cat"] = dfinput["img_cat"]
    # pred_df["rf_pred"] = dfpreds

    pred_df = pred_df[
        ["img_orig", "img_cat", predictioncolname, f"ok_{predictioncolname}"]
    ]

    pred_df["correct_flag_forsummary"] = (
        pred_df["img_cat"] == pred_df[predictioncolname]
    )

    pred_df["ok_flag_forsummary"] = pred_df["img_cat"] == pred_df[predictioncolname]

    splitspecific_list.append(len(pred_df))
    splitspecific_list.append(len(pred_df[pred_df["correct_flag_forsummary"] == True]))
    splitspecific_list.append(len(pred_df[pred_df["ok_flag_forsummary"] == True]))

    # by class
    for c in [
        "snow_severe",
        "snow",
        "wet",
        "dry",
        "poor_viz",
        # "obs",
    ]:  # added obs. Note that this was not included in original merging method w hrrr RF + merge algo logic
        sub = pred_df[pred_df["img_cat"] == c]
        cat_total = len(sub)
        cat_correct = len(sub[sub["correct_flag_forsummary"] == True])
        splitspecific_list.append(cat_total)
        splitspecific_list.append(cat_correct)

    # print("end and printing splitspecific_list")
    # print(splitspecific_list)
    return splitspecific_list


def runstatsfor(ensemblefiles_base):
    """
    Creates stastics summaries for each outerTest | phase, aggregated at the outerTest level -- but results are saved as one csv for the inputted ensemble method

    Args:

        file_distinguish (string): "method1 or method2" or "select" or "select_prioritysspv"

    """

    resultslist = []

    # loop through outerTest numbers
    # otnum (string): "3" or "4

    for otnum in ["0", "1", "2", "3", "4", "5"]:  # ["0","1","2","3","4","5"]

        df = pd.read_csv(f"{ensemblefiles_base}_OT{otnum}.csv", low_memory=False)

        # print(df.columns)

        print(len(df))

        # this is the main one we want! outer test results aggregated
        # run the calcstats fn which returns a list with the info needed
        calcstats_for_df = calcstats(df, "select")
        # append other info to list
        calcstats_for_df = [f"OT{otnum}_outerTest"] + calcstats_for_df
        # append to outside list so all 6 results can be saved in one csv
        resultslist.append(calcstats_for_df)

    calcstats_colname_list = [
        "runname",
        "totalims",
        "correctims",
        "okims",
        "nims_snow_severe",
        "correct_snow_severe",
        "nims_snow",
        "correct_snow",
        "nims_wet",
        "correct_wet",
        "nims_dry",
        "correct_dry",
        "nims_poor_viz",
        "correct_poor_viz",
        # "nims_obs",
        # "correct_obs",
    ]

    resultsdf = pd.DataFrame(resultslist, columns=calcstats_colname_list)

    return resultsdf

