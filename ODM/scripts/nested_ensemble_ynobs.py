import os
import sys

import numpy as np
import pandas as pd
import config

# for each OTnum, grab the predictions from the 5 models, and then apply ensmebling techniques (trying more than one) and save out this df with all of those pieces.

# read in the dfs
# rename the cols since the 5 models will have 5 duplicate col names


def outsideTest_onedf_all5models(
    splitnum, parentdirofmodels
):

    """
    splitnum: str, e.g. "2"
    """

    splitint = int(splitnum)

    # ad hoc - to evaluate on the training data (as a means to check that val is worse to confirm we did all that correctly)... just set the reverse splitnums
    # update HERE!! if evaluation on Train instead of the main way (Val)
    if splitnum == "0":
        splitint = 0  # 0 if evaluating on validation (the main way!)
    else:
        splitint = 1  # 1 if evaluating on validation (the main way!)
    dir = parentdirofmodels
    print("inside outsideTest_onedf_all5models")
    print(parentdirofmodels)

    files = os.listdir(dir)
    print("initial amount")
    print(len(files))

    files = [f for f in files if config.files_subset in f]
    print(files)
    print(len(files))

    print(f"split{splitnum}.csv")
    files2 = [f for f in files if f"split{splitnum}" in f]
    print("files2")
    print(files2)
    files3 = []
    for f in files2:
        if "_A_" in f or "_B_" in f or "_C_" in f:
            files3.append(f)

    trackers_forOuterTest = sorted(files3)
    print("fileslist")
    print(len(trackers_forOuterTest))
    print(trackers_forOuterTest)

    print(
        f"found {len(trackers_forOuterTest)} many relevant trackers for outer test split {splitnum}"
    )
    # print(trackers_forOuterTest)

    dfs_individ = []

    for t in trackers_forOuterTest:
        # grab tracker name from which we will parse out info we need, with the config, to grab the final nowcast results (which was created in the nestedcv_train_blending model)
        # p1 = trackers_forOuterTest[0]
        r = pd.read_csv(f"{dir}/{t}")
        print("file read")
        print(f"{dir}/{t}")
        # print(r)
        print("inside!")
        print(len(r))
        # print("unique foldnum")
        # print(np.unique(r["foldnum"]))
        # # print(r.dtypes)
        # print("unique foldhalf")
        # print(np.unique(r["foldhalf"]))

        # print("unique foldnum_nested")
        # print(np.unique(r["foldnum_nested"]))
        # print(r.columns)

        # subset to just outerTest observations. O/w any observation that is kept in from say innerTest from OT0_m1 will be used as training or val in OT0_m2, so we can't really do this ensembling with anything besides OT

        r = r[r["foldhalf"] == splitint].reset_index()
        # print(r.columns)
        print(len(r))

        # r = r.drop(columns=["Unnamed: 0"])

        # grab info from file name which will be used to make new df
        find1 = t.rfind(f"split{splitnum}")
        innermodelnum = t[find1 - 2 : find1 - 1]  # should be like A or B
        print("here1")
        print(innermodelnum)
        # used for renaming cols for each of the 5 inner models
        print(f"inner model number {innermodelnum}")

        print(r.columns)

        # r["predprob"] = r[["prob_nonobs", "prob_obs"]].max(axis=1)

        # will read in all 5 results dfs for ensembling but all hve same col names so to work in a df (concatting all the columns) will want to rename the columns so not to have overlap
        cols_to_rename = [
            # "phase",  # must be renamed bc innerPhases are different for each of the 5 models inside this outerTest
            "prob_nonobs",
            "prob_obs",
            "model_pred",
            "model_prob",
        ]

        r = r.rename(columns={col: f"{innermodelnum}_{col}" for col in cols_to_rename})

        # print(r.columns)

        r = r.sort_values(
            by="img_name"
        )  # not needed since using merge and keeping img_name column

        if innermodelnum != "A":
            # drop all redundant identifyer cols except img_name
            # Since we will need to mergs all dfs, only keep the identifyer cols from the first df, and then merge all the rest of the dfs with just the model related columns.

            # only keep the columns needed for everything except the first
            r = r[
                [
                    "img_name",
                    f"{innermodelnum}_prob_nonobs",
                    f"{innermodelnum}_prob_obs",
                    f"{innermodelnum}_model_pred",
                    f"{innermodelnum}_model_prob",
                ]
            ]
        # r = r.drop(columns = ["innerPhase"])
        dfs_individ.append(r)

    print("insider here2")
    print(len(dfs_individ))
    merged_df = dfs_individ[0]  # grab the first df which has all the identifyer cols
    print(len(merged_df))
    print("check the three dfs in a list")
    print(len(dfs_individ[0]))
    print(len(dfs_individ[1]))
    print(len(dfs_individ[2]))

    for df_i in dfs_individ[1:]:  # add on by merging each df onto it, in a loop
        print("loop1")
        print(len(df_i))
        merged_df = merged_df.merge(df_i, on="img_name", how = "inner") # for evaluating on ODM data, there are different nonobs examples that don't overlap between the A, B, and C. That's the whole point of having the three samples. Thus, need to outer join across the A,B, and C. All are evaluated on the model trained with fold0, 
        print("loop2")
        print(len(merged_df))
    print(len(merged_df))
    # for i in merged_df.columns:
    #     print(i)

    print("merged df here!")
    print(merged_df.columns)
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
    mlist = ["A", "B", "C"]  # update here for 3-member ensemble "A", "B", "C", "D", "E"
    for i in range(0, len(mlist)):
        # add key value pair
        dict_modelkeys_predcatvalues[f"{mlist[i]}"] = list_5preds[i]
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
            models_w_predcat_probs.append(row[f"{m}_model_prob"])
            probs5.append(row[f"{m}_model_prob"])
            preds5.append(row[f"{m}_model_pred"])
        avg = np.mean(models_w_predcat_probs)
        # add key value pair
        dict_catAsKeys_probsAsValues[predcat] = avg

    # grab the 1 most confident prediction out of the 5 as a dictionary with single value
    dict_mostConfident_singleModel = {}
    for i in range(0, 3):  # update here 3-member ensemble
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
        if "nonobs" in max_keys:
            maxcat = "nonobs"
        elif "obs" in max_keys:
            maxcat = "obs"
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

    obscols = [
        "A_prob_obs",
        "B_prob_obs",
        "C_prob_obs",
        # "D_prob_obs", # update here 3-member ensemble
        # "E_prob_obs", # update here 3-member ensemble
    ]
    nonobscols = [
        "A_prob_nonobs",
        "B_prob_nonobs",
        "C_prob_nonobs",
        # "D_prob_nonobs", # update here 3-member ensemble
        # "E_prob_nonobs", # update here 3-member ensemble
    ]

    dfinput["ensembleAvg_obs"] = dfinput[obscols].mean(axis=1)
    dfinput["ensembleAvg_nonobs"] = dfinput[nonobscols].mean(axis=1)

    # grab the column of probs that was the highest and parse out to just get the cat name from the col name that was max

    # make into a temporary df just to pull the max

    dfinput["ensembleAvg_pred"] = (
        dfinput[
            [
                "ensembleAvg_obs",
                "ensembleAvg_nonobs",
            ]
        ]
        .idxmax(axis=1)
        .str[12:]
    )

    print("through here")

    # grab the max probability to retrun as model_prob

    dfinput["ensembleAvg_model_prob"] = dfinput[
        [
            "ensembleAvg_obs",
            "ensembleAvg_nonobs",
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
    # tried commenting out! 4/14
    elif m == c:
        select = m
        qualit = "align_mode_maxConf"
    else:  # it's a tie! Select the most severe
        qualit = "tie_use_most_severe"
        if "obs" in [a, m, c]:
            select = "obs"
        else:
            select = "nonobs"
    return select, qualit


# Add cols to categorize correct and oks


def dffn_addcols_correct_oks(dfinput):
    dfinput["select_correct"] = dfinput["select"] == dfinput["img_cat"]

    # no oks for the nonobs case

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

    print(len(pred_df))
    # print(pred_df.columns)
    splitspecific_list = []

    # pred_df = pd.DataFrame()

    # pred_df["img_model_cnn"] = dfinput["img_model_cnn"]
    # pred_df["img_model_hrrr"] = dfinput["img_model_hrrr"]
    # pred_df["img_cat"] = dfinput["img_cat"]
    # pred_df["rf_pred"] = dfpreds

    pred_df = pred_df[["img_orig", "img_cat", predictioncolname]]

    pred_df["correct_flag_forsummary"] = (
        pred_df["img_cat"] == pred_df[predictioncolname]
    )

    splitspecific_list.append(len(pred_df))
    splitspecific_list.append(len(pred_df[pred_df["correct_flag_forsummary"] == True]))

    # by class
    for c in [
        "obs",
        "nonobs",
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

    print("running runstatsfor() fn")

    for otnum in ["0", "1"]:

        df = pd.read_csv(f"{ensemblefiles_base}_OT{otnum}.csv", low_memory=False)

        print(f"{ensemblefiles_base}_OT{otnum}.csv")
        print(len(df))

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
        "nims_obs",
        "correct_obs",
        "nims_nonobs",
        "correct_nonobs",
    ]

    resultsdf = pd.DataFrame(resultslist, columns=calcstats_colname_list)

    return resultsdf


# Added to grab YNObsValOnly
def runstatsforYNObsSubsetOnly(ensemblefiles_base):
    """
    Creates stastics summaries for each outerTest | phase, aggregated at the outerTest level -- but results are saved as one csv for the inputted ensemble method

    Args:

        file_distinguish (string): "method1 or method2" or "select" or "select_prioritysspv"

    """

    resultslist = []

    # loop through outerTest numbers
    # otnum (string): "3" or "4

    print("running runstatsfor() fn")

    for otnum in ["0", "1"]:

        # ad hoc - to evaluate on the training data (as a means to check that val is worse to confirm we did all that correctly)... just set the reverse splitnums
        # update HERE!! if evaluation on Train instead of the main way (Val)
        if otnum == "0":
            splitint = 0  # 0 if evaluating on validation (the main way!)
        else:
            splitint = 1  # 1 if evaluating on validation (the main way!)

        df = pd.read_csv(f"{ensemblefiles_base}_OT{otnum}.csv", low_memory=False)

        print(f"{ensemblefiles_base}_OT{otnum}.csv")
        print("full val df")
        print(len(df))
        print(len(np.unique(df["img_name"])))
        print(np.unique(df["foldnum"]))
        print(np.unique(df["foldnum_nested"]))

        # this is what is different than regular runstatsfor() fn. The df we're reading in here is already subsetted just to val, correctly, we just need to subset one more time to take just the examples that were used in ynobs dataset (again, just the val, but as a way to evaluate the model correctly on all distinct nonobs examples which are different between splits A-C)

        dA = pd.read_csv(
            f"/home/csutter/DRIVE/dot/model_trackpaths_results/ynobs_A_split{otnum}_results.csv"
        )
        dA = dA[dA["foldnum"] == splitint]
        print("Number of obstructed images, which are the same for all A-C")
        numObs = len(dA[dA["img_cat"] == "obs"])
        print(numObs)
        dA = list(dA["img_name"])
        # print(dA.columns)
        # print(np.unique(dA["foldnum"]))
        # print(np.unique(dA["foldnum_nested"]))
        # print(len(dA))

        dB = pd.read_csv(
            f"/home/csutter/DRIVE/dot/model_trackpaths_results/ynobs_B_split{otnum}_results.csv"
        )
        dB = dB[dB["foldnum"] == splitint]
        dB = list(dB["img_name"])
        # print(np.unique(dB["foldnum"]))
        # print(np.unique(dB["foldnum_nested"]))
        # print(len(dB))

        dC = pd.read_csv(
            f"/home/csutter/DRIVE/dot/model_trackpaths_results/ynobs_C_split{otnum}_results.csv"
        )
        dC = dC[dC["foldnum"] == splitint]
        dC = list(dC["img_name"])
        # print(np.unique(dC["foldnum"]))
        # print(np.unique(dC["foldnum_nested"]))
        # print(len(dC))

        print("Count total unique")
        # print()
        grab_all_imgname_training = dA + dB + dC
        print(len(grab_all_imgname_training))
        print(len(np.unique(grab_all_imgname_training)))

        # check whats happening wrong w subset
        print("major check here!")
        inog = list(df["img_name"])
        check = [i for i in inog if i in grab_all_imgname_training]
        print(len(check))

        df = df[df["img_name"].isin(grab_all_imgname_training)]
        print("len after subsetting")
        print(len(df))
        print("number of unique nonobs images (which are diff) across the A-C models")
        numUniqueNonobs = len(df) - numObs
        print(numUniqueNonobs)
        print(
            "total number of unique imgs (obs and nonobs, of which nonobs will be ~3x) is:"
        )

        # also save out the subsetted df details for consistency
        df.to_csv(f"{ensemblefiles_base}_OT{otnum}_ynobsSubset.csv")
        print(len(df))
        # print(df.columns)

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
        "nims_obs",
        "correct_obs",
        "nims_nonobs",
        "correct_nonobs",
    ]

    resultsdf = pd.DataFrame(resultslist, columns=calcstats_colname_list)

    return resultsdf


#########################################################

