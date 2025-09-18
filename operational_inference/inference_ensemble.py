# Portions of this code were writen with the assistance of AI tools (e.g., ChatGPT)

import sys
sys.path.append("/home/csutter/DRIVE-clean/ensembling/scripts")

import nested_ensemble
import pandas as pd

model_nums = ["m0","m1","m2","m3","m4"]

# directory each of the ensemble member predictions are already saved
dir_with_preds = "/home/csutter/DRIVE-clean/operational_inference/data_5_downstreamcalib"
datafiles = [f"{dir_with_preds}/downstreamcalib_{i}.csv" for i in model_nums]
print(datafiles)

# Dir to save the output predicted downstream csv
directory_preds = "/home/csutter/DRIVE-clean/operational_inference/data_6_ensembling"
# directory_summaries = "/home/csutter/DRIVE-clean/ensembling/data_results" # wont need for inference mode

# ensemble_flag = True

# # ensemble_summary = True

# # set strings for saving and loading data
# # some description to differentiate
# desc_of_modelflow = "modelFinal"


##### FOR EVALUATION ON ALL 6 OUTER TEST FOLDS


# Ensembling for inference is simpler, won't have the outerloop with 6 different outertest datasets. For inference it's one new instance at a time (not 6 test sets)

## run data prep

dfs_individ = []

for i in range(0,len(datafiles)):

    # read in predicted probs
    r = pd.read_csv(datafiles[i])
    

    print(len(r))

    r = r.drop(columns=["Unnamed: 0"])

    # grab info from file name which will be used to make new df
    innermodelnum = f"m{i}"
    # used for renaming cols for each of the 5 inner models
    print(f"inner model number {innermodelnum}")

    # will read in all 5 results dfs for ensembling but all hve same col names so to work in a df (concatting all the columns) will want to rename the columns so not to have overlap
    cols_to_rename = [
        "calib_prob_dry",
        "calib_prob_poor_viz",
        "calib_prob_snow",
        "calib_prob_snow_severe",
        "calib_prob_wet",
        "calib_pred",
        "calib_prob",  # finalprob
    ]

    r = r.rename(columns={col: f"{innermodelnum}_{col}" for col in cols_to_rename})

    print(r.columns)

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

    print(r.columns)

    print("through col renaming")


# print(dfs_individ)
# print(type(dfs_individ))
# print(type(dfs_individ[0]))
# print(dfs_individ[0])
print(len(dfs_individ))
dd = dfs_individ[0]  # grab the first df which has all the identifyer cols, and merge it with the remaining 4 dfs below
for df in dfs_individ[1:]:  # add on by merging each df onto it, in a loop
    dd = dd.merge(df, on="img_name")

print(len(dd))
# for i in merged_df.columns:
#     print(i)

print("merged df here!")

print(dd.columns)
# print(dd.columns)
# for c in dd.columns:
#     print(c)
# return dd



#################

# print(len(dd))
# print(len(dd[dd["innerPhase"]=="innerTest"]))

print(dd.columns)

## add cols which are lists of predicted cats
list_colvalues = nested_ensemble.dffn_combine_cols_tolist(
    dd,
    columns=[
        "m0_calib_pred",
        "m1_calib_pred",
        "m2_calib_pred",
        "m3_calib_pred", 
        "m4_calib_pred", 
    ],
)
dd["list_5preds"] = list_colvalues
# print(dd)

list_colvalues = nested_ensemble.dffn_combine_cols_tolist(
    dd,
    columns=[
        "m0_calib_prob",
        "m1_calib_prob",
        "m2_calib_prob",
        "m3_calib_prob",
        "m4_calib_prob",
    ],
)
dd["list_5probs"] = list_colvalues
# print(dd)

## add dictionary columns
dd[
    [
        "dict_catAsKeys_modelAsValues",
        "dict_catAsKeys_countAsValues",
        "dict_catAsKeys_probsAsValues",
        "dict_mostConfident_singleModel",
    ]
] = dd.apply(
    nested_ensemble.rowfn_dict_calcs_from_5preds, axis=1, result_type="expand"
)

# print("finished!!")
print(dd[0:5])

## add cols for Method 2: avg

dd2 = nested_ensemble.dffn_return_avg_cols(dd)

print("through averaging")

##  add cols for Methods 1: mode and 3: max

dd2["ensembleMode_pred"] = dd2.apply(nested_ensemble.rowfn_grab_mode, axis=1)
print("through mode")

dd2["ensembleMaxConf_pred"] = dd2.apply(
    nested_ensemble.rowfn_grab_max_confidence, axis=1
)
print("through max confidence")

# print(dd2[0:6])

# Just for printing info/checking as needed. Usually comment out
# for c in dd2.columns:
#     print(c)

dd2["methodalign_1_2"] = dd2["ensembleAvg_pred"] == dd2["ensembleMode_pred"]
dd2["methodalign_1_3"] = dd2["ensembleAvg_pred"] == dd2["ensembleMaxConf_pred"]
dd2["methodalign_2_3"] = dd2["ensembleMode_pred"] == dd2["ensembleMaxConf_pred"]

print(sum(dd2["methodalign_1_2"]) / len(dd2))
print(sum(dd2["methodalign_1_3"]) / len(dd2))
print(sum(dd2["methodalign_2_3"]) / len(dd2))

## Add select from ensembles
# kind of like the ensemble of ensemble steps but really just using methods 1 and 2, with 3 considered as a tie breaker
dd2[["select", "decision"]] = dd2.apply(
    nested_ensemble.rowfn_select_from_ensembles, axis=1, result_type="expand"
)

print(dd2[0:6])
g1 = dd2[dd2["decision"] == "align_avg_mode"]
g2 = dd2[dd2["decision"] == "align_avg_maxConf"]
g3 = dd2[dd2["decision"] == "align_mode_maxConf"]
g4 = dd2[dd2["decision"] == "tie_use_most_severe"]
print(len(g1) / len(dd2))
print(len(g2) / len(dd2))
print(len(g3) / len(dd2))
print(len(g4) / len(dd2))
print("average is used")
print((len(g1) + len(g2)) / len(dd2))

# add columns to of correct and oks based on select pred
dd3 = nested_ensemble.dffn_addcols_correct_oks(dd2)

#### For inference, add in metric of confidence
# Amount of 5 models that align
# But within that ^, if there are two unique cats being predicted, e.g. wet and dry -- but 3 vs 2 is different than 4 vs 1. So the way to do this is: for the final predicted cat, how many (of the 5) models predicted it? 5,4,3,2, or 1. To get that, just grab the key from column "dict_catAsKeys_countAsValues", the cats are keys, and the value is the number of models that predicted it
# We will use two metrics of confidence: 
# 1) amount of 5 models that align -- think of this as "confidence_consistency"
# 2) avg probability -- think of this as "confidence_probability".
# Both pieces may be interesting. For example, if all 5 models predict wet, but the avg probability is 0.6, that is different than if all 5 models predict wet but the avg probability is 0.95. Or what if 4 models predict wet, 1 predicts dry, but the average for wet 0.90? Is that more or less confident than the case where all 5 aligned with wet but had average of 0.6? I dont think it's trivial, but all pieces of info may be valuable. 
# Use these two to get two metrics of confidence:
# A) Consisency score of: 1 (low, at most 3 models align), 2 (medium, four models align), or 3 (high, all five models align). 
# B) Probability score of 1 (<50%), 2 (50-85%), 3(85%+).  CHose 85 bc model does 81% acc so that should be a medium acc
# --> And then average to get overal "confidence". 0-1 is low, 1-2 medium, 2-3 high. 
# TO DO: make sure we have a good distribution of scores in each ^

def num_models_pred_cat(row):
    # grab the value that corresponds to the key which is the predicted cat (from "select" col)
    countnum = row["dict_catAsKeys_countAsValues"][row["select"]]
    return countnum

dd3["num_models_pred_cat"] = dd3.apply(num_models_pred_cat, axis = 1)

# translate the count into confidence 5 = high, 4 = medium, the rest = low
def confidence_consisency(row):
    # grab the value that corresponds to the key which is the predicted cat (from "select" col)
    ct = row["num_models_pred_cat"]
    if ct <= 3:
        conf = 1
    elif ct == 4:
        conf = 2
    elif ct == 5:
        conf = 3
    else:
        conf = 0
    return conf

dd3["conf_consist"] = dd3.apply(confidence_consisency, axis = 1)


def confidence_probability(row):
    probability = row["ensembleAvg_predprob"]
    if probability < 0.5:
        conf = 1
    elif ((probability >= 0.5) & (probability < 0.85)):
        conf = 2
    else:
        conf = 3
    return conf

dd3["conf_probability"] = dd3.apply(confidence_probability, axis = 1)

def confidence_overall(row):
    consistency = row["conf_consist"]
    probability = row["conf_probability"]
    overall = (consistency + probability)/2
    return overall

dd3["conf_overall"] = dd3.apply(confidence_overall, axis = 1)

def confidence_qual(row):
    overall = row["conf_overall"]
    if overall < 1:
        conf = "low"
    elif ((overall >= 1) & (overall < 2)):
        conf = "medium"
    else:
        conf = "high"
    return conf

dd3["confidence"] = dd3.apply(confidence_qual, axis = 1)


print(dd3.columns)

saveto = f"{directory_preds}/finalpreds.csv"
print(saveto)
dd3.to_csv(saveto)


