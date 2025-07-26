import config as config
import nested_ensemble_ynobs

# Currently this code works to evaluate the ODM model
# Not set up to work to evalute SCM data, although I think the code in /home/csutter/DRIVE-clean/ODM/scripts/nested_ensemble_ynobs.py has markings of that working, but need to come back to finish that. Not sure needed for manuscript, more of an inference/operational thing, so come back to this. 

# Note that the CNNs are trained and run predictions in the /CNN directory
# This script just aggregates results for ODM which are done differently than SCM bc there are only two folds, only 3 models to assess per fold, etc. 

# This runs first to create an ensemble csv per split (0 or 1) -- see manuscript for the data flow and organization here. 
for modelstr in ["0", "1"]:
    ## run data prep
    dd = nested_ensemble_ynobs.outsideTest_onedf_all5models(modelstr, config.dir_with_models)
    print("run here1")
    print(len(dd))
    print(dd.columns)
    dh1 = dd[dd["img_cat"] == "obs"]
    dh2 = dd[dd["img_cat"] == "nonobs"]
    print(len(dh1))
    print(len(dh2))
    # print(len(dd))
    # print(len(dd[dd["innerPhase"]=="innerTest"]))

    ## add cols which are lists of predicted cats
    list_colvalues = nested_ensemble_ynobs.dffn_combine_cols_tolist(
        dd,
        columns=[
            "A_model_pred",
            "B_model_pred",
            "C_model_pred",
            # "D_model_pred", # update here 3-member ensemble
            # "E_model_pred", # update here 3-member ensemble
        ],
    )
    dd["list_5preds"] = list_colvalues
    # print(dd)

    list_colvalues = nested_ensemble_ynobs.dffn_combine_cols_tolist(
        dd,
        columns=[
            "A_model_prob",
            "B_model_prob",
            "C_model_prob",
            # "D_predprob", # update here 3-member ensemble
            # "E_predprob", # update here 3-member ensemble
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
    ] = dd.apply(nested_ensemble_ynobs.rowfn_dict_calcs_from_5preds, axis=1, result_type="expand")

    # print("finished!!")
    print(dd[0:5])

    ## add cols for Method 2: avg

    dd2 = nested_ensemble_ynobs.dffn_return_avg_cols(dd)

    print("through averaging")

    ##  add cols for Methods 1: mode and 3: max

    dd2["ensembleMode_pred"] = dd2.apply(nested_ensemble_ynobs.rowfn_grab_mode, axis=1)
    print("through mode")

    dd2["ensembleMaxConf_pred"] = dd2.apply(nested_ensemble_ynobs.rowfn_grab_max_confidence, axis=1)
    print("through max confidence")

    # print(dd2[0:6])

    # Just for printing info/checking as needed. Usually comment out
    # for c in dd2.columns:
    #     print(c)

    dd2["methodalign_1_2"] = dd2["ensembleAvg_pred"] == dd2["ensembleMode_pred"]
    dd2["methodalign_1_3"] = (
        dd2["ensembleAvg_pred"] == dd2["ensembleMaxConf_pred"]
    )
    dd2["methodalign_2_3"] = (
        dd2["ensembleMode_pred"] == dd2["ensembleMaxConf_pred"]
    )

    print(sum(dd2["methodalign_1_2"]) / len(dd2))
    print(sum(dd2["methodalign_1_3"]) / len(dd2))
    print(sum(dd2["methodalign_2_3"]) / len(dd2))

    ## Add select from ensembles
    # kind of like the ensemble of ensemble steps but really just using methods 1 and 2, with 3 considered as a tie breaker
    dd2[["select", "decision"]] = dd2.apply(
        nested_ensemble_ynobs.rowfn_select_from_ensembles, axis=1, result_type="expand"
    )

    print(dd2[0:6])
    g1 = dd2[dd2["decision"] == "align_avg_mode"]
    g2 = dd2[dd2["decision"] == "tie_use_most_severe"]
    print(len(g1) / len(dd2))
    print(len(g2) / len(dd2))

    # add columns to of correct and oks based on select pred
    dd3 = nested_ensemble_ynobs.dffn_addcols_correct_oks(dd2)

    dd3.to_csv(f"{config.dir_save_ensemble}/{config.desc_of_modelflow}_OT{modelstr}.csv")

    print(f"done with {modelstr}")

# Secondly, aggregate results in one-row summary per split

summarydf = nested_ensemble_ynobs.runstatsfor(
    ensemblefiles_base=f"{config.dir_save_ensemble}/{config.desc_of_modelflow}"
)

summarydf.to_csv(f"{config.dir_save_ensemble}/_summary_{config.desc_of_modelflow}.csv")

# This code was related to having run the entire SCM dataset on the ODM model, and THEN subsetting it to the ynobs data only (but now, I've gone back to just evaluating on ODM first since that is priority for manuscript.) Come back to this. See note at top of this script

# summarydf = nested_ensemble_ynobs.runstatsforYNObsSubsetOnly(
#     ensemblefiles_base=f"{config.dir_save_ensemble}/{config.desc_of_modelflow}"
# )

# summarydf.to_csv(
#     f"{config.dir_save_ensemble}/_summary_{config.desc_of_modelflow}_ynobsSubset.csv"
# )
