import config
import nested_ensemble


##### FOR EVALUATION ON ALL 6 OUTER TEST FOLDS


if config.ensemble_flag:
    for modelstr in ["0", "1", "2", "3", "4", "5"]:
        ## run data prep

        dd = nested_ensemble.outsideTest_onedf_all5models(modelstr, config.dir_with_models, subsetflag = config.subset_files_torun, subsetstring = config.subset_string)

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
        ] = dd.apply(nested_ensemble.rowfn_dict_calcs_from_5preds, axis=1, result_type="expand")

        # print("finished!!")
        print(dd[0:5])

        ## add cols for Method 2: avg

        dd2 = nested_ensemble.dffn_return_avg_cols(dd)

        print("through averaging")

        ##  add cols for Methods 1: mode and 3: max

        dd2["ensembleMode_pred"] = dd2.apply(nested_ensemble.rowfn_grab_mode, axis=1)
        print("through mode")

        dd2["ensembleMaxConf_pred"] = dd2.apply(nested_ensemble.rowfn_grab_max_confidence, axis=1)
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

        dd3.to_csv(f"{config.directory_preds}/{config.desc_of_modelflow}_OT{modelstr}.csv")

        print(f"done with {modelstr}")

# to  be run AFTER the above loop is run (evaluated ensembles for all outertest sets)

if config.ensemble_summary:
    summarydf = nested_ensemble.runstatsfor(
        ensemblefiles_base=f"{config.directory_preds}/{config.desc_of_modelflow}"
    )

    summarydf.to_csv(f"{config.directory_summaries}/_summary_{config.desc_of_modelflow}.csv")

# To do;
# 0 - get it working looping through all modelnums
# 1 - confidence values (like for UI)
# 2 - use the avg of the pred cat for confidence? or # that align? or avg prob?
# 3- need to do analyses about how much value each of the ensembling methods provides.  --95%-96% align!
# 4 - need to rerun the blending method vs this new ensembling code
