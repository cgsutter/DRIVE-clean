import _config as config
import helper_fns_adhoc
import call_model_train
import results_predictions_obslevel
import results_summaries
import pandas as pd
import load_dataset
import tensorflow as tf
import gc # Import garbage collection module
import os


def main(train_flag = config.train_flag, eval_flag = config.eval_flag, one_off = config.one_off, hyp_run = config.hyp_run):

    if one_off:
        tracker_rundetails, wandblog = helper_fns_adhoc.prep_str_details_track(
            arch_input=config.arch_set,
            epochs = config.epoch_set,
            l2use = config.l2_set,
            dropoutuse = config.dr_set,
            transfer_learning = config.transfer_learning,
            ast = config.ast,
            adhoc_desc = config.adhoc_desc,
            exp_desc = config.exp_desc,
            auguse = config.aug
            )
        if train_flag:
            for t in config.trackers_list:
                call_model_train.train_model(run_tracker = t, tracker_rundetails = tracker_rundetails, wandblog = wandblog, run_arch = config.arch_set, run_trle = config.transfer_learning, run_ast = config.ast, run_l2 =  config.l2_set, run_dr = config.dr_set, run_aug = config.aug)
        if eval_flag: 
            print("IN PROGRESS: Eval working for hyptuning, but need to get working for one-off runs here, simpler version of hyptuning")
            list_dfpreds = []
            list_descs30 = []
            for t in config.trackers_list:
                if config.eval_pred_csvs:
                    savepredsflag = True
                    phaseuse_set = "" # run preds on all observations
                else:
                    print("Evaluation for one-off model is only set up to make for prediction level inference, saving csvs for entire dataset. No high level summary code is needed for one-offs, those are done in notebooks. Thus, if this is printing, there in issue, should have entered other if statement.")
                if config.inference_other:
                    print("running inference on other data")
                    inference_otherdata_use = config.inference_data_csv
                else:
                    inference_otherdata_use = "" # setting this null will just run the trackers for both the data and corresponding model, the typical way
                list_descs30.append(os.path.basename(t)[:-4])
                # observation level
                preds = results_predictions_obslevel.make_preds(run_tracker = t, tracker_rundetails = tracker_rundetails, wandblog = wandblog, run_arch = config.arch_set, run_trle = config.transfer_learning, run_ast = config.ast, run_l2 = config.l2_set, run_dr = config.dr_set, run_aug = config.aug, saveflag = savepredsflag, phaseuse = phaseuse_set, inference_otherdata = inference_otherdata_use)
                list_dfpreds.append(preds)




    elif hyp_run:
        dfhyp = pd.read_csv(config.hyp_path)
        print("ENTER1: hyp_run")
        # print(dfhyp)
        # loop through each hyp set
        for i in range(0,len(dfhyp)): # 7/1/25 hyptuning starting, running in chunks of 10, may take up to 5-6 days. Typicall could be len(dfhyp) # here!! 7/8 need to train [3,10), and then [13,20). 7/12: run (3,32) all at once, i think it may overload dgx to do multiple eval runs

            # 1. Clear Keras backend session
            # This destroys the current TF graph and frees resources associated with it. 
            # Note: this did not end up fixing the tf issue w preds 
            tf.keras.backend.clear_session()

            # 2. Force Python garbage collection
            # Helps to reclaim memory from Python objects that are no longer referenced.
            # Note: this did not end up fixing the tf issue w preds 
            gc.collect()

            print(f"ENTER2: inside first loop - Hyp {i}")
            # unique to an experiment -- hyperparams and exp_desc
            h= dfhyp.iloc[i]
            print(f"HYPERPARAMS ARE: {h}")
            tracker_rundetails, wandblog = helper_fns_adhoc.prep_str_details_track(
                arch_input=dfhyp["arch"][i],
                epochs = config.epoch_set,
                l2use = dfhyp["l2"][i],
                dropoutuse = dfhyp["dr"][i],
                transfer_learning = dfhyp["trle"][i],
                ast = dfhyp["ast"][i],
                adhoc_desc = config.adhoc_desc,
                exp_desc = config.exp_desc,
                auguse = dfhyp["aug"][i]
                )
            if train_flag:
                for t in config.trackers_list:
                    call_model_train.train_model(run_tracker = t, tracker_rundetails = tracker_rundetails, wandblog = wandblog, run_arch = dfhyp["arch"][i], run_trle = dfhyp["trle"][i], run_ast = dfhyp["ast"][i], run_l2 =  dfhyp["l2"][i], run_dr = dfhyp["dr"][i], run_aug = dfhyp["aug"][i])
            if eval_flag:
                list_dfpreds = []
                list_descs30 = []
                for t in config.trackers_list:
                    if config.eval_pred_csvs:
                        savepredsflag = True
                        phaseuse_set = "" # run preds on all observations
                    elif config.eval_highlevel:
                        savepredsflag = False
                        phaseuse_set = "innerVal" # for HT results, just need o aggregate by innerVal
                    else:
                        print("check setup in config, need to set eval flags")
                    list_descs30.append(os.path.basename(t)[:-4])
                    # observation level
                    preds = results_predictions_obslevel.make_preds(run_tracker = t, tracker_rundetails = tracker_rundetails, wandblog = wandblog, run_arch = dfhyp["arch"][i], run_trle = dfhyp["trle"][i], run_ast = dfhyp["ast"][i], run_l2 =  dfhyp["l2"][i], run_dr = dfhyp["dr"][i], run_aug = dfhyp["aug"][i], saveflag = savepredsflag, phaseuse = phaseuse_set)
                    list_dfpreds.append(preds)

                # if running high level eval, grabbing one summary statistic row for one experiment (aggregated across 30), as is done for BL runs or HT runs, append to one main csv file with all the stats, then this summarizing code also needs to be ran
                if config.eval_highlevel:
                    # tracker level (df with 30 rows for 30 trackers)
                    results_df_of30 = results_summaries.run_results_by_exp(preddfs_input = list_dfpreds, preddfs_desc = list_descs30,  exp_desc_input = config.exp_desc, exp_details_input = tracker_rundetails, subsetphase = "innerVal")
                    print(results_df_of30[0:3])

                    # exp level (one result based on the hyp) -- results added/appended to primary results csv that already exists
                    results_summaries.exp_total_innerVal(df_innerVal = results_df_of30, exp_desc_input = config.exp_desc, exp_details_input = tracker_rundetails)

main()
