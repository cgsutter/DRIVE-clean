import _config as config
import helper_fns_adhoc
import load_data
import call_model_train
import call_model_results
import pandas as pd


def main(train_flag = config.train_flag, eval_flag = config.eval_flag, one_off = config.one_off, hyp_run = config.hyp_run):

    if eval_flag:
        # This is for evaluation, where the same full dataset is used regardless of one-off vs hyp tune run, and this data should be prepared and read in outside of any loops (trackers, HT hyperparams, etc) -- would be redundant to read in for each since it's the same full dataframe.
        # Note that we only need one tracker to pull all examples, the *full* dataset is the same across the 30 trackers. 
        dataset_all, all_labels, all_images = load_data.create_tf_datasets_for_evaluation(tracker = config.trackers_list[0], arch_set = config.arch_set, cat_num = config.cat_num, BATCH_SIZE = config.BATCH_SIZE)

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
        if eval_flag: # Evaluation code has to be ran after the model training, rather than in the same script, bc need to evaluate it on the full dataset (not just the val subset)
            call_model_results.evaluate_exp_models(trackerslist = config.trackers_list, tracker_rundetails = tracker_rundetails, oneoff_flag = config.one_off, hyp_flag = config.hyp_run, dfhyp = None, dataset_all = dataset_all, all_labels = all_labels, all_images = all_images)

    elif hyp_run:
        dfhyp = pd.read_csv(config.hyp_path)
        # print(dfhyp)
        for i in range(27,36): #len(dfhyp) # here!!
            # unique to an experiment -- hyperparams and exp_desc
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
                call_model_results.evaluate_exp_models(trackerslist = config.trackers_list, tracker_rundetails = tracker_rundetails, oneoff_flag = config.one_off, hyp_flag = config.hyp_run, dfhyp = dfhyp, dataset_all = dataset_all, all_labels = all_labels, all_images = all_images)

main()
