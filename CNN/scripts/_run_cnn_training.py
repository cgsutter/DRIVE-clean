import _config as config
import helper_fns_adhoc
import call_model_train
import call_model_results
import pandas as pd
import load_dataset
import tensorflow as tf

# Get a list of all available physical GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')

# if gpus:
#     try:
#         # Loop through each physical GPU and set memory growth
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
        
#         # Optional: Log which GPUs are being used and their memory growth status
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs configured with memory growth.")
#     except RuntimeError as e:
#         # This error typically occurs if memory growth is set after GPUs have been initialized.
#         # Ensure this code runs at the very start of your program.
#         print(e)
# else:
#     print("No GPU devices found. Running on CPU.")

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
        if eval_flag: # Evaluation code has to be ran after the model training, rather than in the same script, bc need to evaluate it on the full dataset (not just the val subset)

            # This is for evaluation, where the same full dataset is used regardless of one-off vs hyp tune run, and this data should be prepared and read in outside of loops through trackers -- would be redundant to read in for each since it's the same full dataframe.Note that we only need one tracker to pull all examples, the *full* dataset is the same across the 30 trackers.
            dataset_all, imgnames, labels, numims, catcounts = load_dataset.load_data(trackerinput = config.trackers_list[0], phaseinput = "", archinput = config.arch_set, auginput = False) # aug always false for evaluation. The data loading is unique to the architecture due to data preprocessing.

            call_model_results.evaluate_exp_models(trackerslist = config.trackers_list, tracker_rundetails = tracker_rundetails, oneoff_flag = config.one_off, hyp_flag = config.hyp_run, dfhyp = None, dataset_all = dataset_all, all_images = imgnames, arch_eval = config.arch_set, trle_eval = config.transfer_learning, ast_eval=config.ast, dr_eval = config.dr_set, l2_eval = config.l2_set)

    elif hyp_run:
        dfhyp = pd.read_csv(config.hyp_path)
        print("ENTER1: hyp_run")
        # print(dfhyp)
        # loop through each hyp set
        for i in range(0,4): #len(dfhyp) # here!! 6/26 skipped 1 for now, error in vim ai2es_error_20164.err. Two jobs running 2,10 and 10,18
            print(f"ENTER2: LOOP Hyp {i}")
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
                print(f"ENTER3: eval flag (should be ran ONCE per LOOP ^")
                dataset_all, imgnames, labels, numims, catcounts = load_dataset.load_data(trackerinput = config.trackers_list[0], phaseinput = "", archinput = dfhyp["arch"][i], auginput = False) # aug always false for evaluation. The data loading is unique to the architecture due to data preprocessing.

                print("INSPECTING")
                for images, labels in dataset_all.take(1): # Take one batch
                    print("Shape of images from dataset:", images.shape)
                    print("Shape of labels from dataset:", labels.shape)
                    break

                call_model_results.evaluate_exp_models(trackerslist = config.trackers_list, tracker_rundetails = tracker_rundetails, oneoff_flag = config.one_off, hyp_flag = config.hyp_run, dfhyp = dfhyp, dataset_all = dataset_all, all_images = imgnames, arch_eval = dfhyp["arch"][i], trle_eval =dfhyp["trle"][i], ast_eval = dfhyp["ast"][i], dr_eval =dfhyp["dr"][i], l2_eval =dfhyp["l2"][i])

main()
