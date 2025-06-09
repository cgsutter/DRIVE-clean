import _config as config
import helper_fns_adhoc
import load_data
import model_compile_fit
import class_weights
import model_build
import callbacks
import model_evaluation
import model_results_summaries
import pandas as pd
import wandb
import os

def train_model(run_tracker = config.trackers_list[0], run_arch = config.arch_set, run_trle = config.transfer_learning, run_ast = config.ast, run_l2 =  config.l2_set, run_dr = config.dr_set, run_aug = config.aug):

    print(f"Running {run_tracker} using architecture {run_arch}. Transfer learning {run_trle}, arch-specific top {run_ast}. Dropout is {run_dr} and l2 weight is {run_l2}.")

    # really should move this outside of this function! It's only unique to an experiment & hyperparams NOT tracker, so since this def train_model is ran for each of the 30, it's superfluous.  Can actually probably remove this and move it outside under the first one_off and hyp_flag
    tracker_filebase = helper_fns_adhoc.prep_basefile_str(tracker_designated = run_tracker)
    tracker_rundetails, wandblog = helper_fns_adhoc.prep_str_details_track(
        # tracker_designated = run_tracker,
        arch_input=run_arch,
        epochs = config.epoch_set,
        l2use = run_l2,
        dropoutuse = run_dr,
        transfer_learning = run_trle,
        ast = run_ast,
        adhoc_desc = config.adhoc_desc,
        exp_desc = config.exp_desc,
        auguse = run_aug
        )

    print("logging to wb")
    print(wandblog)
    wandblog["data_desc"] = tracker_filebase


    wandb.init(
        project="DRIVE-clean",  # your project name
        config=wandblog
    )


    modeldir_set = f"{config.model_path}/{tracker_filebase}_{tracker_rundetails}"

    print("through here")
    print(tracker_filebase)
    print(tracker_rundetails)
    print(f"saving model to dir {modeldir_set}")

    tf_ds_train, tf_ds_val, labels_train, labels_val, numims_train, traincatcounts = load_data.create_tf_datasets(tracker=run_tracker,
        cat_num = config.cat_num,
        BATCH_SIZE = config.BATCH_SIZE)

    print(type(tf_ds_train))
    print(numims_train)
    print(traincatcounts)
    # print(type(tfd2))

    dict_catKey_indValue, dict_indKey_catValue = helper_fns_adhoc.cat_str_ind_dictmap()
    print(dict_catKey_indValue)

    class_weight_set= class_weights.classweights(
            labels_dict = dict_catKey_indValue,
            wts_use = config.class_wts,
            trainlabels =list(labels_train),
            balance=config.class_balance,
            setclassimportance=config.setclassimportance,  # [0.15, 0.10, 0.15, 0.225, 0.225, 0.15]
            num_train_imgs=numims_train,
            train_cat_cts=traincatcounts,
        )

    print(class_weight_set)

    model = model_build.model_baseline(
        # one_off = config.one_off,
        # hyp_run = config.hyp_run,
        evid = config.evid,
        num_classes = config.cat_num,
        input_shape = (config.imheight, config.imwidth, 3),
        arch = run_arch,
        transfer_learning = run_trle,
        ast = run_ast,
        dropout_rate = run_dr,
        l2weight =run_l2,
        activation_layer_def = config.activation_layer_def,
        activation_output_def = config.activation_output_def
        )

    print(model)

    m = model_compile_fit.compile_model(model = model, train_size = numims_train, batchsize = config.BATCH_SIZE, lr_init = config.lr_init, lr_opt=config.lr_opt, lr_after_num_of_epoch =config.lr_after_num_of_epoch, lr_decayrate = config.lr_decayrate, momentum = config.momentum, evid = config.evid, evid_lr_init = config.evid_lr_init )

    print("through compiled model")
    print("model here")
    print(type(m))
    callbacks_use = callbacks.create_callbacks_list(savebestweights = modeldir_set, earlystop_patience = config.earlystop_patience, evid = config.evid)

    print("ran through callbakcs")
    print(type(callbacks_use))
    print(callbacks_use)

    model_compile_fit.train_fit(
        modelinput = m,
        traindata = tf_ds_train,
        valdata = tf_ds_val,
        callbacks_list = callbacks_use,
        class_weights_use = class_weight_set,
        evid = config.evid,
        epoch_set = config.epoch_set,
        BATCH_SIZE = config.BATCH_SIZE)
    
    wandb.finish()

# def eval_model(tf_dataset_input, dataset_imgnames, run_tracker = config.trackers_list[0], run_arch = config.arch_set, run_trle = config.transfer_learning, run_ast = config.ast, run_l2 =  config.l2_set, run_dr = config.dr_set,run_aug = config.aug, flag_save_allpreds = False):

    
#     tracker_filebase = prep_basefile_str(tracker_designated = run_tracker)
#     tracker_rundetails, wandblog = helper_fns_adhoc.prep_str_details_track(
#         # tracker_designated = run_tracker,
#         arch_input=run_arch,
#         epochs = config.epoch_set,
#         l2use = run_l2,
#         dropoutuse = run_dr,
#         transfer_learning = run_trle,
#         ast = run_ast,
#         adhoc_desc = config.adhoc_desc,
#         exp_desc = config.exp_desc,
#         auguse = run_aug
#         )

#     modeldir_set = f"{config.model_path}/{tracker_filebase}_{tracker_rundetails}"
#     predsdir_set = f"{config.preds_path}/{tracker_filebase}_{tracker_rundetails}"

#     df_preds = model_evaluation.evaluate(modeldir = modeldir_set, dataset = tf_dataset_input, imgnames = dataset_imgnames, trackerinput = run_tracker)

#     if flag_save_allpreds:
#         df_preds.to_csv(predsdir_set)

#     return df_preds
    
    



def main(train_flag = config.train_flag, eval_flag = config.eval_flag, summary_flag = config.summary_flag, one_off = config.one_off, hyp_run = config.hyp_run):
    if train_flag:
        if one_off:
            for t in config.trackers_list:
                train_model(run_tracker = t, run_arch = config.arch_set, run_trle = config.transfer_learning, run_ast = config.ast, run_l2 =  config.l2_set, run_dr = config.dr_set, run_aug = config.aug)
        elif hyp_run:
            dfhyp = pd.read_csv(config.hyp_path)
            print(dfhyp)
            for i in range(9,18): #len(dfhyp)
                print(i)
                print(dfhyp["arch"][i]) 
                print(dfhyp["trle"][i])
                print(dfhyp["ast"][i])
                print(dfhyp["l2"][i])
                print(dfhyp["dr"][i])
                for t in config.trackers_list:
                    train_model(run_tracker = t, run_arch = dfhyp["arch"][i], run_trle = dfhyp["trle"][i], run_ast = dfhyp["ast"][i], run_l2 =  dfhyp["l2"][i], run_dr = dfhyp["dr"][i], run_aug = dfhyp["aug"][i])

    # right now not working for hyp. Will need to pass in hyps like for model training
    if eval_flag: # note that this is ran after the model training, rather than in the same script, bc need to evaluate it on the full dataset (not just the val subset)
        # only need one tracker to pull all examples, the *full* dataset is the same across the 30 trackers. Note that this is the same regardless of one-off of hyperparameter tuning run - just need to create one tf dataset of all data observations

        print("eval1")
        t_grabany = config.trackers_list[0]
        dataset_all, all_labels, all_images = load_data.create_tf_datasets_for_evaluation(tracker = t_grabany,
            arch_set = config.arch_set,
            cat_num = config.cat_num,
            BATCH_SIZE = config.BATCH_SIZE)
        if one_off:
            print("one_off evalA")
            tracker_filebase = helper_fns_adhoc.prep_basefile_str(tracker_designated = config.trackers_list[0])
            tracker_rundetails, wandblog = helper_fns_adhoc.prep_str_details_track(
                # tracker_designated = run_tracker,
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
            print("one_off evalB")

            modeldir_set = f"{config.model_path}/{tracker_filebase}_{tracker_rundetails}"
            predsdir_set = f"{config.preds_path}/{tracker_filebase}_{tracker_rundetails}"

            preddfs_30 = []
            descs_30 = []
            for t in config.trackers_list:
                print("one_off evalB")
                print("inside loop for evaluation")
                df_preds = model_evaluation.evaluate(modeldir = modeldir_set, dataset = dataset_all, imgnames = all_images, trackerinput = t)

                t_name = os.path.basename(t)[:-4]
                print(t_name)
                predssaveto = f"{config.preds_path}/{t_name}_{tracker_rundetails}.csv"
                df_preds.to_csv(predssaveto) # saving the preds df to csv for one-off runs
                preddfs_30.append(df_preds)
                # grab the tracker differentiator
                # trackerdesc = helper_fns_adhoc.tracker_differentiator(t)
                descs_30.append(t_name)

            print("one_off evalC")
            for phase in ["", "innerTrain", "innerVal", "innerTest", "outerTest"]:
                results_df_of30 = model_results_summaries.run_results_by_exp(preddfs_input = preddfs_30, preddfs_desc = descs_30,  exp_desc_input = config.exp_desc, exp_details_input = tracker_rundetails, subsetphase = phase)
                csvsave = f"{config.results_path}/{config.exp_desc}_{tracker_rundetails}_{phase}.csv"
                print(f"saving to {csvsave}")
                results_df_of30.to_csv(csvsave) # save out all phases to csv
                print("one_off evalD")
                if phase == "innerVal": # Append to Main tracker of all experiments performance, but only for innerVal
                    model_results_summaries.exp_total_innerVal(df_innerval = results_df_of30, exp_desc_input = config.exp_desc, exp_details_input = tracker_rundetails)
                    print("one_off evalE")

                    
            
    
        if hyp_run:
            dfhyp = pd.read_csv(config.hyp_path)
            print(dfhyp)
            tracker_filebase = helper_fns_adhoc.prep_basefile_str(tracker_designated = config.trackers_list[0])
            tracker_rundetails, wandblog = helper_fns_adhoc.prep_str_details_track(
                # tracker_designated = run_tracker,
                arch_input=dfhyp["arch"][i],
                epochs = config.epoch_set,
                l2use = dfhyp["l2"][i],
                dropoutuse = dfhyp["dr"][i],
                transfer_learning = dfhyp["trle"][i],
                ast = dfhyp["ast"][i],
                adhoc_desc = config.adhoc_desc,
                exp_desc = config.exp_desc,
                auguse = config.aug
                )

            modeldir_set = f"{config.model_path}/{tracker_filebase}_{tracker_rundetails}"
            predsdir_set = f"{config.preds_path}/{tracker_filebase}_{tracker_rundetails}"
            for i in range(9,18): #len(dfhyp)
                preddfs_30 = []
                descs_30 = []
                for t in config.trackers_list:
                    print("inside loop for evaluation")
                    df_preds = model_evaluation.evaluate(modeldir = modeldir_set, dataset = dataset_all, imgnames = all_images, trackerinput = t)

                    # do not save all preds for all trackers under HTing -- too much data, and not needed since all we need is summary analysis
                    preddfs_30.append(df_preds)
                    trackerdesc = helper_fns_adhoc.tracker_differentiator(t)
                    descs_30.append(trackerdesc)
                # only run for inner val, no need to run for all phases bc not saving those to csv, only need inner val to then run the final summary analysis to append to Main tracker for experiment results in total
                results_df_of30 = model_results_summaries.run_results_by_exp(preddfs_input = preddfs_30, preddfs_desc = descs_30,  exp_desc_input = config.exp_desc, exp_details_input = tracker_rundetails, subsetphase = "innerVal")

                model_results_summaries.model_results_summaries.exp_total_innerVal(df_innerval = results_df_of30, exp_desc_input = config.exp_desc, exp_details_input = tracker_rundetails)



                
    if summary_flag: # right now not working for hyp. Will need to pass in hyps like for model training

        tracker_rundetails, wandblog = helper_fns_adhoc.prep_str_details_track( 
            arch_input=config.arch_set,
            epochs = config.epoch_set,
            l2use = config.l2_set,
            dropoutuse = config.dr_set,
            transfer_learning = config.transfer_learning,
            ast = config.ast,
            adhoc_desc = config.adhoc_desc,
            exp_desc = config.exp_desc,
            run_aug = config.aug
            )

        predfiles = model_results_summaries.grab_pred_files(exp_desc = config.exp_desc, preds_path = config.preds_path, exp_details = tracker_rundetails)

        for phase in ["", "innerTrain", "innerVal", "innerTest", "outerTest"]:
            model_results_summaries.run_results_by_exp(preddfs_input = predfiles, exp_desc_input = config.exp_desc, preds_path_input = config.preds_path, results_path_input = config.results_path, exp_details_input = tracker_rundetails, subsetphase = phase)
        
        model_results_summaries.exp_total_innerVal(exp_desc_input = config.exp_desc, preds_path_input = config.preds_path, results_path_input = config.results_path, exp_details_input = tracker_rundetails)

        # tracker_rundetails = helper_fns_adhoc.prep_str_details_track( 
            # arch_input=config.arch_set,
            # epochs = config.epoch_set,
            # l2use = config.l2_set,
            # dropoutuse = config.dr_set,
            # transfer_learning = config.transfer_learning,
            # ast = config.ast,
            # adhoc_desc = config.adhoc_desc,
            # exp_desc = config.exp_desc
        #     )
            
        # results_30dicts = []



        # listy3  = results_summaries(run_exp_desc = config.exp_desc, run_arch = config.arch_set, run_trle = config.transfer_learning, run_ast = config.ast, run_l2 =  config.l2_set, run_dr = config.dr_set)



                
            

            # tf_ds_train, tf_ds_val, labels_train, labels_val, numims_train, traincatcounts = load_data.create_tf_datasets(tracker=run_tracker,cat_num = config.cat_num, BATCH_SIZE = config.BATCH_SIZE)




main()
