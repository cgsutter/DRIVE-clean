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


def train_model(run_tracker = config.trackers_list[0], run_arch = config.arch_set, run_trle = config.transfer_learning, run_ast = config.ast, run_l2 =  config.l2_set, run_dr = config.dr_set):

    print(f"Running {run_tracker} using architecture {run_arch}. Transfer learning {run_trle}, arch-specific top {run_ast}. Dropout is {run_dr} and l2 weight is {run_l2}.")

    tracker_filebase = prep_basefile_str(tracker_designated = run_tracker)
    tracker_rundetails = helper_fns_adhoc.prep_str_details_track(
        arch_input= run_arch,
        l2use = run_l2,
        dropoutuse = run_dr,
        transfer_learning = run_trle,
        ast = run_ast
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

def eval_model(tf_dataset_input, dataset_imgnames, run_tracker = config.trackers_list[0], run_arch = config.arch_set, run_trle = config.transfer_learning, run_ast = config.ast, run_l2 =  config.l2_set, run_dr = config.dr_set):

    
    tracker_filebase = prep_basefile_str(tracker_designated = run_tracker)
    tracker_rundetails = helper_fns_adhoc.prep_str_details_track(
        arch_input= run_arch,
        l2use = run_l2,
        dropoutuse = run_dr,
        transfer_learning = run_trle,
        ast = run_ast
        )

    modeldir_set = f"{config.model_path}/{tracker_filebase}_{tracker_rundetails}"
    predsdir_set = f"{config.preds_path}/{tracker_filebase}_{tracker_rundetails}"


    model_evaluation.evaluate(modeldir = modeldir_set, dataset = tf_dataset_input, imgnames = dataset_imgnames, savepreds = predsdir_set, trackerinput = run_tracker)

def results_summaries(run_exp_desc = config.exp_desc, run_arch = config.arch_set, run_trle = config.transfer_learning, run_ast = config.ast, run_l2 =  config.l2_set, run_dr = config.dr_set):

    
    
    tracker_rundetails = helper_fns_adhoc.prep_str_details_track( 
    arch_input= run_arch,
    l2use = run_l2,
    dropoutuse = run_dr,
    transfer_learning = run_trle,
    ast = run_ast
    )


    listy2 = model_results_summaries.results(exp_desc = run_exp_desc, preds_path = config.preds_path, exp_details = tracker_rundetails)

    return listy2


def main(train_flag = config.train_flag, eval_flag = config.eval_flag, summary_flag = config.summary_flag, one_off = config.one_off, hyp_run = config.hyp_run):
    if train_flag:
        if one_off:
            for t in config.trackers_list:
                train_model(run_tracker = t, run_arch = config.arch_set, run_trle = config.transfer_learning, run_ast = config.ast, run_l2 =  config.l2_set, run_dr = config.dr_set)
    if eval_flag:
        if one_off:
            # only need one tracker to pull all examples, the *full* dataset is the same across the 30 trackers
            t_grabany = config.trackers_list[0]
            dataset_all, all_labels, all_images = load_data.create_tf_datasets_for_evaluation(tracker = t_grabany,
                arch_set = config.arch_set,
                cat_num = config.cat_num,
                BATCH_SIZE = config.BATCH_SIZE)
            
            for t in config.trackers_list:
                print("inside loop for evaluation")
                eval_model(tf_dataset_input = dataset_all, dataset_imgnames = all_images, run_tracker = t, run_arch = config.arch_set, run_trle = config.transfer_learning, run_ast = config.ast, run_l2 =  config.l2_set, run_dr = config.dr_set)
            
    if summary_flag:

        tracker_rundetails = helper_fns_adhoc.prep_str_details_track( 
            arch_input= config.arch_set,
            l2use = config.l2_set,
            dropoutuse = config.dr_set,
            transfer_learning = config.transfer_learning,
            ast = config.ast
            )

        # predfiles = model_results_summaries.grab_pred_files(exp_desc = config.exp_desc, preds_path = config.preds_path, exp_details = tracker_rundetails)

        # for phase in ["", "innerTrain", "innerVal", "innerTest", "outerTest"]:
        #     model_results_summaries.run_results_by_exp(predfiles_input = predfiles, exp_desc_input = config.exp_desc, preds_path_input = config.preds_path, results_path_input = config.results_path, exp_details_input = tracker_rundetails, subsetphase = phase)
        

        model_results_summaries. exp_total_innerVal(exp_desc_input = config.exp_desc, preds_path_input = config.preds_path, results_path_input = config.results_path, exp_details_input = tracker_rundetails)

        # tracker_rundetails = helper_fns_adhoc.prep_str_details_track( 
        #     arch_input= run_arch,
        #     l2use = run_l2,
        #     dropoutuse = run_dr,
        #     transfer_learning = run_trle,
        #     ast = run_ast
        #     )
            
        # results_30dicts = []



        # listy3  = results_summaries(run_exp_desc = config.exp_desc, run_arch = config.arch_set, run_trle = config.transfer_learning, run_ast = config.ast, run_l2 =  config.l2_set, run_dr = config.dr_set)



                
            

            # tf_ds_train, tf_ds_val, labels_train, labels_val, numims_train, traincatcounts = load_data.create_tf_datasets(tracker=run_tracker,cat_num = config.cat_num, BATCH_SIZE = config.BATCH_SIZE)




main()
