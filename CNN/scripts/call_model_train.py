import _config as config
import helper_fns_adhoc
import load_data
import model_compile_fit
import class_weights
import model_build
import callbacks
# import model_evaluation
# import model_results_summaries
import pandas as pd
import wandb
import os

def train_model(run_tracker = config.trackers_list[0], tracker_rundetails = "", wandblog = "", run_arch = config.arch_set, run_trle = config.transfer_learning, run_ast = config.ast, run_l2 =  config.l2_set, run_dr = config.dr_set, run_aug = config.aug, wandb_flag = config.wandb_flag):

    print(f"Running {run_tracker} using architecture {run_arch}. Transfer learning {run_trle}, arch-specific top {run_ast}. Dropout is {run_dr} and l2 weight is {run_l2}.")

    # really should move this outside of this function! It's only unique to an experiment & hyperparams NOT tracker, so since this def train_model is ran for each of the 30, it's superfluous.  Can actually probably remove this and move it outside under the first one_off and hyp_flag
    tracker_filebase = helper_fns_adhoc.prep_basefile_str(tracker_designated = run_tracker)

    print("logging to wb")
    print(wandblog)
    wandblog["data_desc"] = tracker_filebase

    if wandb_flag:
        wandb.init(
            project=config.wanb_projectname,  # your project name
            config=wandblog
        )
    else:
        print("not logging exp to wandb")


    modeldir_set = f"{config.model_path}/{tracker_filebase}_{tracker_rundetails}"

    print("through here")
    print(tracker_filebase)
    print(tracker_rundetails)
    print(f"saving model to dir {modeldir_set}")

    tf_ds_train, tf_ds_val, labels_train, labels_val, numims_train, traincatcounts = load_data.create_tf_datasets(tracker=run_tracker,
        cat_num = config.cat_num,
        BATCH_SIZE = config.BATCH_SIZE,
        augflag_use = run_aug)

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
    if wandb_flag:
        wandb.finish()