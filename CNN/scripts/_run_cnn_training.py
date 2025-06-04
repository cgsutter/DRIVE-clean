import _config as config
import helper_fns_adhoc
import load_data
import model_compile_fit
import class_weights
import model_build
import callbacks


def train_model(run_tracker = config.trackers_list[0], run_arch = config.arch_set, run_trle = config.transfer_learning, run_ast = config.ast, run_l2 =  config.l2_set, run_dr = config.dr_set):

    print(f"Running {run_tracker} using architecture {run_arch}. Transfer learning {run_trle}, arch-specific top {run_ast}. Dropout is {run_dr} and l2 weight is {run_l2}.")

    tracker_filebase, tracker_rundetails = helper_fns_adhoc.prep_str_details_track(
        tracker_designated = run_tracker, 
        arch_input= run_arch,
        l2use = run_l2,
        dropoutuse = run_dr,
        transfer_learning = run_trle,
        ast = run_ast
        )

    print("through here")
    print(tracker_filebase)
    print(tracker_rundetails)
    print(f"saving model to dir {config.results_path}/{tracker_filebase}_{tracker_rundetails}")

    tf_ds_train, tf_ds_val, labels_train, labels_val, numims_train, traincatcounts = load_data.create_tf_datasets(tracker=run_tracker,
        cat_num = config.cat_num,
        BATCH_SIZE = config.BATCH_SIZE)

    print(type(tf_ds_train))
    print(numims_train)
    print(traincatcounts)
    # print(type(tfd2))

    dict_cat_str_ind = helper_fns_adhoc.cat_str_ind_dictmap()
    print(dict_cat_str_ind)
    class_weight_set= class_weights.classweights(
            labels_dict = dict_cat_str_ind,
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
    callbacks_use = callbacks.create_callbacks_list(savebestweights = f"{config.results_path}/{tracker_filebase}_{tracker_rundetails}", earlystop_patience = config.earlystop_patience, evid = config.evid)

    print("ran through callbakcs")
    print(type(callbacks_use))
    print(callbacks_use)

    model_compile_fit.train_fit(
        modelinput = m,
        traindata = tf_ds_train,
        valdata = tf_ds_val,
        callbacks_list = callbacks_use,
        epoch_set = config.epoch_set,
        BATCH_SIZE = config.BATCH_SIZE)

def main(one_off = config.one_off, hyp_run = config.hyp_run):
    if one_off:
        for t in config.trackers_list:
            train_model(run_tracker = t, run_arch = config.arch_set, run_trle = config.transfer_learning, run_ast = config.ast, run_l2 =  config.l2_set, run_dr = config.dr_set)

main()
