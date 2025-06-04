import config
import helper_fns_adhoc
import build_compile
import class_weights
import build_baseline
import build_bn_dropout




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

    tf_ds_train, tf_ds_val, labels_train, labels_val, numims_train, traincatcounts = build_compile.create_tf_datasets(tracker=run_tracker,
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

    model = build_baseline.model_baseline(
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

    m = build_compile.compile_model(model = model, train_size = numims_train, batchsize = config.BATCH_SIZE, lr_init = config.lr_init, lr_opt=config.lr_opt, lr_after_num_of_epoch =config.lr_after_num_of_epoch, lr_decayrate = config.lr_decayrate, momentum = config.momentum, evid = config.evid, evid_lr_init = config.evid_lr_init )

    print("through compiled model")
    print("model here")
    print(type(m))
    callbacks_use = build_bn_dropout.create_callbacks_list(savebestweights = f"{config.results_path}/{tracker_filebase}_{tracker_rundetails}", earlystop_patience = config.earlystop_patience, evid = config.evid)

    print("ran through callbakcs")
    print(type(callbacks_use))
    print(callbacks_use)

    build_compile.build(
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


# build_compile.build(
#     tracker = config.trackers_list[0],
#     save_model_to = f"{config.results_path}/{tracker_filebase}_{tracker_rundetails}",
#     cat_num = config.cat_num,
#     category_dirs = config.category_dirs,
#     colormode = config.colormode,
#     class_wts = config.class_wts,
#     arch_set=config.arch_set,
#     l2_set = config.l2_set,  # hyp
#     dr_set = config.dr_set,  # hyp
#     epoch_set = config.epoch_set,
#     lr_init=config.lr_init,
#     lr_decayrate=config.lr_decayrate,
#     SHUFFLE_BUFFER_SIZE = config.SHUFFLE_BUFFER_SIZE,
#     BATCH_SIZE = config.BATCH_SIZE,
#     evid = config.evid,
#     evid_annealing_coeff = config.evid_annealing_coeff,
#     evid_lr_init = config.evid_lr_init
# )

    
# def train_cnn(l2use,
#     dropoutuse,
#     batchuse,
#     lrinituse,
#     lrdecruse,
#     kfold, # use?
#     split, # use?
#     hypset_on, # wont need with refactoring since will loop through
#     arch_input=config.arch,
#     tracker_designated=""):



#     savemodelto = f"{savemodelfiles}/{tracker_w_details}"  

#     print("Starting model build...")

#     src.build_compile.build(
#         label_dict_keyIs_catString,
#         config.kfoldflag,
#         split,
#         accloss,
#         # savejson,
#         # saveweights,
#         savemodelto,
#         losstitle,
#         acctitle,
#         l2use,
#         dropoutuse,
#         batchuse,
#         tracker=tracker,
#         tracker_name=tracker_name,
#         append_details=append_details,
#         wb_config=wb_config_build_run,
#         # wb_savemodel_art=wb_save_model_artifact,
#         model_detail=f"{config.desc}_{append_details}",
#         lr_init=lrinituse,
#         lr_decr=lrdecruse,
#         arch_input=arch_input,
#     )



# def build_call(
#     l2use,
#     dropoutuse,
#     batchuse,
#     lrinituse,
#     lrdecruse,
#     kfold,
#     split,
#     hypset_on,
#     arch_input=config.arch,
#     tracker_designated="",  # Only used for nested kfold cv right now, see where config.kfoldnested == True for how this is used
# ):

#     """
#     Input hyperparameters
#     Input kfold True or False
#     Input fold (number of split on if using kfold True -- the splits are looped through in final __main__ function)

#     Grabs the right tracker to use depending on if using kfold or not
#     Sets the artifact names for model building
#     Creates string used for saving out model build and results

#     Runs model build based on the creation of the above pieces
#     """

#     print("Preparing information for model build")

#     print("preparing new dictionaries")
#     (
#         label_dict_keyIs_catNumber,
#         label_dict_keyIs_catString,
#     ) = src.data_input_pipeline.create_label_dict(config.category_dirs)
    

#     # Create the string of details to append for file naming (for things like model, acc and loss curves, etc) - this depends on hyperparams used and also whether using kfold splits or not. Also grab the right tracker path, and tracker_name (which is a short name for the artifact naming) -- both depend on if using k fold or not. Note that trackers dont depend on hyperparams bc multiple models can be built from the same original tracker.
#     if kfold == True:
#         print("using k fold")
#         append_details = f"spl{split}_tune_l2{l2use_desc}_dr{dropoutuse_desc}_b{batchuse_desc}_lr{lrinituse_desc}_lrdecr{lrdecruse_desc}_e{config.epoch_def}_es{config.earlystop_patience}_emin{config.min_epochs_before_es}"  # everything that is not in the main desc which has vgg16, 6 class, and no aug
#         # if config.baseline == True:
#         append_details = f"arch_{arch_input}_{append_details}"  # used for saving out model and results with model details
#         tracker = f"{config.base_dir}/dot/model_trackpaths/{config.desc}_split{split}.csv"  # used for reading in image data
#         tracker_name = (
#             f"{config.desc}_split{split}"  # SHOULDNT NEED THIS-ITS JUST FOR WB
#         )
#     elif config.kfoldnested == True:
#         print("using nested k fold")
#         print(
#             "using a list of files to be trained. The files were prepared already in preprocessing, and then the list of files being read in from trackers_to_run.py. Besides the list of trackers, it will use the hyperparameters as set in config for each one."
#         )
#         begintracker = tracker_designated.rfind("/")
#         tracker_filebase = tracker_designated[
#             begintracker + 1 : -4
#         ]  # remove .csv extension

#         tracker_w_details = f"{tracker_filebase}_arch_{arch_input}_l2{l2use_desc}_dr{dropoutuse_desc}_b{batchuse_desc}_lr{lrinituse_desc}_lrdecr{lrdecruse_desc}_e{config.epoch_def}_es{config.earlystop_patience}_emin{config.min_epochs_before_es}"  # everything that is not in the main desc which has vgg16, 6 class, and no aug # used for saving out model and results with model details
#         tracker = tracker_designated  # used for reading in image data
#         tracker_name = tracker_filebase  # SHOULDNT NEED THIS-IT'S JUST FOR WB
#         print(tracker_w_details)
#         append_details = tracker_w_details  # JUST TO KEEP THIS append_details VARIABLE IN HERE TO NOT BREAK OTHER PARTS, bUT REALLY JUST FOR W&B stuff
#     else:  # fix this later, should have arch in here..
#         print("using validation, not k fold")
#         fold = 0  # just a placeholder (not used) so that build() will run
#         append_details = f"_nokfold_notune_l2_{l2use_desc}_drop_{dropoutuse_desc}_batch_{batchuse_desc}_lr{lrinituse_desc}_lrdecr{lrdecruse_desc}_e{config.epoch_def}_es{config.earlystop_patience}_emin{config.min_epochs_before_es}"  # everything that is not in the main desc which has vgg16, 6 class, and no aug
#         tracker = config.trackpaths_final
#         tracker_name = config.desc  # SHOULDNT NEED THIS-ITS JUST FOR WB
#     print(append_details)

#     # for build
#     # full model details based on fold and hyperparams
#     # slightly different for nested cv or not
#     if config.kfoldnested == True:

#         resultsbase = "/home/csutter/DRIVE/dot/model_results"
#         accloss = f"{resultsbase}/acc_loss_curves_{tracker_w_details}.png"
#         model_results_txt = f"{resultsbase}/stats_{tracker_w_details}.txt"
#         metrics_csv = f"{resultsbase}/metricsreport_{tracker_w_details}.csv"
#         metrics_png = f"{resultsbase}/metricsreport_{tracker_w_details}.png"
#         cmsave = f"{resultsbase}/confusion_matrix_final_{tracker_w_details}.png"

#         savemodelfiles = "/home/csutter/DRIVE/dot/model_json"
#         savejson = f"{savemodelfiles}/{tracker_w_details}.json"
#         saveweights = f"{savemodelfiles}/{tracker_w_details}.h5"
#         savemodelto = f"{savemodelfiles}/{tracker_w_details}"  # Dir, this is the method being used

#         losstitle = f"Categorical Cross Entropy Loss - {tracker_w_details}"
#         acctitle = f"Classification Accuracy - {tracker_w_details}"

#     else:
#         accloss = f"{config.acc_loss_curves[:-4]}_{append_details}.png"
#         savejson = f"{config.model_json[:-5]}_{append_details}.json"
#         saveweights = f"{config.model_weights[:-3]}_{append_details}.h5"
#         savemodelto = f"{config.model_savedir}_{append_details}"
#         losstitle = f"Categorical Cross Entropy Loss - {append_details}"
#         acctitle = f"Classification Accuracy - {append_details}"

#         # for results
#         model_results_txt = f"{config.model_results_txt[:-4]}_{append_details}.txt"  # what is this used for?
#         # print(f"{config.metrics[:-4]}{append_details}.csv")
#         metrics_csv = f"{config.metrics[:-4]}_{append_details}.csv"  # f"{config.metrics[:-4]}_{append_details}.csv", #"make this the primary summary df to table
#         print(metrics_csv)
#         metrics_png = f"{config.metrics_png[:-4]}_{append_details}.png"  # probably dont need if we make the one above
#         # print("CHECK HERE CM FIRST, RIGHT BELOW!")
#         # print(f"{config.confusion_matrix_final[:-4]}{append_details}.png")
#         cmsave = f"{config.confusion_matrix_final[:-4]}{append_details}.png"
#         # print(cmsave)

#     print("HERE315 SEE IF FILENAMING WORKED")
#     print(tracker)
#     print(savemodelto)
#     print(accloss)

#     wb_config_build_run = {
#         "architecture": arch_input,
#         "transfer_learning": config.transfer_learning,
#         "baseline": config.baseline,
#         "desc": config.desc,
#         "kfoldflag": config.kfoldflag,
#         "kfolds_total": config.kfold,
#         "kfold_includetest": config.kfoldtestuse,
#         "kfold_num": split,
#         "results_on_val": config.samedata_tf,
#         "hyperparameter_tuning": config.hyptune_model_run,
#         "hyperparameter_tuning_round2": config.hyptune_round2,
#         "total_num_of_hyperparam_sets": config.hyptune_sets_per_split,
#         "hyperparam_set_running": hypset_on,
#         "l2_weight": l2use_desc,
#         "dropout_rate": dropoutuse_desc,
#         "batch_size": batchuse_desc,
#         "category_dirs": config.category_dirs,
#         "cat_outputs": config.cat_outputs,
#         "epochs": config.epoch_def,
#         "earlystop_patience": config.earlystop_patience,
#         "epochs_min": config.min_epochs_before_es,
#         "padding_def": config.padding_def,
#         "activation_layer_def": config.activation_layer_def,
#         "activation_output_def": config.activation_output_def,
#         # define if using learning rate optimization
#         "lr_opt": config.lr_opt,
#         "lr_init": lrinituse_desc,
#         "lr_decayrate": lrdecruse_desc,
#         "lr_after_num_of_epoch": config.lr_after_num_of_epoch,
#         "train_rot": config.train_rot,
#         "train_flip": config.train_flip,
#         "train_fill": config.train_fill,
#         "train_brightness": config.train_brightness,
#         # "train_colormode": config.train_colormode,  # UPDATE GRAYSCALE V RGB
#         "val_rot": config.val_rot,
#         "val_flip": config.val_flip,
#         "val_fill": config.val_fill,
#         "val_brightness": config.val_brightness,
#         # "val_colormode": config.val_colormode,
#         "colormode": config.colormode,
#         "imgsize_height_width": (config.imheight, config.imwidth),
#         "hyperparameter_tuning_round2": config.hyptune_round2,
#         "using_valdata": config.samedata_tf,
#         "using_otherdata": config.other_data_model,
#         "augment_imgs": config.use_gen_aug,
#         "lr_init": lrinituse_desc,
#         "lr_decayrate": lrdecruse_desc,
#         # "shuffle": config.shuff,
#         # "top_arch_spec": config.top_arch_spec,
#     }

#     if config.build == True:
#         print("Starting model build...")
#         print(savejson)
#         print(saveweights)
#         print(savemodelto)
#         src.build_compile.build(
#             label_dict_keyIs_catString,
#             config.kfoldflag,
#             split,
#             accloss,
#             savejson,
#             saveweights,
#             savemodelto,
#             losstitle,
#             acctitle,
#             l2use,
#             dropoutuse,
#             batchuse,
#             tracker=tracker,
#             tracker_name=tracker_name,
#             append_details=append_details,
#             wb_config=wb_config_build_run,
#             # wb_savemodel_art=wb_save_model_artifact,
#             model_detail=f"{config.desc}_{append_details}",
#             lr_init=lrinituse,
#             lr_decr=lrdecruse,
#             arch_input=arch_input,
#         )

#     if config.results == True:
#         (
#             model_load,
#             data_images,
#             datause_tfdataset,  # this may break with the previously used datause_iterator
#             class_labels,
#             predictions,
#             predicted_classes,
#             predicted_classes_string,
#             true_classes,
#             titledata,
#             # labelsdictionary,
#             evid_evidence,  # added for evid MILES GUESS, and 999s o/w (meaningless)
#             evid_ale,  # sqrt of ale. added for evid MILES GUESS, and 999s o/w (meaningless)
#             evid_epi,  # sqrt of epi. added for evid MILES GUESS, and 999s o/w (meaningless)
#             evid_total,  # sqrt of sum of ale and epi (sum of those two returned values wont add bc inside sqrt)
#         ) = src.load_and_evaluate.load_model_data_predictions(
#             "results",
#             label_dict_keyIs_catString,
#             label_dict_keyIs_catNumber,
#             config.kfoldflag,
#             split,
#             savejson,
#             saveweights,
#             savemodelto,
#         )
#         print("print here 1212!")
#         print(len(predicted_classes))
#         print(predicted_classes[0:5])
#         print(len(true_classes))
#         print(true_classes[0:5])
#         print(len(predicted_classes == true_classes))
#         print(list(predicted_classes == true_classes)[0:5])
#         print(sum(list(predicted_classes == true_classes)))

#         # if (config.results == True):
#         src.results_compile.results(
#             true_classes=true_classes,  # here add 1212
#             predicted_classes=predicted_classes,  # here add 1212
#             class_labels=class_labels,  # here add 1212
#             labelsdictionary=label_dict_keyIs_catString,  # here add 1212 # needs to be one of the new dicts made 121224
#             titledata=titledata,
#             kfolduse=config.kfoldflag,
#             foldnumber=split,
#             modeljson=savejson,
#             modelweights=saveweights,
#             model_results_txt=model_results_txt,
#             metrics_csv=metrics_csv,
#             metrics_png=metrics_png,
#             cmsave=cmsave,
#             title_perfmetrics="Performance Metrics",
#             title_cm="Confusion Matrix",
#             modelsavedir=savemodelto,
#             wb_config=wb_config_build_run,
#             # wb_savemodel_art=wb_save_model_artifact,
#             model_detail=f"{config.desc}_{append_details}",
#         )
#     if config.predictions == True:
#         print("ENTERING PREDICTIONS")
#         predictions_run(
#             dictclassinput=label_dict_keyIs_catString,
#             dictclassinput_valuekey=label_dict_keyIs_catNumber,
#             kfolduse=config.kfoldflag,
#             foldnumber=split,
#             modeljson=savejson,
#             modelweights=saveweights,
#             modelsavedir=savemodelto,
#             savepreds=f"{config.predictions_datadir}_{append_details}",
#         )
