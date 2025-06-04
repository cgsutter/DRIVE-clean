import config

def prep_str_details_track(
    tracker_designated,
    arch_input=config.arch_set,
    epochs = config.epoch_set,
    # earlystop = config.earlystop_patience,
    # minepoch = config.min_epochs_before_es,
    # batchuse = config.BATCH_SIZE,
    # lrinituse = config.lr_init,
    # lrdecruse = config.lr_decayrate,
    l2use = config.l2_set,
    dropoutuse = config.dr_set,
    transfer_learning = config.transfer_learning,
    ast = config.ast
    ):

    # make string descriptions of hyperparameters for file naming
    l2use_desc = str(l2use).replace(".", "_")
    dropoutuse_desc = str(dropoutuse).replace(".", "_")
    # batchuse_desc = str(batchuse).replace(".", "_")
    # lrinituse_desc = str(lrinituse).replace(".", "_")
    # lrdecruse_desc = str(lrdecruse).replace(".", "_")

    begintracker = tracker_designated.rfind("/")
    tracker_filebase = tracker_designated[
        begintracker + 1 : -4
    ]  # the base name that differentiates dataset, just remove .csv extension

    tracker_rundetails = f"_A_{arch_input}_TRLE{transfer_learning}_AST{ast}_L2{l2use_desc}_DR{dropoutuse_desc}_E{epochs}"
    # f"_A_{arch_input}_TL{transfer_learning}_AST{ast}_{l2use_desc}_DR{dropoutuse_desc}_B{batchuse_desc}_LR{lrinituse_desc}_LRD{lrdecruse_desc}_E{epochs}_ES{earlystop}_MIN{minepoch}"

    tracker_w_details = f"{tracker_filebase}" # everything that is not in the main desc which has vgg16, 6 class, and no aug # used for saving out model and results with model details):

    # wbtable = wandb.Table(
    #     columns=[
    #         "arch",
    #         "desc",
    #         "split",
    #         "model_split",
    #         "model_full",
    #         "details_only",
    #         "l2_weight",
    #         "dropout_rate",
    #         "learning_rate_initial",
    #         "learning_rate_decayrate",
    #         "batch_size",
    #         "val_acc",
    #         "train_acc",
    #         "val_loss",
    #         "train_loss",
    #         "epochs_ran",
    #     ],
    #     data=[
    #         [
    #             arch_input,
    #             config.desc,
    #             foldnumber,
    #             f"{config.desc}_split{foldnumber}",
    #             f"{config.desc}_{append_details}",
    #             append_details,
    #             l2_weight,
    #             dropout_rate,
    #             lr_init,
    #             lr_decr,
    #             batch_size,
    #             final_val_acc,
    #             final_train_acc,
    #             final_val_loss,
    #             final_train_loss,
    #             number_of_epochs_it_ran,
    #         ]
    #     ],
    # )

    return tracker_filebase, tracker_rundetails #, wbtable

def cat_str_ind_dictmap(listcats = config.category_dirs):
    cs = (
        listcats
    )  # ["wet", "dry", "snow", "snow_severe", "obs", "poor_viz"]

    cats_alphabetical = sorted(cs)

    # make a list of values 0 through 5 for 6-cats. Will be dynamic if cat length changes

    # print("place3")
    cat_inds = [i for i in range(0, len(cats_alphabetical))]

    dictreturn = dict(zip(cats_alphabetical, cat_inds))

    return dictreturn