# Portions of this code were writen with the assistance of AI tools (e.g., ChatGPT).

import _config as config
import os


def prep_basefile_str(tracker_designated):
    begintracker = tracker_designated.rfind("/")
    tracker_filebase = tracker_designated[
        begintracker + 1 : -4
    ]  # the base name that differentiates dataset, just remove .csv extension
    return tracker_filebase



def prep_str_details_track(
    # tracker_designated,
    arch_input=config.arch_set,
    epochs = config.epoch_set,
    l2use = config.l2_set,
    dropoutuse = config.dr_set,
    transfer_learning = config.transfer_learning,
    ast = config.ast,
    adhoc_desc = config.adhoc_desc,
    exp_desc = config.exp_desc,
    auguse = config.aug
    ):
    """
    This function grabs details for tracking information about the run, which is used for saving models with the right name, logging to w&b, etc. It is only unique at the experiment & arch/hyp level - not at the tracker level (e.g. all 30 trackers have the same information returned from this function). 

    This function:
    Returns a string that includes details of the run, which are NOT unique to the 30 trackers, they are only unique to the experiment + arch/hyp details. 
    Returns a dictionary that is used to log details to w&b (and then outside of this function, the 30 tracker-level differentiators are also added to the dict for tracking, but that does not happen within this function)
    """

    # make string descriptions of hyperparameters for file naming
    l2use_desc = str(l2use).replace(".", "_")
    dropoutuse_desc = str(dropoutuse).replace(".", "_")
    # batchuse_desc = str(batchuse).replace(".", "_")
    # lrinituse_desc = str(lrinituse).replace(".", "_")
    # lrdecruse_desc = str(lrdecruse).replace(".", "_")


    
    tracker_rundetails = f"_A_{arch_input}_TRLE{transfer_learning}_AST{ast}_L2{l2use_desc}_DR{dropoutuse_desc}_E{epochs}{adhoc_desc}"
    # f"_A_{arch_input}_TL{transfer_learning}_AST{ast}_{l2use_desc}_DR{dropoutuse_desc}_B{batchuse_desc}_LR{lrinituse_desc}_LRD{lrdecruse_desc}_E{epochs}_ES{earlystop}_MIN{minepoch}"


    dict_for_wb = {"exp_desc": exp_desc, "arch":arch_input,"transfer_learning":transfer_learning, "ast":ast, "aug":auguse, "epochs":epochs, "l2":l2use, "dropoutuse":dropoutuse, "adhoc_desc":adhoc_desc}
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

    return tracker_rundetails, dict_for_wb #, wbtable

def tracker_differentiator(trackerpath):

    filename = os.path.basename(trackerpath)

    tracker_identifier = filename.replace(f'{config.exp_desc}_', '').replace('.csv', '')

    return tracker_identifier

def cat_str_ind_dictmap(listcats = config.category_dirs):
    """
    This function returns a dictionary that maps the class name to the class index, used at various other points in the code
    """
    cs = (
        listcats
    )  # ["wet", "dry", "snow", "snow_severe", "obs", "poor_viz"]

    cats_alphabetical = sorted(cs)

    # make a list of values 0 through 5 for 6-cats. Will be dynamic if cat length changes

    # print("place3")
    cat_inds = [i for i in range(0, len(cats_alphabetical))]

    dict_catKey_indValue = dict(zip(cats_alphabetical, cat_inds))

    # for evaluation, when mapping from ind to class name, need the dict reversed
    dict_indKey_catValue = {value: key for key, value in dict_catKey_indValue.items()}

    return dict_catKey_indValue, dict_indKey_catValue