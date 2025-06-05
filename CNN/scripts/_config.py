
### Data
# See bottom of file -- update the list of trackers
# designate column names from which to grab data (from the CSV files)
observation_col = "placeholder"# a differentating ID between each observation 
image_path_col = "placeholder"
fold_col = "placeholder"
label_col = "placeholder"# the hand-labeled, "true", classification of the observation

### Path where results should be saved
model_path = "/home/csutter/DRIVE-clean/CNN/data_models"
preds_path = "/home/csutter/DRIVE-clean/CNN/data_preds"

### Flags and specifics for the type of model run
train_flag = False # if running model training
eval_flag = False # if running model evaluation (need to have already trained models)
summary_flag = True
# one-off run where you give it one specific architecture and set of hyperparams to use
exp_desc = "nestcv_5cat_twotrain" # identifier string that all 30 trackers (trackers_list below) have in common for a given experiment, e.g. nestcv_5cat_twotrain. This is used in results_summaries to aggregate across multiple models that come from the same base experiment
one_off = True
adhoc_desc = "" # Default to empty string. Used as a desc "_test" ad hoc to differentiate a test code run, added to file naming. 
arch_set = "mobilenet"# ignored if one_off is False
transfer_learning = True 
ast = False # used if transfer_learning is True. Set to ast True if using an architecture specific top, otherwise set to False and will use generic top of architecture
l2_set = 1e-05# ignored if one_off is False
dr_set = 0.4 # ignored if one_off is False
# hyperparameter tuning
hyp_run = True
hyp_path = "placeholder"# path to CSV which has the list of hyperparameters
arch_col = "placeholder"# column name in the csv that corresponds to the architecture
l2_col = "placeholder"# column name in the csv that corresponds to the l2 rate
dr_col = "placeholder"# column name in the csv that corresponds to the dropout rate
activation_layer_def = "relu"
activation_output_def = "softmax"

epoch_set = 75 # default 75
earlystop_patience = 10  # default 10
min_epochs_before_es = 30 

# define if using learning rate optimizationop
lr_opt = True
lr_init = 0.01
lr_after_num_of_epoch = 1
lr_decayrate = 0.99
momentum = 0.9

### Static details: e.g. loss function, early stopping, min number of epochs, and learning rate, are the same for all runs
TARGET_SIZE = (224, 224)  # Adjust based on model needs
BATCH_SIZE = 32  # Adjust as needed
imheight = 224
imwidth = 224
class_wts = "yes"
class_balance = True # if goal is to use balanced importance across all classes, set this to true. Otherwise, set the the specific importance with the list below
setclassimportance = [] # sum to 1, used only if class_balance is False
category_dirs = [
    "wet",
    "dry",
    "snow",
    "snow_severe",
    "poor_viz",
]
# number of cats:
cat_num = 5




# Evidential deep learning (COME BACK TO)
# added MILES GUESS flag
evid = False
evid_output_activation = "softmax"  # should be linear
evid_annealing_coeff = 20  # 1.5 matches e.g. in https://github.com/ai2es/miles-guess/blob/main/mlguess/keras/models.py rather than  34.5 in their config files
# evid_optimizer = "adam"
evid_lr_init = 0.00001  # 0.0027750619126744817

trackers_list = [
    "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT5_m0_T0V1.csv",
    "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT4_m3_T5V0.csv",
    "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT5_m4_T4V0.csv",
    "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT3_m4_T2V4.csv",
    "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT5_m3_T3V4.csv",
    "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT4_m0_T0V1.csv",
    "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT2_m0_T0V1.csv",
    "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT5_m2_T2V3.csv",
    "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT0_m3_T4V5.csv",
    "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT0_m4_T5V1.csv",
    "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT1_m3_T5V0.csv",
    "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT0_m2_T3V4.csv",
    "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT4_m2_T2V3.csv",
    "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT3_m1_T1V2.csv",
    "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT1_m2_T4V5.csv",
    "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT2_m1_T3V4.csv",
    "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT0_m1_T2V3.csv",
    "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT4_m4_T3V5.csv",
    "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT2_m4_T1V3.csv",
    "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT2_m2_T4V5.csv",
    "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT1_m4_T0V2.csv",
    "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT3_m3_T5V0.csv",
    "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT3_m0_T0V1.csv",
    "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT5_m1_T1V2.csv",
    "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT1_m1_T3V4.csv",
    "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT2_m3_T5V0.csv",
    "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT4_m1_T1V2.csv",
    "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT1_m0_T2V3.csv",
    "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT3_m2_T4V5.csv",
    "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT0_m0_T1V2.csv",
]