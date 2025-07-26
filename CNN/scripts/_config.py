### Data
# See bottom of file -- update the list of trackers

### Path where results should be saved
#HERE!!
model_path = "/home/csutter/DRIVE-clean/ODM/data_models"
# "/home/csutter/DRIVE-clean/CNN/data_models"
# "/home/csutter/DRIVE-clean/CNN/data_models_side_experiments"
preds_path = "/home/csutter/DRIVE-clean/ODM/data_preds"
results_path = "/home/csutter/DRIVE-clean/ODM/data_results"
# side experiments to append to the above:
# _expOneTrain, _expShuffle, _expShuffleAndHalved, _expHalved
# e.g. "/home/csutter/DRIVE-clean/CNN/data_models_expShuffle"

### Flags and specifics for the type of model run
train_flag = False # if running model training
eval_flag = True # if running model evaluation (need to have already trained models). Need to then set one or the other for the two eval flags below
eval_highlevel = False # For high level summaries where one experiment is summarized as one line of results in results_path (e.g. BL runs, HT runs)
eval_pred_csvs = True # For after selecting best models, save the predictions out for each tracker as csvs. Set which models to run by setting hyp_path (defined below) to the set of 1 or more models for which saving out preds. Note that evaluation/predictions aren't set up to do a one-off so have to rely on the HT csv to just be the one model that we want to use. 
wandb_flag = False # flag for whether to save experiments to w&b
# one-off run where you give it one specific architecture and set of hyperparams to use
wanb_projectname = "DRIVE-clean" # HERE!! for pure BL or HT runs, "DRIVE-clean", o/w adjust here also for adhoc_desc "DRIVE-side_experiments"
exp_desc = "ODM"  #HERE!!
# "nestcv_5cat_twotrain" <--- this is the main one used from BL/HT
# identifier string that all 30 trackers (trackers_list below) have in common for a given experiment, e.g. nestcv_5cat_twotrain. This is used in results_summaries to aggregate across multiple models that come from the same base experiment, and also for logging to w&b.
# Should be used for all experime`≥ntsq, one_off and hyp_run
adhoc_desc = "" # Default to empty string. Used as a desc "_TEST" "_SaveWeightsOnly" ad hoc to differentiate a test code run, added to file naming. 
one_off = True # one-off runs only set up for training (not for eval/predictions, for that, have to refer to HT csv). See below about running inference on a different dataset than the ones used for model dev trackers
arch_set = "resnet" # ignored if one_off is False
transfer_learning = True # ignored if one_off is False
ast = True # used if transfer_learning is True. Set to ast True if using an architecture specific top, otherwise set to False and will use generic top of architecture
aug = True # ignored if one_off is False
l2_set = 0.1 # ignored if one_off is False, sel: 1e-05
dr_set = 0.2 # ignored if one_off is False, sel: 0.4
# hyperparameter tuning -- must use this for evaluation/predictions
hyp_run = False
hyp_path = "/home/csutter/DRIVE-clean/CNN/data_trackers/HT_hyperparams_oneoff.csv" #HERE!! path to CSV which has the list of hyperparameters
# Baseline runs: "/home/csutter/DRIVE-clean/CNN/data_trackers/baseline_hyperparams.csv" 
# Hyperparameter tuning runs: "/home/csutter/DRIVE-clean/CNN/data_trackers/HT_hyperparams.csv" 
# Selected HTs and models, i.e. the 4 final chosen from model selection:
# "/home/csutter/DRIVE-clean/CNN/data_trackers/HT_hyperparams_selected.csv" 
# For side experiments (for manuscript), only running the model and HTs that were the final selection from the full model flow, use this csv below:
# "/home/csutter/DRIVE-clean/CNN/data_trackers/HT_BL_and_hyperparam_sideExperiments.csv"
# If need to run a one-off couple runs (e.g., backfill experiments that didn't run) "/home/csutter/DRIVE-clean/CNN/data_trackers/HT_hyperparams_oneoff.csv". 
activation_layer_def = "relu"
activation_output_def = "softmax"

epoch_set = 75 # default 75
earlystop_patience = 10  # default 10
min_delta = 0.005 # e.g. need to improve by more than 63.0 to 63.5 over 10 epoch
min_epochs_before_es = 15 # min_epochs + patience = min # of total epochs ran

# define if using learning rate optimizationop
lr_opt = True
lr_init = 0.01 # 0.01
lr_init_small = 0.001 # used for vgg16, or any other archs that require smaller LR if exploding gradients (nan loss values). For now only vgg16 shows this. The lr is set accordingly in call_model_train.py
lr_after_num_of_epoch = 1
lr_decayrate = 0.95#0.95 
momentum = 0.25#0.25

### Static details: e.g. loss function, early stopping, min number of epochs, and learning rate, are the same for all runs
TARGET_SIZE = (224, 224)  # Adjust based on model needs
BATCH_SIZE = 128  # 128 Adjust as needed
imheight = 224
imwidth = 224
class_wts = "yes"
class_balance = True # if goal is to 
# use balanced importance across all classes, set this to true. Otherwise, set the the specific importance with the list below
setclassimportance = [] # sum to 1, used only if class_balance is False
category_dirs = ["obs","nonobs"]
# For SCM
# [
#     "wet",
#     "dry",
#     "snow",
#     "snow_severe",
#     "poor_viz",
# ]

# for ODM
# ["obs","nonobs"]
# number of cats:
cat_num = 2 # 5 for SCM, 2 for ODM




# Evidential deep learning (COME BACK TO)
# added MILES GUESS flag
evid = False
evid_output_activation = "softmax"  # should be linear
evid_annealing_coeff = 20  # 1.5 matches e.g. in https://github.com/ai2es/miles-guess/blob/main/mlguess/keras/models.py rather than  34.5 in their config files
# evid_optimizer = "adam"
evid_lr_init = 0.00001  # 0.0027750619126744817

# If evaluation/inference/predictions is being done on data that is different than the models. E.g. evaluate all ODM data (all observations across the different samples) on each of the 6 ODM models. 
# This is only set up to work for one-off, eval_flag and eval_pred_csvs
inference_other = True #HERE!!
inference_data_csv = "/home/csutter/DRIVE-clean/trackers_ODM/ynobs_entire.csv"

# HERE!! 
trackers_list = ["/home/csutter/DRIVE-clean/trackers_ODM/ynobs_A_split0.csv",
"/home/csutter/DRIVE-clean/trackers_ODM/ynobs_A_split1.csv",
"/home/csutter/DRIVE-clean/trackers_ODM/ynobs_B_split0.csv",
"/home/csutter/DRIVE-clean/trackers_ODM/ynobs_B_split1.csv",
"/home/csutter/DRIVE-clean/trackers_ODM/ynobs_C_split0.csv",
"/home/csutter/DRIVE-clean/trackers_ODM/ynobs_C_split1.csv"]

# # Main trackers list
# trackers_list = [
#     "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT5_m0_T0V1.csv",
#     "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT4_m3_T5V0.csv",
#     "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT5_m4_T4V0.csv",
#     "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT3_m4_T2V4.csv",
#     "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT5_m3_T3V4.csv",
#     "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT4_m0_T0V1.csv",
#     "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT2_m0_T0V1.csv",
#     "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT5_m2_T2V3.csv",
#     "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT0_m3_T4V5.csv",
#     "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT0_m4_T5V1.csv",
#     "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT1_m3_T5V0.csv",
#     "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT0_m2_T3V4.csv",
#     "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT4_m2_T2V3.csv",
#     "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT3_m1_T1V2.csv",
#     "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT1_m2_T4V5.csv",
#     "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT2_m1_T3V4.csv",
#     "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT0_m1_T2V3.csv",
#     "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT4_m4_T3V5.csv",
#     "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT2_m4_T1V3.csv",
#     "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT2_m2_T4V5.csv",
#     "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT1_m4_T0V2.csv",
#     "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT3_m3_T5V0.csv",
#     "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT3_m0_T0V1.csv",
#     "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT5_m1_T1V2.csv",
#     "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT1_m1_T3V4.csv",
#     "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT2_m3_T5V0.csv",
#     "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT4_m1_T1V2.csv",
#     "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT1_m0_T2V3.csv",
#     "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT3_m2_T4V5.csv",
#     "/home/csutter/DRIVE/dot/model_trackpaths/nestcv_5cat_twotrain_OT0_m0_T1V2.csv",
# ]

### shuffleHalved
# trackers_list = ['/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle_halved/nestcv_5cat_twotrain_OT5_m0_T0V1.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle_halved/nestcv_5cat_twotrain_OT4_m3_T5V0.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle_halved/nestcv_5cat_twotrain_OT5_m4_T4V0.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle_halved/nestcv_5cat_twotrain_OT3_m4_T2V4.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle_halved/nestcv_5cat_twotrain_OT5_m3_T3V4.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle_halved/nestcv_5cat_twotrain_OT4_m0_T0V1.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle_halved/nestcv_5cat_twotrain_OT2_m0_T0V1.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle_halved/nestcv_5cat_twotrain_OT5_m2_T2V3.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle_halved/nestcv_5cat_twotrain_OT0_m3_T4V5.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle_halved/nestcv_5cat_twotrain_OT0_m4_T5V1.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle_halved/nestcv_5cat_twotrain_OT1_m3_T5V0.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle_halved/nestcv_5cat_twotrain_OT0_m2_T3V4.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle_halved/nestcv_5cat_twotrain_OT4_m2_T2V3.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle_halved/nestcv_5cat_twotrain_OT3_m1_T1V2.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle_halved/nestcv_5cat_twotrain_OT1_m2_T4V5.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle_halved/nestcv_5cat_twotrain_OT2_m1_T3V4.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle_halved/nestcv_5cat_twotrain_OT0_m1_T2V3.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle_halved/nestcv_5cat_twotrain_OT4_m4_T3V5.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle_halved/nestcv_5cat_twotrain_OT2_m4_T1V3.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle_halved/nestcv_5cat_twotrain_OT2_m2_T4V5.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle_halved/nestcv_5cat_twotrain_OT1_m4_T0V2.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle_halved/nestcv_5cat_twotrain_OT3_m3_T5V0.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle_halved/nestcv_5cat_twotrain_OT3_m0_T0V1.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle_halved/nestcv_5cat_twotrain_OT5_m1_T1V2.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle_halved/nestcv_5cat_twotrain_OT1_m1_T3V4.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle_halved/nestcv_5cat_twotrain_OT2_m3_T5V0.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle_halved/nestcv_5cat_twotrain_OT4_m1_T1V2.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle_halved/nestcv_5cat_twotrain_OT1_m0_T2V3.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle_halved/nestcv_5cat_twotrain_OT3_m2_T4V5.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle_halved/nestcv_5cat_twotrain_OT0_m0_T1V2.csv']

# Shuffle
# trackers_list = ['/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle/nestcv_5cat_twotrain_OT5_m0_T0V1.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle/nestcv_5cat_twotrain_OT4_m3_T5V0.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle/nestcv_5cat_twotrain_OT5_m4_T4V0.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle/nestcv_5cat_twotrain_OT3_m4_T2V4.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle/nestcv_5cat_twotrain_OT5_m3_T3V4.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle/nestcv_5cat_twotrain_OT4_m0_T0V1.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle/nestcv_5cat_twotrain_OT2_m0_T0V1.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle/nestcv_5cat_twotrain_OT5_m2_T2V3.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle/nestcv_5cat_twotrain_OT0_m3_T4V5.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle/nestcv_5cat_twotrain_OT0_m4_T5V1.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle/nestcv_5cat_twotrain_OT1_m3_T5V0.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle/nestcv_5cat_twotrain_OT0_m2_T3V4.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle/nestcv_5cat_twotrain_OT4_m2_T2V3.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle/nestcv_5cat_twotrain_OT3_m1_T1V2.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle/nestcv_5cat_twotrain_OT1_m2_T4V5.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle/nestcv_5cat_twotrain_OT2_m1_T3V4.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle/nestcv_5cat_twotrain_OT0_m1_T2V3.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle/nestcv_5cat_twotrain_OT4_m4_T3V5.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle/nestcv_5cat_twotrain_OT2_m4_T1V3.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle/nestcv_5cat_twotrain_OT2_m2_T4V5.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle/nestcv_5cat_twotrain_OT1_m4_T0V2.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle/nestcv_5cat_twotrain_OT3_m3_T5V0.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle/nestcv_5cat_twotrain_OT3_m0_T0V1.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle/nestcv_5cat_twotrain_OT5_m1_T1V2.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle/nestcv_5cat_twotrain_OT1_m1_T3V4.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle/nestcv_5cat_twotrain_OT2_m3_T5V0.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle/nestcv_5cat_twotrain_OT4_m1_T1V2.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle/nestcv_5cat_twotrain_OT1_m0_T2V3.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle/nestcv_5cat_twotrain_OT3_m2_T4V5.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/shuffle/nestcv_5cat_twotrain_OT0_m0_T1V2.csv']

### Onetrain
# trackers_list = ['/home/csutter/DRIVE-clean/trackers_side_experiments/onetrain/nestcv_5cat_onetrain_OT5_m1_T1V2.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/onetrain/nestcv_5cat_onetrain_OT4_m1_T1V2.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/onetrain/nestcv_5cat_onetrain_OT5_m0_T0V1.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/onetrain/nestcv_5cat_onetrain_OT0_m1_T2V3.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/onetrain/nestcv_5cat_onetrain_OT0_m4_T5V1.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/onetrain/nestcv_5cat_onetrain_OT4_m2_T2V3.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/onetrain/nestcv_5cat_onetrain_OT3_m0_T0V1.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/onetrain/nestcv_5cat_onetrain_OT0_m2_T3V4.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/onetrain/nestcv_5cat_onetrain_OT0_m0_T1V2.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/onetrain/nestcv_5cat_onetrain_OT4_m0_T0V1.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/onetrain/nestcv_5cat_onetrain_OT1_m2_T4V5.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/onetrain/nestcv_5cat_onetrain_OT3_m2_T4V5.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/onetrain/nestcv_5cat_onetrain_OT5_m2_T2V3.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/onetrain/nestcv_5cat_onetrain_OT3_m4_T2V4.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/onetrain/nestcv_5cat_onetrain_OT2_m3_T5V0.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/onetrain/nestcv_5cat_onetrain_OT2_m4_T1V3.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/onetrain/nestcv_5cat_onetrain_OT0_m3_T4V5.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/onetrain/nestcv_5cat_onetrain_OT2_m1_T3V4.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/onetrain/nestcv_5cat_onetrain_OT3_m1_T1V2.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/onetrain/nestcv_5cat_onetrain_OT5_m4_T4V0.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/onetrain/nestcv_5cat_onetrain_OT1_m3_T5V0.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/onetrain/nestcv_5cat_onetrain_OT1_m1_T3V4.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/onetrain/nestcv_5cat_onetrain_OT1_m4_T0V2.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/onetrain/nestcv_5cat_onetrain_OT5_m3_T3V4.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/onetrain/nestcv_5cat_onetrain_OT1_m0_T2V3.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/onetrain/nestcv_5cat_onetrain_OT4_m3_T5V0.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/onetrain/nestcv_5cat_onetrain_OT4_m4_T3V5.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/onetrain/nestcv_5cat_onetrain_OT2_m2_T4V5.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/onetrain/nestcv_5cat_onetrain_OT3_m3_T5V0.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/onetrain/nestcv_5cat_onetrain_OT2_m0_T0V1.csv']

### Halved
# trackers_list = ['/home/csutter/DRIVE-clean/trackers_side_experiments/halved/nestcv_5cat_twotrain_OT5_m0_T0V1.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/halved/nestcv_5cat_twotrain_OT4_m3_T5V0.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/halved/nestcv_5cat_twotrain_OT5_m4_T4V0.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/halved/nestcv_5cat_twotrain_OT3_m4_T2V4.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/halved/nestcv_5cat_twotrain_OT5_m3_T3V4.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/halved/nestcv_5cat_twotrain_OT4_m0_T0V1.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/halved/nestcv_5cat_twotrain_OT2_m0_T0V1.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/halved/nestcv_5cat_twotrain_OT5_m2_T2V3.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/halved/nestcv_5cat_twotrain_OT0_m3_T4V5.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/halved/nestcv_5cat_twotrain_OT0_m4_T5V1.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/halved/nestcv_5cat_twotrain_OT1_m3_T5V0.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/halved/nestcv_5cat_twotrain_OT0_m2_T3V4.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/halved/nestcv_5cat_twotrain_OT4_m2_T2V3.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/halved/nestcv_5cat_twotrain_OT3_m1_T1V2.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/halved/nestcv_5cat_twotrain_OT1_m2_T4V5.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/halved/nestcv_5cat_twotrain_OT2_m1_T3V4.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/halved/nestcv_5cat_twotrain_OT0_m1_T2V3.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/halved/nestcv_5cat_twotrain_OT4_m4_T3V5.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/halved/nestcv_5cat_twotrain_OT2_m4_T1V3.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/halved/nestcv_5cat_twotrain_OT2_m2_T4V5.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/halved/nestcv_5cat_twotrain_OT1_m4_T0V2.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/halved/nestcv_5cat_twotrain_OT3_m3_T5V0.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/halved/nestcv_5cat_twotrain_OT3_m0_T0V1.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/halved/nestcv_5cat_twotrain_OT5_m1_T1V2.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/halved/nestcv_5cat_twotrain_OT1_m1_T3V4.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/halved/nestcv_5cat_twotrain_OT2_m3_T5V0.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/halved/nestcv_5cat_twotrain_OT4_m1_T1V2.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/halved/nestcv_5cat_twotrain_OT1_m0_T2V3.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/halved/nestcv_5cat_twotrain_OT3_m2_T4V5.csv',
# '/home/csutter/DRIVE-clean/trackers_side_experiments/halved/nestcv_5cat_twotrain_OT0_m0_T1V2.csv']
