dir_of_datacsvs = "/home/csutter/DRIVE-clean/CNN/data_preds" #HERE!!
# "/home/csutter/DRIVE-clean/CNN/data_preds" <-- main one for calibrating after CNN
# "/home/csutter/DRIVE-clean/downstream/data_preds" <-- main one for calibrating after downstream model
# For running side experiments:
# /home/csutter/DRIVE-clean/CNN/data_preds_expHalved
# /home/csutter/DRIVE-clean/CNN/data_preds_expOneTrain
# /home/csutter/DRIVE-clean/CNN/data_preds_expShuffle
# /home/csutter/DRIVE-clean/CNN/data_preds_expShuffleAndHalved
# this is a directory containing all csvs, for which calibration will be done on each one. Each csv represents data that is used to train/eval the calibration. The csvs stem from previous classification model outputs, so they are the predictions from multiple splits and models. Each one require calibration (as a new calib model).

subset_files_torun = True # if running calib every file in the dir_of_datacsvs, set this to False. Otherwise, if running for one specific file (e.g. _A_resnet_TRLETrue_ASTFalse_L20_1_DR0_2_E75_AugFalse), subset to just those files to run calib for (and accordingly, if set to True, set the subset string below)
subset_string = "A_mobilenet_TRLETrue_ASTFalse_L20_1_DR0_4_E75_AugFalse"
# A_resnet_TRLETrue_ASTFalse_L20_1_DR0_4_E75_AugTrue


#####  SET variables and then the right chunk of code will run
calib_model_type = "isotonic"  # "isotonic"

classif_model =  "CNN" #HERE!! "CNN" or "downstream" or 'fcstOnly". A dir will be made to save the data and models to according to this classif model
# for side experiments for paper: CNNexpHalved, CNNexpOneTrain, CNNexpShuffle, CNNexpShuffleAndHalved
# "CNN_selecti87"

# IF running calibrate_fcstOnly, MUST also set these below
# Note that the code is not set up for this yet.
# So will run this script for 3 final combinations (since we use fh2 model for fhs 2-6, fh12 for fhs 9-15, and fh18 o/w)
modelforeval = "FH2"  # or "FH2", "FH12"
datatoeval = [
    2
]  # not sure there was ever need to calibrate the other eval data! Since calibration is for nowcasting needs
# [2,3,4,5,6] or [9, 12, 15] or[18, 24, 30, 36, 42, 48] HERE! for what data to be evaluating on


