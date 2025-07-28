# this is a directory containing all csvs, for which calibration will be done on each one. Each csv represents data that is used to train/eval the calibration. The csvs stem from previous classification model outputs, so they are the predictions from multiple splits and models. Each one require calibration (as a new calib model).
dir_of_datacsvs = "/home/csutter/DRIVE-clean/downstream/data_predsFinal"
# "/home/csutter/DRIVE-clean/CNN/data_preds" <-- main one for calib CNN
# "/home/csutter/DRIVE-clean/downstream/data_preds" <-- main one for calib DS 
# For running side experiments: e.g. "/home/csutter/DRIVE-clean/side_experiments_data_and_models/CNN/data_preds_expOneTrain"


subset_files_torun = True # if running calib every file in the dir_of_datacsvs (above), set this to False. Otherwise, if running for one specific file (e.g. _A_resnet_TRLETrue_ASTFalse_L20_1_DR0_2_E75_AugFalse, or even further, one specific dowsntream model) then set this to True and set these identifier model information strings below
subset_string = "A_resnet_TRLETrue_ASTTrue_L20_1_DR0_2_E75_AugTrue"
subset_downstream = "{'max_depth': 10, 'max_samples': 0.5, 'n_estimators': 300, 'max_features': 3, 'min_samples_leaf': 5, 'bootstrap': True}"

#####  SET variables and then the right chunk of code will run
calib_model_type = "isotonic"  # "isotonic"

classif_model =  "downstreamFinal" #HERE!! "CNN" or "downstream" or 'fcstOnly" or for side experiments "CNNexpHalved", "CNNexpOneTrain". A dir will be made to save the data and models to according to this classif model


# Note that the code is not set up for this yet, but keeping placeholder in here for now
modelforeval = "FH2"  # or "FH2", "FH12"
datatoeval = [2]


