directory_preds = "/home/csutter/DRIVE-clean/ensembling/data_preds"
directory_summaries = "/home/csutter/DRIVE-clean/ensembling/data_results"

# directory where all the models' predictions are already saved
dir_with_models = "/home/csutter/DRIVE-clean/calibration/calib_downstream_data"
subset_files_torun = True # if running calib every file in the dir_of_datacsvs, set this to False. Otherwise, if running for one specific file (e.g. _A_resnet_TRLETrue_ASTFalse_L20_1_DR0_2_E75_AugFalse), subset to just those files to run calib for (and accordingly, if set to True, set the subset string below)
subset_string = "A_resnet_TRLETrue_ASTFalse_L20_1_DR0_4_E75_AugTrue"
# A_resnet_TRLETrue_ASTFalse_L20_1_DR0_4_E75_AugTrue
ensemble_flag = True

ensemble_summary = True

# set strings for saving and loading data
# some description to differentiate
desc_of_modelflow = "alternateselection" 
# see /home/csutter/DRIVE-clean/downstream/scripts/config.py for methods and how they map to downstream selection
# note, if running other variations, need to run 1) CNN calibration on the selected CNN model 2) train downstream 3) calibrate downstream (subset to those with the model name in it since it's one dir w everything in there) 4) ensemble (again the subset note above)


