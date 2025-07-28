directory_preds = "/home/csutter/DRIVE-clean/ensembling/data_preds"
directory_summaries = "/home/csutter/DRIVE-clean/ensembling/data_results"

# directory where all the models' predictions are already saved
dir_with_models = "/home/csutter/DRIVE-clean/calibration/calib_downstream_data"
subset_files_torun = True # designate which models to run ensembling for, since this dir_with_models conatins multiple runs. Set identifiers below.
subset_string = "A_resnet_TRLETrue_ASTTrue_L20_1_DR0_2_E75_AugTrue"
subset_downstream = "{'max_depth': 10, 'max_samples': 0.5, 'n_estimators': 300, 'max_features': 3, 'min_samples_leaf': 5, 'bootstrap': True}"

ensemble_flag = True

ensemble_summary = True

# set strings for saving and loading data
# some description to differentiate
desc_of_modelflow = "final_selection_resnetTTT" 
