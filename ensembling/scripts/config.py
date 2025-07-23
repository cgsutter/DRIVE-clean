directory_preds = "/home/csutter/DRIVE-clean/ensembling/data_preds"
directory_summaries = "/home/csutter/DRIVE-clean/ensembling/data_results"

# directory where all the models' predictions are saved
dir_with_models = "/home/csutter/DRIVE-clean/calibration/calib_downstream_data"

ensemble_flag = True

ensemble_summary = True

# set strings for saving and loading data
# some description to differentiate
desc_of_modelflow = "method_1" 
# see /home/csutter/DRIVE-clean/downstream/scripts/config.py for methods and how they map to downstream selection
# note, if running other variations, need to run 1) CNN calibration on the selected CNN model 2) train downstream 3) calibrate downstream (subset to those with the model name in it since it's one dir w everything in there) 4) ensemble (again the subset note above)


