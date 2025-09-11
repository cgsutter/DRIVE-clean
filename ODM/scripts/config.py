# Training for ynobs models is done in the CNN dir
# For training, just need to have set up trackers with innerPhase col designated for 0 and 1 (even though not using nested CV, just keep with the same phase column name). Update the config in the CNN dir accordingly to ODM.
# For predictions, also use the CNN dir (set the inference csv, and see next note related to this)
# For predictions, need to have one tracker set up  (see tracker_ODM/ dir) for which all the models will be evaluated with that one tracker of data. Foldnum and phases are irrelevant there, since it's just predictions. Foldnum will matter when doing ensembling step, which is what is in this config and /home/csutter/DRIVE-clean/ODM/scripts/_run_ODM.py

# runAllVal = False
# Updates needed Only HERE!!
# set flags on parts A and B
# part A does the ensembling for each outertest (main code!)s
partA_ensemble = True
# part B just creates one brief summary file (should be run after part A)
partB_summary = True

# set strings for saving and loading data
desc_of_modelflow = "ynobs_ensemble_eval5catdata"  #
dir_with_models = "/home/csutter/DRIVE-clean/ODM/data_preds_eval5catdata"
files_subset = "ynobs"  # doesnt matter for data_preds_eval5catdata bc only ran the one set of full dataset, just use some string that exists in the files
# "ynobs_entire" # file_subset to designate to run on ynobs_entire which includes all data samples (which is needed for ensembling), and runs for each of the 6 models (rather than running everything in that dir which also includes the splits which are all different)

dir_save_ensemble = f"/home/csutter/DRIVE-clean/ODM/ensemble_results"  # generally don't need to change since will be saved as a fn of the desc_of_modelflow

runYNObsVal = True  # this can ONLY be run AFTER runAllVal and partA and partB have been run. Have to work from the runAllVal dfs that are saved out because the nonobs examples from the different A-E models do not have overlap bc they were different random samples (which was the whole point) -- thus, to grab the A-E model preds for all the nonobs examples, need to use the allVal run and simply subset to df['in_ynobs'] = "included", and simply repeat partB summary for that subsetted df.
