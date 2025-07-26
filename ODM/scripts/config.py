
runAllVal = False
# Updates needed Only HERE!!
# set flags on parts A and B
# part A does the ensembling for each outertest (main code!)s
partA_ensemble = True  # should ONLY be true if runAllVal is true
# part B just creates one brief summary file (should be run after part A)
partB_summary = True  # should ONLY be true if runAllVal is true

# set strings for saving and loading data
desc_of_modelflow = "ynobs_ensemble"  # leave this the same for if running runAllVal and its corresponding runYNObsVal
dir_with_models = "/home/csutter/DRIVE-clean/ODM/data_preds"
files_subset = "ynobs_entire" # designate to run on ynobs_entire which includes all data samples (which is needed for ensembling), and runs for each of the 6 models

dir_save_ensemble = f"/home/csutter/DRIVE-clean/ODM/ensemble_results"  # generally don't need to change since will be saved as a fn of the desc_of_modelflow

runYNObsVal = True  # this can ONLY be run AFTER runAllVal and partA and partB have been run. Have to work from the runAllVal dfs that are saved out because the nonobs examples from the different A-E models do not have overlap bc they were different random samples (which was the whole point) -- thus, to grab the A-E model preds for all the nonobs examples, need to use the allVal run and simply subset to df['in_ynobs'] = "included", and simply repeat partB summary for that subsetted df.
