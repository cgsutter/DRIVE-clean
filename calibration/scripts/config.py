dir_of_datacsvs =  "/home/csutter/DRIVE-clean/downstream/data_preds"
# "/home/csutter/DRIVE-clean/CNN/data_preds"
# this is a directory containing all csvs, for which calibration will be done on each one. Each csv represents data that is used to train/eval the calibration. The csvs stem from previous classification model outputs, so they are the predictions from multiple splits and models. Each one require calibration (as a new calib model).

#####  SET variables and then the right chunk of code will run
calib_model_type = "isotonic"  # "isotonic"

classif_model = "downstream" # "CNN" or "downstream" or 'fcstOnly". A dir will be made to save the data and models to according to this classif model

# IF running calibrate_fcstOnly, MUST also set these below
# Note that the code is not set up for this yet.
# So will run this script for 3 final combinations (since we use fh2 model for fhs 2-6, fh12 for fhs 9-15, and fh18 o/w)
modelforeval = "FH2"  # or "FH2", "FH12"
datatoeval = [
    2
]  # not sure there was ever need to calibrate the other eval data! Since calibration is for nowcasting needs
# [2,3,4,5,6] or [9, 12, 15] or[18, 24, 30, 36, 42, 48] HERE! for what data to be evaluating on


