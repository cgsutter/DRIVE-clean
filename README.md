# DRIVE
Detection of Road Imagery in Variable Environments

In collaboration with the Department of Transportation, classification modeling is performed to predict roads surface conditions using convolutional neural networks and random forests. The Surface Condition Model developed in this repo predicts five weather-related road surface classes include: 1) severe snow, 2) snow, 3) wet, 4) dry, 5) poor visibility. Another classification task, the Obstruction Detection Model, predicts obstructed or non-obstructed, which are camera/image related issues, using solely camera data. All details about the underlying dataset, including the hand-labeling process following quantitative content analysis, are included here: https://zenodo.org/records/15257486

This repository contains scripts for model training, hyperparameter tuning, ensembling, results summaries, and inference for the Surface Condition Model. The data used to train the models include image data (for the first model, a CNN) and weather data (for the downstream model). 

The primary purpose of this repo and work is related to the Surface Condition Model. However, the Obstruction Detection Model (ODM), is also ran using the CNN and calibration directories (the same directories for the Surface Condition Model), but the remaining work for the ODM model, including ensembling, inference, and summaries, are included in the /ODM directory. 

Data preparation should be done prior to training models. CSV files, referred to as "trackers", should be set up to contain each unique observation with its corresponding image path, classification label, and weather data, as well as fold designations 0 through 5. Nested cross validation with 6 folds is used such that for each test dataset, 5-fold cross validation on the remaining 5 folds, resulting in 30 trackers that differ by fold designation. Since all modeling steps are repeated for each of the trackers, 30 sets of results/predictions (after each step of model training) are also saved out in CSV form. 

The basic modeling flow for the Surface Condition Model basic is as follows:
1. Train CNN using training dataset 1 (directory: /CNN)
2. Inference CNN to get predictions on dataset 2 (which serves as the training dataset for the downstream model) (directory: /CNN)
3. Calibrate CNN probabilities (directory: /calibration)
4. Train downstream model on weather data & calibrated CNN probabilities using training dataset 2 (directory /downstream)
5. Calibrate downstream model probabilities (directory: /calibration)
6. Ensemble calibrated downstream model predictions to get final predictions for test dataset (directory: /ensembling)

Steps 1 and 4 include code for performing architecture testing and hyperparameter tuning. Additionally, model evaluation on validation data, and subsequent model selection, are also performed in steps 1 and 4. 

<!-- I think it is COMPLETELY fair to keep all my data preprocessing code to myself -- bc anyone else's data will be different. Our data is internal. At the very least, prioritize getting the modeling stuff working first and then come back to the data (since Kara/others may want it down the road) -->

<!-- Data is preprocessed and split into two training datasets, validation, and testing datasplits using nested cross validation, with 6 test datasets, and 5-fold cross validation within each.

Using CNNs, a set of baseline models using combinations of 6 architectures, transfer learning, and augmentation are used to identify top performing model choice combinations. Of the top performers, hyperparameter tuning is done by randomly selecting L2 weights and dropout rates to adjust the amount of regularization.

Incorporating weather data is done using two different methods. Originally, Method 1 was used. Method 2 was developed based on changing end-use needs and a clearner inference pipeline for dashboard/UI dev.
- Method 1: HRRR weather forecast model data is used to train a random forest to predict road surface conditions using the gridded forecast data -- see: weather_data/rf_hrrr.py. Further, a merged model (weather_data/weather_merged_algorithm.ipynb) is created by using an algorithm to select a final prediction, considering both the CNN predicted probabilities (image model) with the random forest predicted probabilities (HRRR forecast data).
- Method 2: In /src/ files with hrrr in the file name (config_hrrr.py, hrrr_colocate.py, hrrr_and_img_preds.py). Use the predictions from CNN (already trained, evaluated, and predictions saved out). Colocate images with HRRR grid point and read in data. Train a model on image CNN predictions and HRRR variables.


The ultimate goal is to predict road surface conditions on a wide variety of publicly available camera images, to be done in real-time. The random forest can be used to predict road surface conditions in the future using forecast data alone. -->
