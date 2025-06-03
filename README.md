# DRIVE
Detection of Road Imagery in Variable Environments

Department of Transportation image classification is performed to predict roads surface conditions using convolutional neural networks. The six road surface classes include: 1) severe snow, 2) snow, 3) wet, 4) dry, 5) poor visibility and 6) obstructed. All details about the underlying dataset, including the hand-labeling process following quantitative content analysis, are included here: https://zenodo.org/records/8370665

Data is preprocessed and split into training, validation, and testing datasplits using 5-fold cross validation.

Using CNNs, a set of baseline models using combinations of 6 architectures, transfer learning, and augmentation are used to identify top performing model choice combinations. Of the top performers, hyperparameter tuning is done by randomly selecting L2 weights and dropout rates to adjust the amount of regularization.

Incorporating weather data is done using two different methods. Originally, Method 1 was used. Method 2 was developed based on changing end-use needs and a clearner inference pipeline for dashboard/UI dev.
- Method 1: HRRR weather forecast model data is used to train a random forest to predict road surface conditions using the gridded forecast data -- see: weather_data/rf_hrrr.py. Further, a merged model (weather_data/weather_merged_algorithm.ipynb) is created by using an algorithm to select a final prediction, considering both the CNN predicted probabilities (image model) with the random forest predicted probabilities (HRRR forecast data).
- Method 2: In /src/ files with hrrr in the file name (config_hrrr.py, hrrr_colocate.py, hrrr_and_img_preds.py). Use the predictions from CNN (already trained, evaluated, and predictions saved out). Colocate images with HRRR grid point and read in data. Train a model on image CNN predictions and HRRR variables.


The ultimate goal is to predict road surface conditions on a wide variety of publicly available camera images, to be done in real-time. The random forest can be used to predict road surface conditions in the future using forecast data alone.
