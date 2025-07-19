import downstream_model_train
import config
import os
import pandas as pd

import sys
cnn_scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../CNN/scripts'))
sys.path.append(cnn_scripts_path)

import results_summaries

# Note that this is structured a bit differently from training and evaluating CNNs, which were split up into two parts where we first do the training, and then do evaluation. Here, given the simplicity of the models, train and evaluate (getting one row statistic) all at once, appending the one-row statistic into a static csv. Then, after all training and evaluation is done, have another set of code to summarize innerVal results stats *within* one model (i.e., one CNN + downstream HT set) which has 30 folds to aggregate innerVal across.

if config.train_flag:

    # Data set up is similar to the calibration step...
    # This code is set up to train downstream ML model, and evaluate on all predictions, where the data is organized in a csv. This code points to a directory (as designated in config), and this downstream modeling process will be complete for each csv file of data within that directory. Each csv represents data that is used to train/eval the dowsntream ML model. The csvs stem from previous classification model & calibration outputs, so they are the predictions from multiple splits and models. Each one requires a downstream model.

    # Given the note above, there is no need to tie in specific tracker lists, as they would have been included in the csvs in the directory that is being pointed to.

    datafiles = os.listdir(config.dir_of_datacsvs_CNNCalibPreds)

    for f in datafiles: 
        # runname = f[:-4] # remove the .csv
        csv = f"{config.dir_of_datacsvs_CNNCalibPreds}/{f}"
        print(f"reading file {csv}")

        # parse out some information for tracking/logging results
        ident = f.find("__")
        tracker = f[:ident]
        exp = f[ident+2:-4]

        # read in data
        cnndata = pd.read_csv(csv)

        # load in weather data
        hrrr = downstream_model_train.hrrr_data_load_prep(config.hrrr_data_csv)

        # merge data 
        cnn_and_weather_data = downstream_model_train.merge_cnn_and_weather_data(cnndata_df = cnndata, weatherdata_df = hrrr,cols_to_keep_cnn = config.cols_from_cnn)

        # data prepped for model training
        train_i, train_o, val_i, val_o = downstream_model_train.prepare_data_fortraining(dfinput = cnn_and_weather_data, features = config.features_for_training)

        print("amt of data for training")
        print(len(train_i))

        print("amt of data for validation")
        print(len(val_i))

        # train models
        for alginput in ["logistic", "gnb", "svm", "DNN", "rf"]: 
            hyps_torun = downstream_model_train.grab_alg_hyps(alg = alginput)

            for hypinput in hyps_torun:

                img_cat, model_pred = downstream_model_train.run_training(alg = alginput, hypselected = hypinput, train_input_scaled = train_i, train_output_data = train_o, val_input_scaled = val_i, val_output_data = val_o)

                # may be an issue here w what form img_cat is returning, these may be numerics not cat names.., so may affect how the calc_summary function runs

                df_i = pd.DataFrame({"img_cat": img_cat, "model_pred": model_pred}).reset_index()

                print(df_i[0:6])

                # grab one-line summary results per model | fold (later, after running all HT, will then have to summarize/collect the results for a single experiment across the 30, since these are separated by fold and not aggregated across the 30)
                dict_i = results_summaries.calc_summary(dfinput = df_i, tracker_desc = tracker, exp_desc = exp,  exp_details = f"{alginput}_{hypinput}")

                print(dict_i)

                downstream_model_train.track_results_main_file(dict_i, mainresultsfile = config.file_collect_results)
