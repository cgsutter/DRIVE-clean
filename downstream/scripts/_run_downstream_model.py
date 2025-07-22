import downstream_model_train
import config
import os
import pandas as pd

from joblib import dump, load

import sys
cnn_scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../CNN/scripts'))
sys.path.append(cnn_scripts_path)

# imported reusable code from CNN directory
import results_summaries
import helper_fns_adhoc

# Note that this is structured a bit differently from training and evaluating CNNs, which were split up into two parts where we first do the training, and then do evaluation. Here, given the simplicity of the models, train and evaluate (getting one row statistic) all at once, appending the one-row statistic into a static csv. Then, after all training and evaluation is done, have another set of code to summarize innerVal results stats *within* one model (i.e., one CNN + downstream HT set) which has 30 folds to aggregate innerVal across.

if config.HT_flag:

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
        train_i, train_o, val_i, val_o, all_i, all_o, all_imgname, scaler_model = downstream_model_train.prepare_data_fortraining(dfinput = cnn_and_weather_data, features = config.features_for_training)

        print("amt of data for training")
        print(len(train_i))

        print("amt of data for validation")
        print(len(val_i))

        # train models
        for alginput in ["logistic", "gnb", "svm", "DNN", "rf"]: 
            hyps_torun = downstream_model_train.grab_alg_hyps(alg = alginput)

            for hypinput in hyps_torun:

                img_cat, model_pred, model, allprobs, allpreds, all_img_cat = downstream_model_train.run_training(alg = alginput, hypselected = hypinput, train_input_scaled = train_i, train_output_data = train_o, val_input_scaled = val_i, val_output_data = val_o, makepreds_flag = False, alldata_input_use = None)

                df_i = pd.DataFrame({"img_cat": img_cat, "model_pred": model_pred}).reset_index()

                print(df_i[0:6])

                # grab one-line summary results per model | fold (later, after running all HT, will then have to summarize/collect the results for a single experiment across the 30, since these are separated by fold and not aggregated across the 30)
                dict_i = results_summaries.calc_summary(dfinput = df_i, tracker_desc = tracker, exp_desc = exp,  exp_details = f"{alginput}_{hypinput}")

                print(dict_i)

                downstream_model_train.track_results_main_file(dict_i, mainresultsfile = config.file_collect_results)

elif config.final_selected_train:

    # This is similar to the HT training above except for 1) need to save out the models (both the classifier model AND the scaler model), 2) and also save out the predictions, 3) one single model being ran, as set in config, so don't need to loop through tons of HT models in a loop. Insetad, we're just running one model as set in config. 4) We do still need to run it for each data file (30), but only for one CNN model's predictions (and it's best downstream classification model), this is because we have ONE final model for the whole flow: one CNN (recall we carried/tested the top 4), and one downstream model (which was hyptuning above). 

    datafiles = os.listdir(config.dir_of_datacsvs_CNNCalibPreds)

    # The final downstream model is ran for one final CNN. See note above. 
    datafiles = [x for x in datafiles if config.final_cnn in x]

    print(f"Grabbed {len(datafiles)} data files to train the final downstream classification model on. CHECK: This should be be 30.")

    for f in datafiles: 
        # runname = f[:-4] # remove the .csv
        csv = f"{config.dir_of_datacsvs_CNNCalibPreds}/{f}"
        print(f"reading file {csv}")

        # parse out some information for tracking/logging results
        ident = f.find("__")
        tracker = f[:ident]
        exp = f[ident+2:-4]

        # savetoname used for both models and data
        # needs to incorporate the data tracker (of the 30), the cnn info, and the selected downstream model info (from config)
        hypstring = str(config.final_downstream_hyp)
        savetoname = f"{f[:-4]}_{config.final_downstream_alg}_{hypstring}"

        # read in data
        cnndata = pd.read_csv(csv)

        # load in weather data
        hrrr = downstream_model_train.hrrr_data_load_prep(config.hrrr_data_csv)

        # merge data 
        cnn_and_weather_data = downstream_model_train.merge_cnn_and_weather_data(cnndata_df = cnndata, weatherdata_df = hrrr,cols_to_keep_cnn = config.cols_from_cnn)

        # data prepped for model training
        train_i, train_o, val_i, val_o, all_i, all_o, all_imgname, scaler_model = downstream_model_train.prepare_data_fortraining(dfinput = cnn_and_weather_data, features = config.features_for_training)

        dump(scaler_model, f"{config.scalarmodel_directory}/{savetoname}.pkl")

        print("saved scaler model")

        # train one model as set in config
        alginput = config.final_downstream_alg
        hypinput = config.final_downstream_hyp

        img_cat, model_pred, model, allprobs, allpreds, all_img_cat = downstream_model_train.run_training(alg = alginput, hypselected = hypinput, train_input_scaled = train_i, train_output_data = train_o, val_input_scaled = val_i, val_output_data = val_o, makepreds_flag = True, alldata_input_use = all_i, alldata_output_use = all_o)

        dump(model, f"{config.model_directory}/{savetoname}.pkl")

        print("saved downstream classification model")

        print("check lengths of output")

        print(len(cnn_and_weather_data))
        print(len(allpreds))
        print(len(allprobs))

        dict_catKey_indValue, dict_indKey_catValue= helper_fns_adhoc.cat_str_ind_dictmap()

        df_results = pd.DataFrame(
            allprobs, columns=[f"ds_prob_{dict_indKey_catValue[i]}" for i in range(len(dict_indKey_catValue))]
        )

        df_results["ds_pred"] = allpreds

        df_final = pd.concat([cnn_and_weather_data, df_results], axis=1)

        print(len(df_final))

        # print results - quick check, if there is something wrong in the ordering of predictions such that merging it with the full dataframe doesn't match, the results will be poor, and can check that with quick accuracy check below. 
        correcty = sum(df_final["ds_pred"]==df_final["img_cat"])
        print(correcty/len(df_final))
