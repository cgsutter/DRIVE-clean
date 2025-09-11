import _config as config
import helper_fns_adhoc
import model_compile_fit
import class_weights
import model_build
import callbacks
import results_predictions as results_predictions
import results_summaries
import pandas as pd
import wandb
import os


def evaluate_exp_models(
    trackerslist,
    tracker_rundetails,
    oneoff_flag,
    hyp_flag,
    dfhyp,
    dataset_all,
    all_images,
    arch_eval,
    trle_eval,
    ast_eval,
    dr_eval,
    l2_eval,
):  # all_labels
    """This function prepares results from models in multiple steps, first, by evaluating the full dataset (22k) on all models, and for one-off models we save these predictions from the 30 models as csvs (do not for hyp tuning, would be too much data, when all we need is the summary stats). Second, summaries are created by aggregating the results of the 30 models for the innerVal phase, creating a dataframe with all the innerVal predictions for each of the 30 models. Third, that innerVal dataframe is summarized by summing over all predictions, to get one single row of summary stats for that experiment (which includes all 30 datasets in it) -- this is appended to an existing csv to have one main source to collect all model summary statistics.

    Args:
        trackerslist (list): A list of tracker paths
        tracker_rundetails (string): String containing the details which align with how the models naming convention was saved out
        oneoff_flag (bool): Type of run, set in config
        hyp_flag (bool): Type of run, set in config
        dfhyp (pd.DataFrame): DataFrame containing hyperparameter sets
        dataset_all (tensorflow dataset): Dataset prepared for evaluation on a TF model
        all_labels (list): List of labels corresponding to dataset_all
        all_images (list): List of image file paths corresponding to dataset_all
    """

    if oneoff_flag:
        preddfs_30 = []
        descs_30 = []

        for t in trackerslist:
            print(t)
            tracker_details = helper_fns_adhoc.prep_filedetails(
                tracker_designated=t, rundetails=tracker_rundetails
            )

            df_preds, t_name = results_predictions.evaluate(
                modeldir=f"{config.model_path}/{tracker_details}",
                dataset=dataset_all,
                imgnames=all_images,
                trackerinput=t,
                saveto=f"{config.preds_path}/{tracker_details}.csv",
                saveflag=True,
                run_arch=arch_eval,
                run_trle=trle_eval,
                run_ast=ast_eval,
                run_dr=dr_eval,
                run_l2=l2_eval,
            )
            preddfs_30.append(df_preds)
            descs_30.append(t_name)

        print("one_off evalC")
        for phase in ["", "innerTrain", "innerVal", "innerTest", "outerTest"]:
            results_df_of30 = results_summaries.run_results_by_exp(
                preddfs_input=preddfs_30,
                preddfs_desc=descs_30,
                exp_desc_input=config.exp_desc,
                exp_details_input=tracker_rundetails,
                subsetphase=phase,
            )
            csvsave = f"{config.results_path}/{config.exp_desc}_{tracker_rundetails}_{phase}.csv"
            print(f"saving to {csvsave}")
            results_df_of30.to_csv(csvsave)  # save out all phases to csv
            print(results_df_of30.head(4))
            print("one_off evalD")
            if (
                phase == "innerVal"
            ):  # Append to Main tracker of all experiments performance, but only for innerVal
                results_summaries.exp_total_innerVal(
                    df_innerVal=results_df_of30,
                    exp_desc_input=config.exp_desc,
                    exp_details_input=tracker_rundetails,
                )
                print("one_off evalE")
    if (
        hyp_flag
    ):  # have two sets of loops for both the trackers and the hyperparameter set

        preddfs_30 = []
        descs_30 = []
        for t in config.trackers_list:
            print(
                f"ENTER4: inside loop by tracker within call_model_results, should be ran the amount of times dependent on # of trackers"
            )
            print("inside loop for evaluation")

            tracker_details = helper_fns_adhoc.prep_filedetails(
                tracker_designated=t, rundetails=tracker_rundetails
            )

            df_preds, t_name = results_predictions.evaluate(
                modeldir=f"{config.model_path}/{tracker_details}",
                dataset=dataset_all,
                imgnames=all_images,
                trackerinput=t,
                saveto="",
                saveflag=False,
                run_arch=arch_eval,
                run_trle=trle_eval,
                run_ast=ast_eval,
                run_dr=dr_eval,
                run_l2=l2_eval,
            )
            preddfs_30.append(df_preds)
            descs_30.append(t_name)
        # only run for inner val, no need to run for all phases bc not saving those to csv, only need inner val to then run the final summary analysis to append to Main tracker for experiment results in total

        results_df_of30 = results_summaries.run_results_by_exp(
            preddfs_input=preddfs_30,
            preddfs_desc=descs_30,
            exp_desc_input=config.exp_desc,
            exp_details_input=tracker_rundetails,
            subsetphase="innerVal",
        )

        results_summaries.exp_total_innerVal(
            df_innerVal=results_df_of30,
            exp_desc_input=config.exp_desc,
            exp_details_input=tracker_rundetails,
        )
