import pandas as pd
import os
import _config as config
import csv
import datetime



def identify_ok_preds(ytrue, ypred):
    """
    Determines which predictions are 'acceptable' based on relaxed class-matching logic.

    Returns a list of booleans indicating whether each prediction is acceptable.
    """
    oks = []
    for i in range(0, len(ytrue)):
        if ytrue[i] == ypred[i]:
                ok = True
        elif (ytrue[i] == "snow_severe") & (
            ypred[i] == "snow"
        ):
            ok = True
        elif (ytrue[i] == "snow") & (
            (ypred[i] == "snow_severe")
            | (ypred[i] == "wet")
        ):
            ok = True
        elif (ytrue[i] == "wet") & (
            (ypred[i] == "snow") | (ypred[i] == "dry")
        ):
            ok = True
        elif (ytrue[i] == "dry") & (
            ypred[i] == "wet"
        ):
            ok = True
        else:
            ok = False
        oks.append(ok)

    return oks

def calcstats_onefold(ytrueinput, ypredinput):
    """
    Calculates summary statistics from true and predicted labels, including accuracy and class-wise totals.

    Returns a tuple of (metric names list, results list).
    """
    dfforcalc = pd.DataFrame({"img_cat": ytrueinput, "pred": ypredinput})

    results_list = []

    dfforcalc["correct_flag"] = ytrueinput == ypredinput
    results_list.append(len(dfforcalc))
    results_list.append(sum(dfforcalc["correct_flag"]))

    # by class
    for c in [
        "snow_severe",
        "snow",
        "wet",
        "dry",
        "poor_viz",
        # "obs",
    ]:  # added obs. Note that this was not included in original merging method w hrrr RF + merge algo logic
        sub = dfforcalc[dfforcalc["img_cat"] == c]
        cat_total = len(sub)
        cat_correct = len(sub[sub["correct_flag"] == True])
        results_list.append(cat_total)
        results_list.append(cat_correct)
    

    oks = identify_ok_preds(ytrue = ytrueinput, ypred = ypredinput)
    sum(oks)
    results_list.append(sum(oks))

    # print("end and printing results_list")
    # print(results_list)

    metrics_list = ["totalims","correctims","nims_snow_severe","correct_snow_severe","nims_snow","correct_snow","nims_wet","correct_wet","nims_dry","correct_dry","nims_poor_viz","correct_poor_viz", "ok"]


    return metrics_list, results_list


def grab_pred_files(exp_desc = config.exp_desc, preds_path = config.preds_path, exp_details = ""):
    """
    Returns a list of prediction CSV filenames in the specified directory matching given experiment descriptors.
    """
    entiredir = os.listdir(preds_path)
    subset_to_exp = [i for i in entiredir if exp_desc in i]
    pred_csv_list = [i for i in subset_to_exp if exp_details in i]
    print(len(pred_csv_list))
    return pred_csv_list

def calc_summary(dfinput = "", pred_csv_name = "",exp_desc = config.exp_desc, preds_path = config.preds_path, exp_details = ""):
    """
    Summarizes metrics for a single prediction file, including overall and class-wise accuracy,
    and attaches identifying experiment details.

    Returns a dictionary of metrics and values.
    """
    # print(dfinput.columns)
    metrics_list, results_list = calcstats_onefold(ytrueinput = dfinput["img_cat"], ypredinput= dfinput["model_pred"])
    # print(results_dict)
    
    # grab the info that differentiates this tracker from the others in this experiment, which is done by taking the tracker name, and removing the string text with "exp_desc" and also removes "exp_details", both of these are the same across. Also must remove path.
    tracker_desc = pred_csv_name.replace(exp_desc, '').replace(preds_path, '').replace('.csv', '').replace(exp_details, '')

    # these will be the same across all 30 trackers, just adding for thoroughness
    results_list.append(exp_desc)
    metrics_list.append("exp_desc")
    results_list.append(exp_details)
    metrics_list.append("exp_details")
    # this will be unique differentiator for the tracker specifically
    results_list.append(tracker_desc)
    metrics_list.append("tracker")
    # print(metrics_list)

    dict_results = dict(zip(metrics_list, results_list))
    # print(dict_results)

    return dict_results


# exp_details come from helper function, this is just another differentiator of the experiment, beyond just the simple exp_desc


def save_results(listofdics = [], newresultsfile = ""):
    """
    Saves a list of result dictionaries to a CSV file, writing the header only once.
    """
    # newresultsfile = f"{config.results_path}/{exp_desc}_{exp_details}.csv"
    with open(newresultsfile, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=listofdics[0].keys())
        writer.writeheader()
        writer.writerows(listofdics)
    

def run_results_by_exp(predfiles_input, exp_desc_input = config.exp_desc, preds_path_input = config.preds_path, results_path_input = config.results_path, exp_details_input = "", subsetphase = ""):
    """
    Processes multiple prediction CSVs for a given experiment, optionally subsetting by phase,
    calculates summary metrics for each, and saves results to a combined CSV, with one model per row. 
    """

    results_30dicts = []

    for predfile_i in predfiles_input:
        df = pd.read_csv(f"{preds_path_input}/{predfile_i}")
        if ((subsetphase == "innerTrain")|(subsetphase == "innerVal")|(subsetphase == "innerTest")):
            df = df[df["innerPhase"] == subsetphase]
            df = df.reset_index()
        elif (subsetphase == "outerTest"):
            df = df[df["outerPhase"] == subsetphase]
            df = df.reset_index()
        # o/w no subsetting

        if subsetphase == "":
            phase_save_string = "All"
        else:
            phase_save_string = subsetphase

        dict_i = calc_summary(dfinput = df, pred_csv_name = predfile_i, exp_desc = exp_desc_input, preds_path = preds_path_input, exp_details = exp_details_input)

        results_30dicts.append(dict_i)



    csvsave = f"{results_path_input}/{exp_desc_input}_{exp_details_input}_{phase_save_string}.csv"
    print(f"saving to {csvsave}")
    save_results(listofdics = results_30dicts, newresultsfile = csvsave)

def exp_total_innerVal(exp_desc_input = config.exp_desc, preds_path_input = config.preds_path, results_path_input = config.results_path, exp_details_input = ""):
    """
    Loads results from all models in the inner validation set, sums their metrics,
    adds metadata, and appends to a main results file.
    """
    resultssaved = f"{results_path_input}/{exp_desc_input}_{exp_details_input}_innerVal.csv"
    df_innerVal = pd.read_csv(resultssaved)

    # summing across these cols wont work, just append as one value for the one row being produced
    df_innerVal = df_innerVal.drop(columns = ["exp_desc","exp_details","tracker"])

    now = datetime.datetime.now() # for logging this experiment

    # aggregate across all 30 models to get final resul
    column_sums = df_innerVal.sum().tolist()
    print(type(column_sums))
    column_sums.extend([exp_desc_input,exp_details_input,now])
    column_names = list(df_innerVal.columns)
    print(type(column_names))
    column_names.extend(["exp_desc","exp_details", "rundatetime"])

    dict_results = dict(zip(column_names, column_sums))
    print(dict_results)

    # save out to a main file that will collect all model results, even hyptuning
    newresultsfile = f"{config.results_path}/results_by_exp_innerVal.csv"

    file_exists = os.path.isfile(newresultsfile)

    with open(newresultsfile, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=dict_results.keys())
        if not file_exists:
            writer.writeheader() # write header only once
        writer.writerow(dict_results) # always write the data row
