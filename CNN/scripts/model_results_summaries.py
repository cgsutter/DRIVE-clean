import pandas as pd
import os
import _config as config

# def results(predsfile_csv):
#     df = pd.read_csv(predsfile_csv)
#     tracker_filebase
#     tracker_rundetails


def identify_ok_preds(ytrue, ypred):
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
    dfforcalc = pd.DataFrame({"img_cat": ytrueinput, "pred": ypredinput})

    splitspecific_list = []

    dfforcalc["correct_flag"] = ytrueinput == ypredinput
    splitspecific_list.append(len(dfforcalc))
    splitspecific_list.append(sum(dfforcalc["correct_flag"]))

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
        splitspecific_list.append(cat_total)
        splitspecific_list.append(cat_correct)
    

    oks = identify_ok_preds(ytrue = ytrueinput, ypred = ypredinput)
    sum(oks)
    splitspecific_list.append(sum(oks))

    print("end and printing splitspecific_list")
    print(splitspecific_list)
    return splitspecific_list


def results(exp_desc = config.exp_desc, preds_path = config.preds_path, exp_details = ""):

    """
    exp_details come from helper function, this is just another differentiator of the experiment, beyond just the simple exp_desc
    """

    entiredir = os.listdir(preds_path)
    subset_to_exp = [i for i in entiredir if exp_desc in i]
    pred_csv_list = [i for i in subset_to_exp if exp_details in i]

    print(len(pred_csv_list))

    for c in pred_csv_list[0:2]:
        df = pd.read_csv(f"{config.preds_path}/{c}")
        print(df.columns)
        l = calcstats_onefold(ytrueinput = df["img_cat"], ypredinput= df["model_pred"])
        print(l)



