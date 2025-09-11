import pandas as pd
import numpy as np

# Not needed for model training..

# This code is just reference code to print the hyperparams in a list, a format for the config file, rather than a csv to read in and preprocess. Just keep all HT combinations in list format in config for readibility.

###### For RF
# hyps_path = "/home/csutter/DRIVE/dot/models_streamline/HT/hypgrid_rf_11feat_grid288.csv"
# hypdf = pd.read_csv(hyps_path)
# print(len(hypdf))

# # hypdf = hypdf.replace({np.nan: None})
# print(len(hypdf))  # run them all for dnn not as many as rf
# hypdf = hypdf.drop(
#     columns=[col for col in hypdf.columns if "Unnamed" in col], axis=1
# )
# print(hypdf)

# param_combinations = []

# # for ds in [regular,nomaxdepth,nomaxsamples,nomaxdepth_nomaxsamples]:
# for _, row in hypdf.iterrows():
#     row_dict = {}

#     # Skip max_depth if it's 999
#     if row["max_depth"] != 999:
#         row_dict["max_depth"] = int(
#             row["max_depth"]
#         )  # make sure it's int, not float

#     # Skip max_samples if it's 999
#     if row["max_samples"] != 999.00:
#         row_dict["max_samples"] = row["max_samples"]

#     # Include all other parameters
#     for col in hypdf.columns:
#         if col not in ["max_depth", "max_samples"]:
#             row_dict[col] = row[col]

#     param_combinations.append(row_dict)

# print("rf_HT = [")
# for param in param_combinations:
#     print(f"    {param},")
# print("]")

###### For DNN
hyps_path = "/home/csutter/DRIVE/dot/models_concatdata/nowcast/features12a/sitesplit/hypgrid_dnn.csv"
hypdf = pd.read_csv(hyps_path)
print(len(hypdf))

hypdf = hypdf.replace({np.nan: None})
print(len(hypdf))  # run them all for dnn not as many as rf
hypdf = hypdf.drop(columns=["index"])
# this param_combinations is normally where the long list  of hyperparams would be
param_combinations = hypdf.to_dict(orient="records")

print("dnn_HT = [")
for param in param_combinations:
    print(f"    {param},")
print("]")
