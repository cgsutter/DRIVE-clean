# Portions of this code were writen with the assistance of AI tools (Gemini)

import pandas as pd
import numpy as np
import os

# This code summarizes the results of HT by algo & hyp, because during model training (hyperparameter tuning) results are saved out at the split | alg | hyp level, and we need to aggregate across splits to see results at the alg & hyp level. 

# This should be run after all model training (hyperparameter tuning, from /home/csutter/DRIVE-clean/forecasting/model_training_HT.py) is complete

# Run this code for FH "02", "09", and "24", as those were the cases we ran model training (hyp tuning) for. 

# =========================================
# CONFIGURATION
# =========================================
CURRENT_FH = "24"
input_csv = f"/home/csutter/DRIVE-clean/forecasting/data_HT_results/FH{CURRENT_FH}/HT_results_FH{CURRENT_FH}.csv"
output_dir = f"/home/csutter/DRIVE-clean/forecasting/data_HT_aggregate/FH{CURRENT_FH}/"
output_csv = os.path.join(output_dir, f"Aggregated_HT_FH{CURRENT_FH}.csv")

os.makedirs(output_dir, exist_ok=True)

classes_to_track = ["snow_severe", "snow", "wet", "dry", "poor_viz"]

print(f"Aggregating results for FH {CURRENT_FH}...")

# Load the actively generating CSV safely
try:
    df = pd.read_csv(input_csv, on_bad_lines='skip')
except FileNotFoundError:
    print(f"Error: {input_csv} not found. Check if the HT script has started writing.")
    exit()

# Columns to sum across the 30 trackers
count_cols = ["n_labeled_total", "n_correct_total"]
for c in classes_to_track:
    count_cols.extend([f"n_labeled_{c}", f"n_correct_{c}"])

# Ensure columns are numeric
for col in count_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=count_cols)

# Group by the specific model architecture
agg_funcs = {col: 'sum' for col in count_cols}
agg_funcs['tracker_name'] = 'count' # This counts how many trackers have finished

grouped = df.groupby(['algorithm', 'hyperparameters']).agg(agg_funcs).reset_index()
grouped = grouped.rename(columns={'tracker_name': 'trackers_completed'})

# Recalculate true metrics based on the summed raw counts
grouped['total_acc'] = np.where(
    grouped['n_labeled_total'] > 0, 
    grouped['n_correct_total'] / grouped['n_labeled_total'], 
    0.0
)

recalls = []
for c in classes_to_track:
    recall_col = f"recall_{c}"
    labeled_col = f"n_labeled_{c}"
    correct_col = f"n_correct_{c}"
    
    grouped[recall_col] = np.where(
        grouped[labeled_col] > 0, 
        grouped[correct_col] / grouped[labeled_col], 
        0.0
    )
    recalls.append(grouped[recall_col])

# Calculate macro average recall across the 5 classes
grouped['avg_recall'] = pd.concat(recalls, axis=1).mean(axis=1)

# Sort by best average recall
grouped = grouped.sort_values(by='avg_recall', ascending=False)

grouped.to_csv(output_csv, index=False)
print(f"Aggregation complete. Saved to: {output_csv}")