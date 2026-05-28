import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# =========================================
# CONFIGURATION
# =========================================
tracker_paths = [
    "/home/csutter/DRIVE-clean/operational_models/trackers/tracker_m0.csv",
    "/home/csutter/DRIVE-clean/operational_models/trackers/tracker_m1.csv",
    "/home/csutter/DRIVE-clean/operational_models/trackers/tracker_m2.csv",
    "/home/csutter/DRIVE-clean/operational_models/trackers/tracker_m3.csv",
    "/home/csutter/DRIVE-clean/operational_models/trackers/tracker_m4.csv"
]

hrrr_data_path = "/home/csutter/DRIVE-clean/forecasting/coloc_hrrrdata/labeleddata_FH02.csv"
output_dir = "/home/csutter/DRIVE-clean/forecast_operational_models/models/"

os.makedirs(output_dir, exist_ok=True)

features = ["t2m", "r2", "asnow", "tp", "tcc", "uavg"]
target_col = "img_cat"
merge_key = "img_name"  # The unique ID to securely join the datasets

# The winning architecture from the FH02 evaluation
rf_hyps = {
    "max_depth": 10,
    "max_samples": 0.5,
    "n_estimators": 300,
    "max_features": 2,
    "min_samples_leaf": 5,
    "bootstrap": True,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": 4
}

# =========================================
# MAIN EXECUTION
# =========================================
print(f"=== Starting Final Operational Model Training ===")

# 1. Load the compiled HRRR data once outside the loop to save memory/time
print("Loading HRRR weather data...")
hrrr_df = pd.read_csv(hrrr_data_path)

# Subset HRRR data to just the unique key, the tracking path, and the features we need
hrrr_cols_to_keep = [merge_key, "hrrr_file_path"] + features
hrrr_subset = hrrr_df[hrrr_cols_to_keep].copy()

# Drop any duplicate rows in the HRRR subset just in case, to ensure a clean 1:1 merge
hrrr_subset = hrrr_subset.drop_duplicates(subset=[merge_key])

for t_idx, tracker_path in enumerate(tracker_paths):
    model_id = f"m{t_idx}"
    print(f"\nProcessing {model_id}...")
    
    # 2. Load the base tracker
    df_tracker = pd.read_csv(tracker_path)
    
    # 3. Merge the HRRR features into the tracker safely using the unique image name
    df_merged = df_tracker.merge(hrrr_subset, on=merge_key, how="left")
    
    # 4. Apply the operational split logic (Train on Train + Test, ignore Val)
    train_mask = df_merged["innerPhase"].isin(["innerTrain", "innerTest"])
    df_train = df_merged[train_mask].copy()
    
    # Drop rows that are missing the HRRR features
    df_clean = df_train.dropna(subset=features).copy()
    
    X_raw = df_clean[features].values
    y_raw = df_clean[target_col].values
    
    print(f"  -> Filtered for training pool. Training on {len(X_raw)} observations...")
    
    # 5. Fit and transform the Scaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_raw)
    
    # 6. Fit and transform the Label Encoder
    le = LabelEncoder()
    y_train = le.fit_transform(y_raw)
    
    # 7. Train the Model
    print("  -> Fitting Random Forest...")
    model = RandomForestClassifier(**rf_hyps)
    model.fit(X_train, y_train)
    
    # 8. Save the operational artifacts
    model_out = os.path.join(output_dir, f"operational_rf_{model_id}.joblib")
    scaler_out = os.path.join(output_dir, f"operational_scaler_{model_id}.joblib")
    encoder_out = os.path.join(output_dir, f"operational_encoder_{model_id}.joblib")
    
    joblib.dump(model, model_out)
    joblib.dump(scaler, scaler_out)
    joblib.dump(le, encoder_out)
    
    print(f"  -> Successfully saved model, scaler, and encoder for {model_id}.")

print("\n=== Operational Training Complete! ===")
print(f"Artifacts are located in: {output_dir}")