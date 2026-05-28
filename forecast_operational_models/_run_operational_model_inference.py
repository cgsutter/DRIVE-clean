import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime, timedelta

# =========================================
# 1. CONFIGURATION & PATHS
# =========================================
FORECAST_HOURS = ["02", "03", "04", "05", "06", "09", "12", "15", "18", "24", "30", "36", "42", "48"]

# Base Directories
base_run_dir = "/home/csutter/DRIVE-clean/forecast_operational_runs/set8_jan292022/" #HERE!! This is the only spot to update
hrrr_base_dir = "/home/csutter/AI2ES/cleaned/HRRR/"
models_base_dir = "/home/csutter/DRIVE-clean/forecast_operational_models/models/"

# Inputs
dates_csv_path = os.path.join(base_run_dir, "dates.csv")
features = ["t2m", "r2", "asnow", "tp", "tcc", "uavg"]
classes_to_track = ["snow_severe", "snow", "wet", "dry", "poor_viz"]

# Severity mappings for vectorized tie-breakers
severity_order = {"dry": 1, "wet": 2, "snow": 3, "poor_viz": 4, "snow_severe": 5}
reverse_sev_map = {v: k for k, v in severity_order.items()}

# NYS Bounding Box
NYS_LAT_MIN, NYS_LAT_MAX = 40.47, 45.02
NYS_LON_MIN, NYS_LON_MAX = -79.77, -71.85

# =========================================
# 2. PRE-LOAD MODELS (HUGE SPEEDUP)
# =========================================
print("=== Loading 5 Operational Models into Memory ===")
operational_artifacts = {}
for m_idx in range(5):
    model_id = f"m{m_idx}"
    operational_artifacts[model_id] = {
        "scaler": joblib.load(os.path.join(models_base_dir, f"operational_scaler_{model_id}.joblib")),
        "encoder": joblib.load(os.path.join(models_base_dir, f"operational_encoder_{model_id}.joblib")),
        "model": joblib.load(os.path.join(models_base_dir, f"operational_rf_{model_id}.joblib"))
    }

# =========================================
# 3. MAIN OPERATIONAL PIPELINE
# =========================================
print("\n=== Starting Operational Inference Pipeline ===")

dates_df = pd.read_csv(dates_csv_path)

for index, row in dates_df.iterrows():
    raw_timestamp = str(row['date']) # e.g., "20231101_0800"
    valid_time = datetime.strptime(raw_timestamp, "%Y%m%d_%H%M")
    
    yyyy, mm, dd = valid_time.strftime("%Y"), valid_time.strftime("%m"), valid_time.strftime("%d")
    
    out_hrrr_dir = os.path.join(base_run_dir, "data1_hrrr", yyyy, mm, dd, raw_timestamp)
    out_preds_dir = os.path.join(base_run_dir, "data2_modelpreds", yyyy, mm, dd, raw_timestamp)
    out_ens_dir = os.path.join(base_run_dir, "data3_ensembling", yyyy, mm, dd, raw_timestamp)
    
    os.makedirs(out_hrrr_dir, exist_ok=True)
    os.makedirs(out_preds_dir, exist_ok=True)
    os.makedirs(out_ens_dir, exist_ok=True)

    print(f"\n=========================================")
    print(f" PROCESSING VALID TIME: {valid_time}")
    print(f"=========================================")

    for fh in FORECAST_HOURS:
        print(f"\n  [Forecast Hour: {fh}]")
        
        init_time = valid_time - timedelta(hours=int(fh))
        init_yyyy, init_mm, init_dd = init_time.strftime("%Y"), init_time.strftime("%m"), init_time.strftime("%d")
        init_hh = init_time.strftime("%H")
        
        # -----------------------------------------
        # STEP 1: HRRR DATA PREP 
        # -----------------------------------------
        hrrr_filename = f"{init_yyyy}{init_mm}{init_dd}_hrrr.t{init_hh}z_{fh}.parquet"
        hrrr_path = os.path.join(hrrr_base_dir, init_yyyy, init_mm, hrrr_filename)
        
        if not os.path.exists(hrrr_path):
            print(f"    -> [ERROR] File not found: {hrrr_filename}. Skipping FH{fh}.")
            continue
            
        hrrr_df = pd.read_parquet(hrrr_path)
        
        # Stamp the HRRR file path for traceability
        hrrr_df["hrrr_file_path"] = hrrr_filename
        
        # Apply NYS Filter
        nys_mask = (
            (hrrr_df["latitude"] >= NYS_LAT_MIN) & (hrrr_df["latitude"] <= NYS_LAT_MAX) &
            (hrrr_df["longitude"] >= NYS_LON_MIN) & (hrrr_df["longitude"] <= NYS_LON_MAX)
        )
        hrrr_df = hrrr_df[nys_mask].copy()
        
        hrrr_df["uavg"] = np.sqrt(hrrr_df["u10"]**2 + hrrr_df["v10"]**2)
        hrrr_df["location_id"] = hrrr_df["latitude"].astype(str) + "_" + hrrr_df["longitude"].astype(str)
        
        # Keep features + our essential tracking columns
        cols_to_keep = features + ["location_id", "latitude", "longitude", "hrrr_file_path"]
        hrrr_clean = hrrr_df.dropna(subset=features)[cols_to_keep].copy()
        
        # Save Step 1 with _FH appendage
        hrrr_clean.to_csv(os.path.join(out_hrrr_dir, f"step1_hrrr_FH{fh}.csv"), index=False)
        
        X_live = hrrr_clean[features].values
        
        # Initialize the base dataframe for Steps 2 & 3
        merged_df = hrrr_clean[["location_id", "latitude", "longitude", "hrrr_file_path"]].copy()
        
        # -----------------------------------------
        # STEP 2: RUN THE 5 OPERATIONAL MODELS
        # -----------------------------------------
        for m_idx in range(5):
            model_id = f"m{m_idx}"
            arts = operational_artifacts[model_id]
            
            X_scaled = arts["scaler"].transform(X_live)
            y_pred_enc = arts["model"].predict(X_scaled)
            y_prob = arts["model"].predict_proba(X_scaled)
            
            # Predict and map probabilities directly onto the merged dataframe
            merged_df[f"{model_id}_pred"] = arts["encoder"].inverse_transform(y_pred_enc)
            for i, class_name in enumerate(arts["encoder"].classes_):
                merged_df[f"{model_id}_prob_{class_name}"] = y_prob[:, i]
                
            # Save Step 2 isolated model outputs (now including the file path)
            m_cols = ["location_id", "hrrr_file_path", f"{model_id}_pred"] + [f"{model_id}_prob_{c}" for c in classes_to_track]
            merged_df[m_cols].to_csv(os.path.join(out_preds_dir, f"model_{model_id}_FH{fh}.csv"), index=False)

        # -----------------------------------------
        # STEP 3: HIGH-SPEED VECTORIZED ENSEMBLING
        # -----------------------------------------
        pred_cols = [f"m{i}_pred" for i in range(5)]
        all_prob_cols = [f"m{i}_prob_{c}" for i in range(5) for c in classes_to_track]
        
        # A) Vectorized Average Probability
        for c in classes_to_track:
            prob_cols = [f"m{i}_prob_{c}" for i in range(5)]
            merged_df[f"ensembleAvg_prob_{c}"] = merged_df[prob_cols].mean(axis=1)
            
        avg_cols = [f"ensembleAvg_prob_{c}" for c in classes_to_track]
        merged_df["ensembleAvg_pred"] = merged_df[avg_cols].idxmax(axis=1).str.replace("ensembleAvg_prob_", "")
        
        # B) Vectorized Max Confidence
        max_col = merged_df[all_prob_cols].idxmax(axis=1)
        merged_df["ensembleMaxConf_pred"] = max_col.str.split("_prob_").str[1]
        
        # C) Vectorized Mode & Tie-Breaker
        for c in classes_to_track:
            merged_df[f"count_{c}"] = (merged_df[pred_cols] == c).sum(axis=1)
        
        count_cols = [f"count_{c}" for c in classes_to_track]
        max_count = merged_df[count_cols].max(axis=1)
        
        for c in classes_to_track:
            merged_df[f"tie_weight_{c}"] = (merged_df[f"count_{c}"] == max_count) * severity_order[c]
            
        winner_sev = merged_df[[f"tie_weight_{c}" for c in classes_to_track]].max(axis=1)
        merged_df["ensembleMode_pred"] = winner_sev.map(reverse_sev_map)
        
        # D) Vectorized Final Selection
        cond_am = merged_df["ensembleAvg_pred"] == merged_df["ensembleMode_pred"]
        cond_ac = merged_df["ensembleAvg_pred"] == merged_df["ensembleMaxConf_pred"]
        cond_mc = merged_df["ensembleMode_pred"] == merged_df["ensembleMaxConf_pred"]
        
        merged_df["sev_a"] = merged_df["ensembleAvg_pred"].map(severity_order)
        merged_df["sev_m"] = merged_df["ensembleMode_pred"].map(severity_order)
        merged_df["sev_c"] = merged_df["ensembleMaxConf_pred"].map(severity_order)
        merged_df["tie_pred"] = merged_df[["sev_a", "sev_m", "sev_c"]].max(axis=1).map(reverse_sev_map)
        
        merged_df["select"] = np.select(
            [cond_am, cond_ac, cond_mc],
            [merged_df["ensembleAvg_pred"], merged_df["ensembleAvg_pred"], merged_df["ensembleMode_pred"]],
            default=merged_df["tie_pred"]
        )
        
        merged_df["decision_logic"] = np.select(
            [cond_am, cond_ac, cond_mc],
            ["align_avg_mode", "align_avg_maxConf", "align_mode_maxConf"],
            default="tie_use_most_severe"
        )
        
        # E) Vectorized Confidence Scoring
        for c in classes_to_track:
            mask = merged_df["select"] == c
            merged_df.loc[mask, "select_prob"] = merged_df.loc[mask, f"ensembleAvg_prob_{c}"]
            
        merged_df["num_models_pred_cat"] = (merged_df[pred_cols].values == merged_df[["select"]].values).sum(axis=1)
        
        merged_df["conf_consist"] = np.select(
            [merged_df["num_models_pred_cat"] <= 3, merged_df["num_models_pred_cat"] == 4, merged_df["num_models_pred_cat"] == 5],
            [1, 2, 3], default=0
        )
        merged_df["conf_probability"] = np.select(
            [merged_df["select_prob"] < 0.5, merged_df["select_prob"] < 0.85],
            [1, 2], default=3
        )
        merged_df["conf_overall"] = (merged_df["conf_consist"] + merged_df["conf_probability"]) / 2
        merged_df["confidence"] = np.select(
            [merged_df["conf_overall"] <= 1.5, merged_df["conf_overall"] == 2.0, merged_df["conf_overall"] >= 2.5],
            ["low", "medium", "high"], default="issue_confidence"
        )
        
        # Clean up temporary math columns
        cols_to_drop = count_cols + [f"tie_weight_{c}" for c in classes_to_track] + ["sev_a", "sev_m", "sev_c", "tie_pred"]
        merged_df = merged_df.drop(columns=cols_to_drop)
        
        # Save Step 3 (now includes hrrr_file_path natively)
        final_out = os.path.join(out_ens_dir, f"finalpreds_FH{fh}.csv")
        merged_df.to_csv(final_out, index=False)
        
        print(f"    -> Success! Final ensemble saved with HRRR traceback.")

print("\n=== All Dates Processed Successfully! ===")