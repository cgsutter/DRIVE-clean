# Portions of this code were writen with the assistance of AI tools (Gemini)

import pandas as pd
import numpy as np
import os
import glob
from collections import Counter

# =========================================
# CONFIGURATION
# =========================================
FORECAST_HOURS = ["02","03", "04","05","06", "09", "12","15","18", "24","30","36","42","48"] # Update with your 14 FHs
TARGET_MODEL = "modelFH02" # Hardcoded to use ONLY the FH02-trained model

base_preds_dir = "/home/csutter/DRIVE-clean/forecasting/data_final_preds/"
output_ensemble_preds = "/home/csutter/DRIVE-clean/forecasting/data_ensembled_preds/"
output_ensemble_stats = "/home/csutter/DRIVE-clean/forecasting/data_ensembled_stats/"

classes_to_track = ["snow_severe", "snow", "wet", "dry", "poor_viz"]

# Severity hierarchy for tie-breakers
severity_order = {"snow_severe": 5, "poor_viz": 4, "snow": 3, "wet": 2, "dry": 1}

# =========================================
# HELPER FUNCTIONS
# =========================================
def get_mode_with_tiebreaker(preds):
    counts = Counter(preds)
    max_count = max(counts.values())
    modes = [k for k, v in counts.items() if v == max_count]
    
    if len(modes) > 1:
        # Tie-breaker: return the most severe weather class
        return max(modes, key=lambda x: severity_order.get(x, 0))
    return modes[0]

def get_most_severe(preds):
    return max(preds, key=lambda x: severity_order.get(x, 0))

def check_relaxed_ok(row):
    truth = row["img_cat"]
    pred = row["select"]  # <-- Updated to match the new column name
    
    if truth == pred: return True
    if truth == "snow_severe" and pred == "snow": return True
    if truth == "snow" and pred in ["snow_severe", "wet"]: return True
    if truth == "wet" and pred in ["snow", "dry"]: return True
    if truth == "dry" and pred == "wet": return True
    return False

# --- NEW CONFIDENCE SCORING FUNCTIONS ---
def get_select_prob(row):
    # Grabs the exact average probability for the class that was ultimately selected
    return row[f"ensembleAvg_prob_{row['select']}"]

def get_consistency_count(row):
    # Counts how many of the 5 models predicted the final selected category
    preds = [row[f"m{i}_pred"] for i in range(5)]
    return preds.count(row["select"])

def get_conf_consist(count):
    if count <= 3: return 1
    elif count == 4: return 2
    elif count == 5: return 3
    return 0

def get_conf_prob(prob):
    if prob < 0.5: return 1
    elif prob < 0.85: return 2
    else: return 3

def get_conf_qual(val):
    if val <= 1.5: return "low"
    elif val == 2.0: return "medium"
    elif val >= 2.5: return "high"
    return "issue_confidence"

# =========================================
# MAIN EXECUTION
# =========================================
for fh in FORECAST_HOURS:
    search_path = os.path.join(base_preds_dir, f"FH{fh}", f"*_details_{TARGET_MODEL}")
    target_dirs = glob.glob(search_path)
    
    if not target_dirs:
        print(f"Directory not found for FH{fh} and {TARGET_MODEL}. Ensure Script 3 has run.")
        continue
        
    model_dir = target_dirs[0]
    subdir_name = os.path.basename(model_dir)
    
    print(f"\n=== Ensembling: Evaluated on FH{fh} | Using {subdir_name} ===")
    
    out_pred_dir = os.path.join(output_ensemble_preds, f"FH{fh}", subdir_name)
    out_stat_dir = os.path.join(output_ensemble_stats, f"FH{fh}", subdir_name)
    os.makedirs(out_pred_dir, exist_ok=True)
    os.makedirs(out_stat_dir, exist_ok=True)
    
    ot_summary_stats = []
    
    for otnum in range(6): # OT0 through OT5
        print(f"  -> Processing Outer Test {otnum}...")
        
        ot_files = glob.glob(os.path.join(model_dir, f"*_OT{otnum}_m*.csv"))
        ot_files = sorted(ot_files)
        
        if len(ot_files) != 5:
            print(f"     [Warning] Found {len(ot_files)} files for OT{otnum}. Skipping.")
            continue
            
        dfs = []
        for m_idx, f in enumerate(ot_files):
            df = pd.read_csv(f)
            df = df[df["innerPhase"] == "NAOuterTest"].copy()
            
            rename_map = {"predicted_cat": f"m{m_idx}_pred"}
            for c in classes_to_track:
                rename_map[f"prob_{c}"] = f"m{m_idx}_prob_{c}"
            
            df = df.rename(columns=rename_map)
            
            if m_idx > 0:
                cols_to_keep = ["img_name"] + list(rename_map.values())
                df = df[cols_to_keep]
            
            dfs.append(df)
        
        merged_df = dfs[0]
        for df in dfs[1:]:
            merged_df = merged_df.merge(df, on="img_name")
        
        # --------------------------------------------------
        # ENSEMBLING LOGIC
        # --------------------------------------------------
        pred_cols = [f"m{i}_pred" for i in range(5)]
        
        # Method 1: Mode
        merged_df["ensembleMode_pred"] = merged_df[pred_cols].apply(get_mode_with_tiebreaker, axis=1)
        
        # Method 2: Average Probability
        for c in classes_to_track:
            prob_cols = [f"m{i}_prob_{c}" for i in range(5)]
            merged_df[f"ensembleAvg_prob_{c}"] = merged_df[prob_cols].mean(axis=1)
        
        avg_cols = [f"ensembleAvg_prob_{c}" for c in classes_to_track]
        merged_df["ensembleAvg_pred"] = merged_df[avg_cols].idxmax(axis=1).str.replace("ensembleAvg_prob_", "")
        
        # Method 3: Max Confidence
        all_prob_cols = [f"m{i}_prob_{c}" for i in range(5) for c in classes_to_track]
        
        def get_max_conf_pred(row):
            max_col = row[all_prob_cols].astype(float).idxmax()
            return max_col.split("_prob_")[1]
            
        merged_df["ensembleMaxConf_pred"] = merged_df.apply(get_max_conf_pred, axis=1)
        
        # Final Decision Logic
        def select_final_ensemble(row):
            a = row["ensembleAvg_pred"]
            m = row["ensembleMode_pred"]
            c = row["ensembleMaxConf_pred"]
            
            if a == m: return a, "align_avg_mode"
            if a == c: return a, "align_avg_maxConf"
            if m == c: return m, "align_mode_maxConf"
            
            return get_most_severe([a, m, c]), "tie_use_most_severe"
        
        # <-- Renamed to "select" here -->
        merged_df[["select", "decision_logic"]] = merged_df.apply(
            select_final_ensemble, axis=1, result_type="expand"
        )
        
        # --------------------------------------------------
        # CONFIDENCE SCORING LOGIC
        # --------------------------------------------------
        # 1. Grab the probability of the final selected class
        merged_df["select_prob"] = merged_df.apply(get_select_prob, axis=1)
        
        # 2. Count consistency
        merged_df["num_models_pred_cat"] = merged_df.apply(get_consistency_count, axis=1)
        
        # 3. Apply scoring
        merged_df["conf_consist"] = merged_df["num_models_pred_cat"].apply(get_conf_consist)
        merged_df["conf_probability"] = merged_df["select_prob"].apply(get_conf_prob)
        
        # 4. Overall Metric
        merged_df["conf_overall"] = (merged_df["conf_consist"] + merged_df["conf_probability"]) / 2
        merged_df["confidence"] = merged_df["conf_overall"].apply(get_conf_qual)

        # --------------------------------------------------
        # FINAL METRICS & SAVING
        # --------------------------------------------------
        merged_df["correct_strict"] = merged_df["img_cat"] == merged_df["select"]
        merged_df["correct_relaxed"] = merged_df.apply(check_relaxed_ok, axis=1)
        
        pred_filename = os.path.join(out_pred_dir, f"ensembled_preds_OT{otnum}.csv")
        merged_df.to_csv(pred_filename, index=False)
        
        ot_stats = {
            "OT_Fold": f"OT{otnum}",
            "total_ims": len(merged_df),
            "correct_strict": merged_df["correct_strict"].sum(),
            "correct_relaxed": merged_df["correct_relaxed"].sum()
        }
        
        for c in classes_to_track:
            sub = merged_df[merged_df["img_cat"] == c]
            ot_stats[f"nims_{c}"] = len(sub)
            ot_stats[f"correct_{c}"] = sub["correct_strict"].sum()
        
        ot_summary_stats.append(ot_stats)
        
    if ot_summary_stats:
        summary_df = pd.DataFrame(ot_summary_stats)
        summary_df.loc['Total'] = summary_df.sum(numeric_only=True)
        summary_df.at['Total', 'OT_Fold'] = "ALL_OTs"
        
        stat_filename = os.path.join(out_stat_dir, f"Ensemble_Summary_Stats.csv")
        summary_df.to_csv(stat_filename, index=False)

print("\nEnsemble Pipeline Complete!")