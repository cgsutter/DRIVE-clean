# Portions of this code were writen with the assistance of AI tools (Gemini)

import pandas as pd
import numpy as np
import os
import glob
import csv
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Scikit-Learn models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Keras/TensorFlow models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

# =========================================
# CONFIGURATION
# =========================================
TEST_MODE = False  # Set to False to run the full pipeline

FORECAST_HOURS = ["02","03", "04","05","06", "09", "12","15","18", "24","30","36","42","48"] # Update with your 14 FHs 

# Populate this with the exact winners from /home/csutter/DRIVE-clean/forecasting/c_model_selection.ipynb
WINNING_MODELS = {
    "modelFH02": {"algorithm": "RandomForest", "hyps": {"max_depth": 10, "max_samples": 0.5, "n_estimators": 300, "max_features": 2, "min_samples_leaf": 5, "bootstrap": True}},
    "modelFH09": {"algorithm": "RandomForest", "hyps": {"max_depth": 10, "max_samples": 0.5, "n_estimators": 100, "max_features": 2, "min_samples_leaf": 5, "bootstrap": True}},
    "modelFH24": {"algorithm": "RandomForest", "hyps": {"max_depth": None, "max_samples": 0.5, "n_estimators": 100, "max_features": 2, "min_samples_leaf": 5, "bootstrap": True}}
}

base_tracker_dir = "/home/csutter/DRIVE-clean/forecasting/data_trackers_withhrrr/"
output_preds_dir = "/home/csutter/DRIVE-clean/forecasting/data_final_preds/"
output_models_dir = "/home/csutter/DRIVE-clean/forecasting/data_final_models/"
output_stats_dir = "/home/csutter/DRIVE-clean/forecasting/data_final_stats/"

features = ["t2m", "r2", "asnow", "tp", "tcc", "uavg"]
target_col = "img_cat"
classes_to_track = ["snow_severe", "snow", "wet", "dry", "poor_viz"]

# --- ADDED NEW COLUMNS HERE ---
csv_headers = [
    "tracker_name", "model_id", "algorithm", "hyperparameters", "evaluation_subset",
    "n_labeled_total", "n_correct_total", "total_acc", "avg_recall"
]
for c in classes_to_track:
    csv_headers.extend([f"n_labeled_{c}", f"n_correct_{c}", f"recall_{c}"])

if TEST_MODE:
    print("\n!!! RUNNING IN TEST MODE (1 FH, 1 Tracker) !!!\n")
    FORECAST_HOURS = ["02"]

# =========================================
# MAIN EXECUTION
# =========================================
for fh in FORECAST_HOURS:
    print(f"\n=== Processing Forecast Hour: {fh} ===")
    
    # 1. Setup subdirectories and initialize Stats CSVs for this FH
    stats_csv_paths = {}
    for model_id, config in WINNING_MODELS.items():
        subdir = f"{config['algorithm']}_details_{model_id}"
        
        os.makedirs(os.path.join(output_preds_dir, f"FH{fh}", subdir), exist_ok=True)
        os.makedirs(os.path.join(output_models_dir, f"FH{fh}", subdir), exist_ok=True)
        
        stats_dir = os.path.join(output_stats_dir, f"FH{fh}", subdir)
        os.makedirs(stats_dir, exist_ok=True)
        
        stats_path = os.path.join(stats_dir, "Final_Stats.csv")
        stats_csv_paths[model_id] = stats_path
        
        if not os.path.isfile(stats_path):
            with open(stats_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(csv_headers)

    # 2. Iterate through the Tracker Data
    tracker_files = glob.glob(os.path.join(base_tracker_dir, f"FH{fh}", "*.csv"))
    if TEST_MODE: tracker_files = tracker_files[:1]

    for t_idx, tracker_path in enumerate(tracker_files, 1):
        tracker_name = os.path.basename(tracker_path)
        tracker_base = tracker_name.replace(".csv", "")
        print(f"  [{t_idx}/{len(tracker_files)}] Tracker: {tracker_name}")
        
        df = pd.read_csv(tracker_path)
        df_clean = df.dropna(subset=features).copy()
        
        # Training Split (innerTrain + innerTest)
        train_mask = df_clean["innerPhase"].isin(["innerTrain", "innerTest"])
        df_train = df_clean[train_mask].copy()
        
        X_train_raw = df_train[features].values
        y_train_raw = df_train[target_col].values
        X_full_raw = df_clean[features].values
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_full = scaler.transform(X_full_raw)
        
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train_raw)
        
        from sklearn.utils import class_weight
        weights = class_weight.compute_class_weight("balanced", classes=np.unique(y_train_enc), y=y_train_enc)
        class_weights_dict = dict(enumerate(weights))

        # 3. Train and process each winning model architecture
        for model_id, config in WINNING_MODELS.items():
            alg_name = config["algorithm"]
            hyps = config["hyps"]
            subdir = f"{alg_name}_details_{model_id}"
            
            # --- MODEL TRAINING ---
            if alg_name == "LogisticRegression":
                model = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=42, **hyps)
                model.fit(X_train, y_train_enc)
                y_pred_enc_full = model.predict(X_full)
                joblib.dump(model, os.path.join(output_models_dir, f"FH{fh}", subdir, f"{model_id}_{tracker_base}.joblib"))
                
            elif alg_name == "GaussianNB":
                model = GaussianNB(**hyps)
                model.fit(X_train, y_train_enc)
                y_pred_enc_full = model.predict(X_full)
                joblib.dump(model, os.path.join(output_models_dir, f"FH{fh}", subdir, f"{model_id}_{tracker_base}.joblib"))
                
            elif alg_name == "SVM":
                model = SVC(class_weight="balanced", random_state=42, cache_size=2000, **hyps)
                model.fit(X_train, y_train_enc)
                y_pred_enc_full = model.predict(X_full)
                joblib.dump(model, os.path.join(output_models_dir, f"FH{fh}", subdir, f"{model_id}_{tracker_base}.joblib"))
                
            elif alg_name == "RandomForest":
                if hyps.get("bootstrap", False):
                    model = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=4, **hyps)
                else:
                    hyps_clean = {k: v for k, v in hyps.items() if k != "max_samples"}
                    model = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=4, **hyps_clean)
                
                model.fit(X_train, y_train_enc)
                y_pred_enc_full = model.predict(X_full)
                joblib.dump(model, os.path.join(output_models_dir, f"FH{fh}", subdir, f"{model_id}_{tracker_base}.joblib"))
                
            elif alg_name == "DNN":
                y_train_cat = to_categorical(y_train_enc)
                from ast import literal_eval
                hidden_units = literal_eval(hyps["hidden_units"]) if isinstance(hyps["hidden_units"], str) else hyps["hidden_units"]
                
                model = Sequential()
                for i in range(hyps["hidden_layers"]):
                    if i == 0:
                        model.add(Dense(hidden_units[i], input_shape=(X_train.shape[1],), activation="relu", kernel_regularizer=regularizers.l2(hyps["l2_reg"])))
                    else:
                        model.add(Dense(hidden_units[i], activation="relu", kernel_regularizer=regularizers.l2(hyps["l2_reg"])))
                    model.add(Dropout(hyps["dropout"]))
                
                model.add(Dense(len(classes_to_track), activation="softmax"))
                model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
                
                model.fit(X_train, y_train_cat, epochs=30, batch_size=128, verbose=0, class_weight=class_weights_dict)
                
                y_prob = model.predict(X_full, verbose=0)
                y_pred_enc_full = np.argmax(y_prob, axis=1)
                
                model.save(os.path.join(output_models_dir, f"FH{fh}", subdir, f"{model_id}_{tracker_base}.h5"))
                K.clear_session()
            

            # --- SAVE PREDICTIONS ---
            df_pred = df_clean.copy()
            df_pred["predicted_cat"] = le.inverse_transform(y_pred_enc_full)
            
            # Extract raw probabilities and map them to the class names
            # (Note: If you ever switch back to SVM, you must add probability=True to the SVC instantiation)
            y_prob_full = model.predict_proba(X_full)
            for i, class_name in enumerate(le.classes_):
                df_pred[f"prob_{class_name}"] = y_prob_full[:, i]
            
            pred_out_path = os.path.join(output_preds_dir, f"FH{fh}", subdir, f"preds_{tracker_name}")
            df_pred.to_csv(pred_out_path, index=False)

            
            # --- CALCULATE & SAVE STATS ---
            subsets = {
                "Full_Dataset": df_pred,
                "Validation_Only": df_pred[df_pred["innerPhase"] == "innerVal"],
                "Training_Only": df_pred[df_pred["innerPhase"].isin(["innerTrain", "innerTest"])],
                "Testing_Only": df_pred[df_pred["innerPhase"] == "NAOuterTest"]
            }
            
            for subset_name, subset_df in subsets.items():
                if len(subset_df) == 0: continue
                    
                y_true = subset_df[target_col].values
                y_pred = subset_df["predicted_cat"].values
                
                n_labeled_total = len(y_true)
                n_correct_total = np.sum(y_true == y_pred)
                total_acc = n_correct_total / n_labeled_total if n_labeled_total > 0 else 0.0
                
                # --- ADDED NEW DATA TO RESULTS DICTIONARY HERE ---
                run_results = {
                    "tracker_name": tracker_name,
                    "model_id": model_id,
                    "algorithm": alg_name,
                    "hyperparameters": str(hyps),
                    "evaluation_subset": subset_name,
                    "n_labeled_total": n_labeled_total,
                    "n_correct_total": n_correct_total,
                    "total_acc": total_acc
                }
                
                recalls = []
                for c in classes_to_track:
                    true_mask = (y_true == c)
                    n_labeled = np.sum(true_mask)
                    n_correct = np.sum((y_pred == c) & true_mask)
                    
                    recall = (n_correct / n_labeled) if n_labeled > 0 else 0.0
                    recalls.append(recall)
                    
                    run_results[f"n_labeled_{c}"] = n_labeled
                    run_results[f"n_correct_{c}"] = n_correct
                    run_results[f"recall_{c}"] = recall
                
                run_results["avg_recall"] = np.mean(recalls)
                
                with open(stats_csv_paths[model_id], "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=csv_headers)
                    writer.writerow(run_results)

print("\nPipeline complete. Models, Predictions, and Stats successfully generated and isolated.")