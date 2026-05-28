# Portions of this code were writen with the assistance of AI tools (Gemini)

import pandas as pd
import numpy as np
import os
import csv
import glob
from ast import literal_eval
from tqdm import tqdm  

# Scikit-Learn imports
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score

# TensorFlow/Keras imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K  # <--- Imported for memory clearing

# Import your hyperparameter grid
import hyp_grid 

# =========================================
# CONFIGURATION
# =========================================
CURRENT_FH = "24"
TEST_MODE = False  # <--- Set to False when you are ready for the real run!

# Paths 
tracker_dir = f"/home/csutter/DRIVE-clean/forecasting/data_trackers_withhrrr/FH{CURRENT_FH}/"
output_dir = f"/home/csutter/DRIVE-clean/forecasting/data_HT_results/FH{CURRENT_FH}/"
output_csv = os.path.join(output_dir, f"HT_results_FH{CURRENT_FH}.csv")

os.makedirs(output_dir, exist_ok=True)

# Variables
features = ["t2m", "r2", "asnow", "tp", "tcc", "uavg"]
target_col = "img_cat"
classes_to_track = ["snow_severe", "snow", "wet", "dry", "poor_viz"]

# =========================================
# SETUP OUTPUT FILE
# =========================================
csv_headers = [
    "tracker_name", "algorithm", "hyperparameters", 
    "n_labeled_total", "n_correct_total", "total_acc", "avg_recall"
]
for c in classes_to_track:
    csv_headers.extend([f"n_labeled_{c}", f"n_correct_{c}", f"recall_{c}"])

if not os.path.isfile(output_csv):
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

# =========================================
# MAIN EXECUTION LOOP
# =========================================
tracker_files = glob.glob(os.path.join(tracker_dir, "*.csv"))

if TEST_MODE:
    print("\n!!! RUNNING IN TEST MODE (1 Tracker, 2 Hyps per Algo) !!!\n")
    tracker_files = tracker_files[:1]

for t_idx, tracker_path in enumerate(tracker_files, 1):
    tracker_name = os.path.basename(tracker_path)
    print(f"\n--- Processing Tracker {t_idx}/{len(tracker_files)}: {tracker_name} ---")
    
    # 1. Load Data
    df = pd.read_csv(tracker_path)
    df = df.dropna(subset=features)
    
    # 2. Split Data (InnerTrain/InnerTest -> Train | InnerVal -> Test)
    train_mask = df["innerPhase"].isin(["innerTrain", "innerTest"])
    test_mask = df["innerPhase"] == "innerVal"
    
    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()
    
    X_train_raw = df_train[features].values
    y_train_raw = df_train[target_col].values
    X_test_raw = df_test[features].values
    y_test_raw = df_test[target_col].values
    
    # 3. Scale Data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    # 4. Encode Targets
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_raw)
    y_test_enc = le.transform(y_test_raw)
    
    weights = class_weight.compute_class_weight(
        "balanced", classes=np.unique(y_train_enc), y=y_train_enc
    )
    class_weights_dict = dict(enumerate(weights))

    # =========================================
    # ALGORITHM TESTING
    # =========================================
    algorithms_to_test = {
        "LogisticRegression": hyp_grid.logistic_HT,
        "GaussianNB": hyp_grid.gnb_HT,
        "SVM": hyp_grid.svm_HT,
        "RandomForest": hyp_grid.rf_HT,
        "DNN": hyp_grid.dnn_HT
    }
    
    if TEST_MODE:
        for key in algorithms_to_test:
            algorithms_to_test[key] = algorithms_to_test[key][:2]
            
    total_models = sum(len(grid) for grid in algorithms_to_test.values())
    current_model = 0
    
    for alg_name, grid in algorithms_to_test.items():
        for hyp in grid:
            current_model += 1
            print(f"  [{current_model}/{total_models}] Training {alg_name}...")
            
            # --- MODEL TRAINING ---
            if alg_name == "LogisticRegression":
                model = LogisticRegression(
                    multi_class="multinomial", solver="lbfgs", random_state=42,
                    max_iter=hyp["max_iter"], C=hyp["C"]
                )
                model.fit(X_train, y_train_enc)
                y_pred_enc = model.predict(X_test)
                
            elif alg_name == "GaussianNB":
                model = GaussianNB(var_smoothing=hyp["var_smoothing"])
                model.fit(X_train, y_train_enc)
                y_pred_enc = model.predict(X_test)
                
            elif alg_name == "SVM":
                model = SVC(
                    kernel=hyp["kernel"], C=hyp["C"], gamma=hyp["gamma"], 
                    class_weight="balanced", random_state=42,
                    cache_size=2000  # <--- 2GB Cache Fix Applied!
                )
                model.fit(X_train, y_train_enc)
                y_pred_enc = model.predict(X_test)
                
            elif alg_name == "RandomForest":
                if hyp["bootstrap"]:
                    model = RandomForestClassifier(
                        max_depth=hyp["max_depth"], n_estimators=hyp["n_estimators"],
                        max_features=hyp["max_features"], min_samples_leaf=hyp["min_samples_leaf"],
                        max_samples=hyp["max_samples"], bootstrap=True, 
                        class_weight="balanced", random_state=42, 
                        n_jobs=4  # <--- Thread limit fix applied!
                    )
                else:
                    model = RandomForestClassifier(
                        max_depth=hyp["max_depth"], n_estimators=hyp["n_estimators"],
                        max_features=hyp["max_features"], min_samples_leaf=hyp["min_samples_leaf"],
                        bootstrap=False, class_weight="balanced", random_state=42, 
                        n_jobs=4  # <--- Thread limit fix applied!
                    )
                model.fit(X_train, y_train_enc)
                y_pred_enc = model.predict(X_test)
                
            elif alg_name == "DNN":
                y_train_cat = to_categorical(y_train_enc)
                hidden_units = literal_eval(hyp["hidden_units"])
                
                model = Sequential()
                for i in range(hyp["hidden_layers"]):
                    if i == 0:
                        model.add(Dense(
                            hidden_units[i], input_shape=(X_train.shape[1],), activation="relu",
                            kernel_regularizer=regularizers.l2(hyp["l2_reg"])
                        ))
                    else:
                        model.add(Dense(
                            hidden_units[i], activation="relu",
                            kernel_regularizer=regularizers.l2(hyp["l2_reg"])
                        ))
                    model.add(Dropout(hyp["dropout"]))
                
                model.add(Dense(len(classes_to_track), activation="softmax"))
                model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
                
                model.fit(
                    X_train, y_train_cat, epochs=30, batch_size=128, 
                    verbose=0, class_weight=class_weights_dict
                )
                
                y_prob = model.predict(X_test, verbose=0)
                y_pred_enc = np.argmax(y_prob, axis=1)
                
                K.clear_session() # <--- Memory leak fix applied!

            # --- DECODE PREDICTIONS ---
            y_pred_strings = le.inverse_transform(y_pred_enc)
            
            # --- METRICS CALCULATION ---
            n_labeled_total = len(y_test_raw)
            n_correct_total = np.sum(y_test_raw == y_pred_strings)
            total_acc = n_correct_total / n_labeled_total if n_labeled_total > 0 else 0.0
            
            run_results = {
                "tracker_name": tracker_name,
                "algorithm": alg_name,
                "hyperparameters": str(hyp),
                "n_labeled_total": n_labeled_total,
                "n_correct_total": n_correct_total,
                "total_acc": total_acc
            }
            
            recalls = []
            
            for c in classes_to_track:
                true_mask = (y_test_raw == c)
                n_labeled = np.sum(true_mask)
                n_correct = np.sum((y_pred_strings == c) & true_mask)
                
                recall = (n_correct / n_labeled) if n_labeled > 0 else 0.0
                recalls.append(recall)
                
                run_results[f"n_labeled_{c}"] = n_labeled
                run_results[f"n_correct_{c}"] = n_correct
                run_results[f"recall_{c}"] = recall
            
            run_results["avg_recall"] = np.mean(recalls)
            
            # --- APPEND TO CSV ---
            with open(output_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_headers)
                writer.writerow(run_results)

print("\nModel training and hyperparameter tracking complete!")