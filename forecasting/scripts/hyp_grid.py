# 1. Logistic Regression
logistic_HT = [
    {"max_iter": m, "C": C}
    for m in [100, 300, 500]
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]
]

# Granular steps (e.g. 1e-11) removed
# 2. Gaussian Naive Bayes 
gnb_HT = [{"var_smoothing": v} for v in [1e-12, 1e-10, 1e-8, 1e-6]]

# Adjustments (too much for 6 vars): removed C = 100, removed gamma = 10
# 3. Support Vector Machine 
svm_HT = [
    {"kernel": "rbf", "C": C, "gamma": gamma}
    for C in [1, 0.1, 1e-2, 1e-3, 10]
    for gamma in ["scale", 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1]
]

# Adjustments (too much for 6 vars): stop at 3 layers
# 4. Deep Neural Network 
dnn_configs = [
    {"hidden_layers": 3, "hidden_units": "[64, 32, 16]"},
    {"hidden_layers": 2, "hidden_units": "[32, 16]"},
    {"hidden_layers": 1, "hidden_units": "[16]"},
]

dnn_HT = [
    {**config, "dropout": dropout, "l2_reg": l2_reg}
    for config in dnn_configs
    for dropout in [0.0, 0.2, 0.4, 0.6, 0.8]
    for l2_reg in [0.0, 0.001, 0.01, 0.1]
]

# Adjustments (too much for 6 vars): removed 20 from max depth. Removed 300. Adjusted max features to 2,4,6
# 5. Random Forest 
rf_HT = []
for depth in [5, 10, None]: # None handles your final block that had no depth limit
    for n_est in [25, 100, 300]:
        for max_feat in [2, 4, 6]: 
            for min_leaf in [1, 5]:
                # Bootstrap False (does not use max_samples)
                rf_HT.append({
                    "max_depth": depth, 
                    "n_estimators": n_est, 
                    "max_features": max_feat, 
                    "min_samples_leaf": min_leaf, 
                    "bootstrap": False
                })
                # Bootstrap True (uses max_samples)
                for samples in [0.5, 0.75, 1.0]:
                    rf_HT.append({
                        "max_depth": depth, 
                        "max_samples": samples, # only here when bootstrap is true
                        "n_estimators": n_est, 
                        "max_features": max_feat, 
                        "min_samples_leaf": min_leaf, 
                        "bootstrap": True
                    })

