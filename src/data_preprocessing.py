# data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from imblearn.combine import SMOTETomek

# def select_top_features_mic(X, y, top_k=15):
#     selector = SelectKBest(score_func=mutual_info_classif, k=top_k)
#     X_new = selector.fit_transform(X, y)
#     selected_features = selector.get_support(indices=True)
#     return X_new, selected_features

# def load_and_preprocess_data(csv_path="dataset/diabetes_binary_health_indicators_BRFSS2015.csv", top_k_features=15):
#     df = pd.read_csv(csv_path)
#     X = df.drop(columns=["Diabetes_binary"])
#     y = df["Diabetes_binary"]

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     X_selected, selected_feature_indices = select_top_features_mic(X_scaled, y, top_k=top_k_features)

#     X_train, X_temp, y_train, y_temp = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)
#     X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

#     smt = SMOTETomek(random_state=42)
#     X_train_balanced, y_train_balanced = smt.fit_resample(X_train, y_train)

#     return X_train_balanced, X_val, X_test, y_train_balanced, y_val, y_test, selected_feature_indices

def load_and_preprocess_data(csv_path="dataset/diabetes_binary_health_indicators_BRFSS2015.csv"):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["Diabetes_binary"])
    y = df["Diabetes_binary"]

    print("Initial data:")
    print(" - X shape:", X.shape)
    print(" - y shape:", y.shape)

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply SMOTETomek to the entire dataset
    smt = SMOTETomek(random_state=42)
    X_balanced, y_balanced = smt.fit_resample(X_scaled, y)

    # Split the balanced data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print("Balanced data:")
    print(" - X_train shape:", X_train.shape)
    print(" - y_train shape:", y_train.shape)
    print(" - X_val shape:", X_val.shape)
    print(" - y_val shape:", y_val.shape)
    print(" - X_test shape:", X_test.shape)
    print(" - y_test shape:", y_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test


