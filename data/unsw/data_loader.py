import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_unsw_data(base_path):
    """Load UNSW-NB15 dataset"""
    train_df = pd.read_csv(f"{base_path}/UNSW_NB15_training-set.csv")
    test_df = pd.read_csv(f"{base_path}/UNSW_NB15_testing-set.csv")

    # Combine data to ensure consistent encoding
    combined = pd.concat([train_df, test_df])

    # Drop irrelevant columns
    cols_to_drop = ['id', 'attack_cat']  # Attack categories can be used for multi-class classification
    combined = combined.drop(cols_to_drop, axis=1, errors='ignore')

    # Encode categorical features
    cat_cols = ['proto', 'service', 'state']
    for col in cat_cols:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))

    # Separate features and labels
    X = combined.drop('label', axis=1)
    y = combined['label'].values

    # Standardize numerical features
    num_cols = X.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    # Restore datasets according to original split
    X_train = X.iloc[:len(train_df)]
    X_test = X.iloc[len(train_df):]
    y_train = y[:len(train_df)]
    y_test = y[len(train_df):]

    # Multi-class support
    num_classes = 2  # Binary classification
    class_names = ['Normal', 'Attack']

    return X_train.values, X_test.values, y_train, y_test, num_classes, class_names
