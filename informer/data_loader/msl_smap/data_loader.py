import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_msl_smap_data(base_path, dataset_type):
    """
    Load MSL or SMAP dataset
    :param base_path: Dataset root directory path
    :param dataset_type: 'msl' or 'smap'
    :return: X_train, X_test, y_train, y_test, num_classes, class_names
    """
    # Read annotation information
    anomaly_df = pd.read_csv(os.path.join(base_path, 'labeled_anomalies.csv'))

    # Filter specified type of dataset
    dataset_df = anomaly_df[anomaly_df['spacecraft'] == dataset_type.upper()]

    # Initialize data_loader containers
    all_train_data = []
    all_test_data = []
    all_test_labels = []

    # Traverse each time series file
    for idx, row in dataset_df.iterrows():
        chan_id = row['chan_id']

        # Load training data_loader
        train_path = os.path.join(base_path, 'train', f"{chan_id}.npy")
        train_data = np.load(train_path)
        all_train_data.append(train_data)

        # Load test data_loader
        test_path = os.path.join(base_path, 'test', f"{chan_id}.npy")
        test_data = np.load(test_path)
        all_test_data.append(test_data)

        # Create test labels (0=normal, 1=anomaly)
        test_labels = np.zeros(len(test_data), dtype=int)
        anomaly_regions = eval(row['anomaly_sequences'])

        for start, end in anomaly_regions:
            # Ensure indices are within range
            start = max(0, min(start, len(test_data) - 1))
            end = max(0, min(end, len(test_data) - 1))
            test_labels[start:end + 1] = 1

        all_test_labels.append(test_labels)

    # Merge data_loader from all channels
    X_train = np.concatenate(all_train_data)
    X_test = np.concatenate(all_test_data)
    y_test = np.concatenate(all_test_labels)

    # Training data_loader has no labels, all marked as normal
    y_train = np.zeros(len(X_train), dtype=int)

    # Convert to 2D array (n_samples, n_features)
    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)

    # Dataset information
    num_classes = 2
    class_names = ['Normal', 'Anomaly']

    print(f"\n{dataset_type.upper()} Dataset Summary:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Anomaly ratio: {np.sum(y_test) / len(y_test):.4f}")

    return X_train, X_test, y_train, y_test, num_classes, class_names
