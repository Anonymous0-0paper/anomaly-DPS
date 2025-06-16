import os
import numpy as np
import pandas as pd

def load_smd_data(base_path, machine_id):
    """
    Load SMD dataset
    :param base_path: Dataset root directory path
    :param machine_id: Machine ID, e.g., 'machine-1-1'
    :return: X_train, X_test, y_train, y_test, num_classes, class_names
    """
    # Construct file paths
    train_path = os.path.join(base_path, 'train', f'{machine_id}.txt')
    test_path = os.path.join(base_path, 'test', f'{machine_id}.txt')
    label_path = os.path.join(base_path, 'test_label', f'{machine_id}.txt')

    # Check if files exist
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")

    print(f"Loading SMD data for machine: {machine_id}")
    print(f"  Train data: {train_path}")
    print(f"  Test data: {test_path}")
    print(f"  Test labels: {label_path}")

    # Load data - handle possible space/comma separation
    def load_txt_file(file_path):
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                # Try both delimiters: space and comma
                if ',' in line:
                    values = line.strip().split(',')
                else:
                    values = line.strip().split()
                # Convert to float
                try:
                    data.append([float(v) for v in values])
                except ValueError:
                    # Skip empty lines or invalid lines
                    continue
        return np.array(data)

    # Load training data
    X_train = load_txt_file(train_path).astype(np.float32)

    # Load test data
    X_test = load_txt_file(test_path).astype(np.float32)

    # Load test labels
    y_test = load_txt_file(label_path).astype(np.int32)

    # Validate data dimensions
    if y_test.ndim > 1:
        # If labels are multiple columns, take the first column
        y_test = y_test[:, 0]

    # Training data has no labels, all marked as normal
    y_train = np.zeros(X_train.shape[0], dtype=np.int32)

    # Check dimension consistency
    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError(f"Test data and label quantities do not match: {X_test.shape[0]} vs {y_test.shape[0]}")

    # Dataset information
    num_classes = 2
    class_names = ['Normal', 'Anomaly']

    print(f"\nSMD Dataset ({machine_id}) Summary:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Anomaly ratio: {np.sum(y_test) / len(y_test):.6f}")
    print(f"  Feature dimension: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test, num_classes, class_names
