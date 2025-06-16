import os
import time
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from config import *
from data import get_data_loader, AnomalyDetectionDatasets
from model.anomaly_detection_informer import AnomalyDetectionInformer
from utils.train_utils import train_model
from utils.metrics import evaluate_model, plot_confusion_matrix, plot_training_history

def preprocess_time_series(X_train, X_test):
    """Time series data preprocessing - Enhanced version"""
    # Combine data for standardization
    combined = np.vstack((X_train, X_test))

    # Handle NaN and infinite values
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)

    if TS_NORMALIZE:
        # Calculate mean and standard deviation for each feature
        means = np.nanmean(combined, axis=0)
        stds = np.nanstd(combined, axis=0)

        # Avoid division by zero
        zero_std_indices = np.where(stds == 0)[0]
        if len(zero_std_indices) > 0:
            print(f"Warning: {len(zero_std_indices)} features have zero std, replacing with 1.0")
            stds[zero_std_indices] = 1.0

        # Standardize
        X_train = (X_train - means) / stds
        X_test = (X_test - means) / stds

        # Check for outliers again
        X_train = np.clip(X_train, -10, 10)  # Limit to range [-10, 10]
        X_test = np.clip(X_test, -10, 10)

    return X_train, X_test

def main():
    # Declare global variables at the beginning of the function
    global WINDOW_SIZE, STRIDE, D_MODEL, E_LAYERS

    # Create output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_{DATASET_CHOICE}_{SMD_MACHINE_ID if DATASET_CHOICE == 'smd' else ''}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to directory: {output_dir}")

    # Use specific configuration for time series datasets
    if DATASET_CHOICE in ['msl', 'smap', 'smd']:
        # Use larger window size
        WINDOW_SIZE = 100
        STRIDE = 20

        # SMD specific configuration
        if DATASET_CHOICE == 'smd':
            D_MODEL = 128  # SMD has 38 features, needs a larger model
            E_LAYERS = 3
            print(f"Using SMD specific config: D_MODEL={D_MODEL}, E_LAYERS={E_LAYERS}")
        else:
            D_MODEL = 64
            E_LAYERS = 2

        print(f"Using time series config: WINDOW_SIZE={WINDOW_SIZE}, STRIDE={STRIDE}")

    # Load dataset
    print(f"\nLoading dataset: {DATASET_CHOICE.upper()}")
    data_path = DATA_PATHS[DATASET_CHOICE]
    kwargs = {}

    if DATASET_CHOICE == 'kdd':
        X_train, X_test, y_train, y_test, num_classes, class_names = get_data_loader(
            'kdd', data_path
        )
    elif DATASET_CHOICE == 'iot':
        # Load a small portion of the data to inspect column names
        try:
            sample_df = pd.read_csv(data_path, nrows=5)
            print("\nFirst 5 rows of the dataset:")
            print(sample_df)
            print("\nDataset column names:", sample_df.columns.tolist())
        except Exception as e:
            print(f"\nError reading dataset: {e}")

        X_train, X_test, y_train, y_test, num_classes, class_names = get_data_loader(
            'iot', data_path, task_type=TASK_TYPE
        )
    elif DATASET_CHOICE == 'msl':
        X_train, X_test, y_train, y_test, num_classes, class_names = get_data_loader(
            'msl', data_path
        )
    elif DATASET_CHOICE == 'smap':
        X_train, X_test, y_train, y_test, num_classes, class_names = get_data_loader(
            'smap', data_path
        )
    elif DATASET_CHOICE == 'smd':
        # Load SMD dataset
        kwargs['machine_id'] = SMD_MACHINE_ID
        X_train, X_test, y_train, y_test, num_classes, class_names = get_data_loader(
            'smd', data_path, **kwargs
        )
    else:
        raise ValueError(f"Unknown dataset choice: {DATASET_CHOICE}")

    # Print data distribution
    print(f"\nClass distribution in training set: Normal={np.sum(y_train == 0)}, Anomaly={np.sum(y_train == 1)}")
    print(f"Class distribution in test set: Normal={np.sum(y_test == 0)}, Anomaly={np.sum(y_test == 1)}")

    # ====== Time Series Preprocessing ======
    # All time series datasets need normalization
    if DATASET_CHOICE in ['msl', 'smap', 'smd']:
        print("\nPreprocessing time series data...")
        X_train, X_test = preprocess_time_series(X_train, X_test)
    else:
        print("\nSkipping time series preprocessing for IoT/KDD datasets")

    print(f"\nDataset information:")
    print(f"  Training set size: {len(X_train)}")
    print(f"  Test set size: {len(X_test)}")
    print(f"  Feature dimension: {X_train.shape[1]}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Class names: {class_names}")

    # Check data shape (ensure it's a 2D array)
    if len(X_train.shape) == 1:
        print(f"Reshaping data from 1D to 2D (adding feature dimension)")
        X_train = X_train.reshape(-1, 1)
        X_test = X_test.reshape(-1, 1)

    # Create datasets
    train_dataset = AnomalyDetectionDatasets(
        X_train, y_train,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        num_classes=num_classes
    )

    test_dataset = AnomalyDetectionDatasets(
        X_test, y_test,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        num_classes=num_classes
    )

    print(f"\nSequential dataset information:")
    print(f"  Number of training sequences: {len(train_dataset)}")
    print(f"  Number of test sequences: {len(test_dataset)}")
    print(f"  Sequence length: {WINDOW_SIZE}")

    # Determine device type
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Create data loaders (only use pin_memory on CUDA devices)
    pin_memory = (device.type == "cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=pin_memory
    )

    # Initialize model
    model = AnomalyDetectionInformer(
        num_features=X_train.shape[1],
        num_classes=num_classes,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        e_layers=E_LAYERS,
        dropout=DROPOUT
    )

    print("\nModel architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Train model
    trained_model, history = train_model(
        model,
        train_loader,
        test_loader,
        num_classes,
        output_dir,
        device,
        epochs=EPOCHS,
        lr=LEARNING_RATE
    )

    # Final evaluation
    model_path = os.path.join(output_dir, "best_model.pth")
    if os.path.exists(model_path):
        trained_model.load_state_dict(torch.load(model_path))
        trained_model = trained_model.to(device)

        test_metrics = evaluate_model(trained_model, test_loader, device, num_classes)

        # Print test results
        print("\n" + "=" * 80)
        print(f"{'Final Test Results':^80}")
        print("=" * 80)
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test Precision: {test_metrics['precision']:.4f}")
        print(f"Test Recall: {test_metrics['recall']:.4f}")
        print(f"Test F1 Score: {test_metrics['f1']:.4f}")

        # Plot confusion matrix
        if SAVE_PLOTS and test_metrics['confusion_matrix'] is not None:
            conf_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
            plot_confusion_matrix(
                test_metrics['confusion_matrix'],
                classes=class_names,
                save_path=conf_matrix_path,
                normalize=True,
                title=f'{DATASET_CHOICE.upper()} ({SMD_MACHINE_ID if DATASET_CHOICE == "smd" else ""}) Confusion Matrix'
            )
    else:
        print(f"\nWarning: Best model not found at {model_path}. Skipping final evaluation.")

    # Save full model
    if SAVE_MODEL:
        full_model_path = os.path.join(output_dir, "full_model.pth")
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'model_config': {
                'num_features': X_train.shape[1],
                'num_classes': num_classes,
                'd_model': D_MODEL,
                'n_heads': N_HEADS,
                'e_layers': E_LAYERS
            }
        }, full_model_path)
        print(f"\nFull model saved to: {full_model_path}")

if __name__ == "__main__":
    main()
