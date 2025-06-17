import os
import time
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from config import CONFIG  # Use the new configuration interface
from data import get_data_loader, AnomalyDetectionDatasets
from model.anomaly_detection_informer import AnomalyDetectionInformer
from utils.train_utils import train_model
from utils.metrics import evaluate_model, plot_confusion_matrix, plot_training_history

def preprocess_time_series(X_train, X_test):
    """Time series data preprocessing"""
    # Combine data for standardization
    combined = np.vstack((X_train, X_test))

    # Handle NaN and infinite values
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)

    if CONFIG.TS_NORMALIZE:
        # Calculate mean and standard deviation for each feature
        means = np.nanmean(combined, axis=0)
        stds = np.nanstd(combined, axis=0)

        # Avoid division by zero
        zero_std_indices = np.where(stds == 0)[0]
        if len(zero_std_indices) > 0:
            print(f"Warning: {len(zero_std_indices)} features have zero standard deviation, replacing with 1.0")
            stds[zero_std_indices] = 1.0

        # Standardize
        X_train = (X_train - means) / stds
        X_test = (X_test - means) / stds

        # Check for outliers again
        X_train = np.clip(X_train, -10, 10)  # Limit to range [-10, 10]
        X_test = np.clip(X_test, -10, 10)

    return X_train, X_test

def main():
    # Get current dataset configuration
    dataset_config = CONFIG.dataset_config
    dataset_name = CONFIG.DATASET_CHOICE

    # Create output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Include machine ID for SMD dataset
    dataset_id = f"{dataset_name}"
    if dataset_name == 'smd' and hasattr(dataset_config, 'machine_id'):
        dataset_id += f"_{dataset_config.machine_id}"

    output_dir = f"results_{dataset_id}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to directory: {output_dir}")

    # Load dataset
    print(f"\nLoading dataset: {dataset_name.upper()}")

    # Prepare dataset loading parameters
    kwargs = {}

    # Add dataset-specific parameters
    if hasattr(dataset_config, 'params'):
        kwargs.update(dataset_config.params)

    # Add task type for network datasets
    if hasattr(dataset_config, 'task_type'):
        kwargs['task_type'] = dataset_config.task_type

    # Add additional parameters for specific datasets
    if dataset_name == 'smd':
        machine_id = dataset_config.params.get('machine_id', 'machine-1-1')
        kwargs['machine_id'] = machine_id

    if dataset_name == 'wadi' and 'version' in dataset_config.params:
        kwargs['version'] = dataset_config.params['version']

    # Load dataset
    X_train, X_test, y_train, y_test, num_classes, class_names = get_data_loader(
        dataset_name,
        dataset_config.path,
        **kwargs
    )

    # Print data distribution
    print(f"\nTraining set class distribution: Normal={np.sum(y_train == 0)}, Anomaly={np.sum(y_train == 1)}")
    print(f"Test set class distribution: Normal={np.sum(y_test == 0)}, Anomaly={np.sum(y_test == 1)}")

    # Time series preprocessing
    if dataset_name in ['msl', 'smap', 'smd', 'swat', 'wadi']:
        print("\nPreprocessing time series data...")
        X_train, X_test = preprocess_time_series(X_train, X_test)
    else:
        print("\nSkipping time series preprocessing (network datasets)")

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

    # Automatically select training mode
    if np.sum(y_train) == 0:  # No anomaly samples in the training set
        training_mode = 'unsupervised'
        print("\nWarning: No anomaly samples in the training set, switching to unsupervised training mode")
    else:
        training_mode = 'supervised'
        print("\nUsing supervised training mode")

    # Create datasets
    train_dataset = AnomalyDetectionDatasets(
        X_train, y_train,
        window_size=CONFIG.window_size,
        stride=CONFIG.stride,
        num_classes=num_classes,
        training_mode=training_mode
    )

    test_dataset = AnomalyDetectionDatasets(
        X_test, y_test,
        window_size=CONFIG.window_size,
        stride=CONFIG.stride,
        num_classes=num_classes,
        training_mode='supervised'
    )

    print(f"\nSequential dataset information:")
    print(f"  Number of training sequences: {len(train_dataset)}")
    print(f"  Number of test sequences: {len(test_dataset)}")
    print(f"  Sequence length: {CONFIG.window_size}")

    # Check dataset validity
    if len(X_test) == 0 or np.sum(y_test) == 0:
        print("Warning: Test set is empty or has no anomaly samples! Using alternative split strategy")

        # Combine training and test sets
        X_full = np.vstack((X_train, X_test)) if len(X_test) > 0 else X_train
        y_full = np.concatenate((y_train, y_test)) if len(y_test) > 0 else y_train

        # Resplit the dataset (80% training, 20% testing)
        split_idx = int(len(X_full) * 0.8)
        X_train = X_full[:split_idx]
        X_test = X_full[split_idx:]
        y_train = y_full[:split_idx]
        y_test = y_full[split_idx:]

        # Ensure there are anomaly samples in the test set
        if np.sum(y_test) == 0:
            print("Forcing addition of anomaly samples to the test set")
            # Find anomaly samples in the training set (if any)
            anomaly_indices = np.where(y_train == 1)[0]
            if len(anomaly_indices) > 0:
                # Move some anomaly samples to the test set
                move_count = min(10, len(anomaly_indices))
                move_indices = anomaly_indices[:move_count]

                # Update test set
                X_test = np.vstack((X_test, X_train[move_indices]))
                y_test = np.concatenate((y_test, y_train[move_indices]))

                # Update training set
                X_train = np.delete(X_train, move_indices, axis=0)
                y_train = np.delete(y_train, move_indices)

    # Get device information
    device = torch.device(CONFIG.DEVICE)
    print(f"\nUsing device: {device}")

    # Create data loaders (only use pin_memory on CUDA devices)
    pin_memory = (device.type == "cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG.BATCH_SIZE,
        shuffle=True,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG.BATCH_SIZE,
        shuffle=False,
        pin_memory=pin_memory
    )

    # Initialize model
    model = AnomalyDetectionInformer(
        num_features=X_train.shape[1],
        num_classes=num_classes,
        d_model=CONFIG.d_model,
        n_heads=CONFIG.N_HEADS,
        e_layers=CONFIG.e_layers,
        dropout=CONFIG.DROPOUT
    )

    # print("\nModel architecture:")
    # print(model)
    # print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Train model
    trained_model, history = train_model(
        model,
        train_loader,
        test_loader,
        num_classes,
        output_dir,
        device,
        epochs=CONFIG.EPOCHS,
        lr=CONFIG.LEARNING_RATE
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
        # if CONFIG.SAVE_PLOTS and test_metrics['confusion_matrix'] is not None:
        #     conf_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
        #     plot_confusion_matrix(
        #         test_metrics['confusion_matrix'],
        #         classes=class_names,
        #         save_path=conf_matrix_path,
        #         normalize=True,
        #         title=f'{dataset_name.upper()} Confusion Matrix'
        #     )
    else:
        print(f"\nWarning: Best model not found at {model_path}, skipping final evaluation")

    # Save full model
    if CONFIG.SAVE_MODEL:
        full_model_path = os.path.join(output_dir, "full_model.pth")
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'model_config': {
                'num_features': X_train.shape[1],
                'num_classes': num_classes,
                'd_model': CONFIG.d_model,
                'n_heads': CONFIG.N_HEADS,
                'e_layers': CONFIG.e_layers
            }
        }, full_model_path)
        print(f"\nFull model saved to: {full_model_path}")

if __name__ == "__main__":
    main()
