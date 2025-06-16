import os
import time
import torch
import pandas as pd
from torch.utils.data import DataLoader
from config import *
from data import get_data_loader, AnomalyDetectionDatasets
from model.anomaly_detection_informer import AnomalyDetectionInformer
from utils.train_utils import train_model
from utils.metrics import evaluate_model, plot_confusion_matrix, plot_training_history

def main():
    # Create output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_{DATASET_CHOICE}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to directory: {output_dir}")

    # Load dataset
    print(f"\nLoading dataset: {DATASET_CHOICE.upper()}")
    data_path = DATA_PATHS[DATASET_CHOICE]

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
    else:
        raise ValueError(f"Unknown dataset choice: {DATASET_CHOICE}")

    print(f"\nDataset information:")
    print(f"  Training set size: {len(X_train)}")
    print(f"  Test set size: {len(X_test)}")
    print(f"  Feature dimension: {X_train.shape[1]}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Class names: {class_names}")

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

    # Create data loaders (only use pin_memory on CUDA devices)
    pin_memory = (device.type == "cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=pin_memory  # Dynamically set pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=pin_memory  # Dynamically set pin_memory
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

    # Train model
    trained_model, history = train_model(
        model,
        train_loader,
        test_loader,  # Using test set as validation set
        num_classes,
        output_dir,
        device,  # Pass device information
        epochs=EPOCHS,
        lr=LEARNING_RATE
    )

    # Final evaluation
    model_path = os.path.join(output_dir, "best_model.pth")
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
            title=f'{DATASET_CHOICE.upper()} Confusion Matrix'
        )

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
