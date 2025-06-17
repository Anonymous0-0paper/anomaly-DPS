import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
import torch
from sklearn.metrics import roc_curve, auc

def evaluate_model(model, loader, device, num_classes, training_mode='supervised'):
    """Support supervised and unsupervised modes"""
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []  # For storing anomaly scores

    # Additional recording of reconstruction error for unsupervised mode
    if training_mode == 'unsupervised':
        reconstruction_errors = []

    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device)
            y = batch['y'].cpu().numpy()
            all_labels.extend(y)

            if training_mode == 'unsupervised':
                # Unsupervised mode: Get reconstruction error
                errors = model.predict_anomaly(x).cpu().numpy()
                reconstruction_errors.extend(errors)
                all_scores.extend(errors)  # Save anomaly scores
            else:
                # Supervised mode: Get classification logits
                logits, _ = model(x, training_mode='supervised')
                probs = torch.softmax(logits, dim=1).cpu().numpy()

                # For binary classification, take the probability of the positive class; for multi-class, take the predicted probability
                if num_classes == 2:
                    scores = probs[:, 1]  # Probability of the positive class as the anomaly score
                else:
                    scores = np.max(probs, axis=1)  # Maximum probability as confidence

                all_scores.extend(scores)
                preds = np.argmax(probs, axis=1)
                all_preds.extend(preds)

    # Unsupervised mode: Use reconstruction error for prediction
    if training_mode == 'unsupervised' and len(reconstruction_errors) > 0:
        # Dynamically find the best threshold (based on maximizing F1)
        best_f1 = 0
        best_threshold = 0
        errors = np.array(reconstruction_errors)

        # Try multiple thresholds
        thresholds = np.quantile(errors, np.linspace(0.5, 0.99, 50))
        for threshold in thresholds:
            temp_preds = (errors > threshold).astype(int)
            f1 = f1_score(all_labels, temp_preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        # Use the best threshold for prediction
        all_preds = (errors > best_threshold).astype(int)
        print(f"Unsupervised mode best threshold: {best_threshold:.4f}, F1: {best_f1:.4f}")

    # Calculate metrics
    if len(all_preds) == 0 and training_mode == 'supervised':
        # For supervised mode, use classification prediction
        all_preds = np.argmax(np.array(all_scores), axis=1) if num_classes > 2 else (np.array(all_scores) > 0.5).astype(int)

    if len(all_preds) == 0:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'confusion_matrix': None
        }

    accuracy = accuracy_score(all_labels, all_preds)

    if num_classes > 2:
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    else:
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Calculate AUC (binary classification only)
    auc_score = 0.0
    if num_classes == 2 and len(all_scores) > 0:
        fpr, tpr, _ = roc_curve(all_labels, all_scores)
        auc_score = auc(fpr, tpr)

    result = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score,
        'confusion_matrix': conf_matrix
    }

    # Add reconstruction error information for unsupervised mode
    if training_mode == 'unsupervised' and len(reconstruction_errors) > 0:
        result['recon_error'] = np.mean(reconstruction_errors)
        result['best_threshold'] = best_threshold

    return result

def plot_training_history(history, save_path):
    """Plot training history charts"""
    plt.figure(figsize=(12, 5))

    # Training loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.title('Training Loss Change')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Validation metrics plot
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Validation Accuracy', color='blue')
    plt.plot(history['val_f1'], label='Validation F1 Score', color='orange')
    plt.title('Validation Metrics Change')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training history chart saved to: {save_path}")

def plot_confusion_matrix(conf_matrix, classes, save_path, normalize=False, title='Confusion Matrix'):
    """Plot confusion matrix"""
    if normalize:
        # Prevent division by zero
        row_sums = conf_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1
        conf_matrix = conf_matrix.astype('float') / row_sums[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        cbar=False
    )
    plt.title(title)
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")
