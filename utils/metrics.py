import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
import torch

def evaluate_model(model, loader, device, num_classes):
    """Evaluate model performance"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            x, y = batch['x'].to(device), batch['y'].to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    if len(all_preds) == 0:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'confusion_matrix': None
        }

    # Calculate metrics (add zero division protection)
    accuracy = accuracy_score(all_labels, all_preds)

    if num_classes > 2:  # Multiclass
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    else:  # Binary classification
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

    conf_matrix = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix
    }

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
