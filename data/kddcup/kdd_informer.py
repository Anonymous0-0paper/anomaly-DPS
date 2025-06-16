import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# 1. Dataset loading and preprocessing
def load_and_preprocess_kdd99(file_path, test_size=0.2, random_state=42):
    # Define column names
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
        'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
        'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
        'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
    ]

    # Load data
    df = pd.read_csv(file_path, header=None, names=columns)

    # Get all unique tags and map to numeric IDs
    all_labels = df['label'].unique()
    label_to_id = {label: idx for idx, label in enumerate(all_labels)}
    num_classes = len(all_labels)

    print(f" {num_classes} classes found:")
    for label, idx in label_to_id.items():
        print(f"  {label} -> {idx}")

    # Map labels to category IDs
    df['target'] = df['label'].map(label_to_id)

    # Extract features and labels
    X = df.drop(['label', 'target'], axis=1)
    y = df['target'].values  # Using NumPy directly

    # Separate numerical and categorical features
    numerical_features = [
        'duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot',
        'num_failed_logins', 'num_compromised', 'root_shell', 'su_attempted',
        'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
        'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
        'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
    ]

    categorical_features = ['protocol_type', 'service', 'flag']

    # Category feature encoding
    for col in categorical_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Normalize numerical features
    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])

    # Divide into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train.values, X_test.values, y_train, y_test, num_classes, label_to_id


# 2. Dataset class
class KDD99Dataset(Dataset):
    def __init__(self, features, labels, max_seq_len=10):
        """
        :param features: feature array (num_samples, num_features)
        :param labels: label array (num_samples,) - already integer IDs
        :param max_seq_len: pseudo sequence length
        """
        self.features = features
        self.labels = labels  # Already a NumPy array of integer IDs
        self.max_seq_len = max_seq_len
        self.num_features = features.shape[1]

        # Create sample sequence
        self.samples = []
        for i in range(0, len(features), max_seq_len):
            end_idx = min(i + max_seq_len, len(features))
            self.samples.append((i, end_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        start, end = self.samples[idx]
        actual_len = end - start

        # Feature sequence
        seq_features = self.features[start:end]
        padded_features = np.zeros((self.max_seq_len, self.num_features))
        padded_features[:actual_len] = seq_features

        # Tag sequence
        seq_labels = self.labels[start:end]
        padded_labels = np.full(self.max_seq_len, -100)  # -100 means ignore the position
        padded_labels[:actual_len] = seq_labels

        return {
            'x': torch.tensor(padded_features, dtype=torch.float32),
            'y': torch.tensor(padded_labels, dtype=torch.long),
            'seq_len': actual_len
        }


# 3. Custom collate function
def kdd_collate_fn(batch):
    """Processing batches of variable-length sequences"""
    xs = torch.stack([item['x'] for item in batch])
    ys = torch.stack([item['y'] for item in batch])
    seq_lens = torch.tensor([item['seq_len'] for item in batch])
    return xs, ys, seq_lens


# 4. Simplified Informer model (multi-classification)
class KDDInformer(nn.Module):
    def __init__(self, num_features, num_classes=23, d_model=128, n_heads=8, e_layers=3):
        super().__init__()

        # Input embedding layer
        self.input_embedding = nn.Linear(num_features, d_model)

        # Positional encoding
        self.position_embedding = nn.Embedding(1000, d_model)  # Assume the maximum sequence length is 1000

        # Informer encoder simplified version
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                batch_first=True  # Using batch_first format
            )
            for _ in range(e_layers)
        ])

        # Classification Head - Multi-classification Output
        self.classifier = nn.Linear(d_model, num_classes)  # 直接输出类别数

    def forward(self, x, seq_len):
        batch_size, seq_len_max, num_features = x.shape

        # Input embedding
        x_emb = self.input_embedding(x)  # (batch_size, seq_len, d_model)

        # Positional Embedding
        positions = torch.arange(seq_len_max, device=x.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)  # (1, seq_len, d_model)

        # Merge embeddings
        combined = x_emb + pos_emb

        # Create attention mask
        mask = torch.zeros((batch_size, seq_len_max), device=x.device)
        for i, length in enumerate(seq_len):
            mask[i, length:] = 1  # Fill position is 1
        mask = mask.bool()

        # Pass through encoder layers (batch_first format)
        encoder_output = combined
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, src_key_padding_mask=mask)

        # Classification prediction
        logits = self.classifier(encoder_output)

        return logits


# 5. Function to calculate category weights
def calculate_class_weights(y_train, num_classes):
    """Calculate the weight of each category"""
    # Calculate the number of samples for each category
    class_counts = np.bincount(y_train, minlength=num_classes)

    print("\nCategory distribution statistics:")
    for i, count in enumerate(class_counts):
        print(f"  Category {i}: {count} samples")

    # Avoiding division by zero errors
    class_counts = np.maximum(class_counts, 1)

    # Calculate weight (inverse of frequency)
    weights = 1.0 / np.sqrt(class_counts)

    # Normalize weights
    weights = weights / weights.sum() * num_classes

    return torch.tensor(weights, dtype=torch.float32)


# 6. Training function
def train_model(model, train_loader, val_loader, num_classes, class_weights, epochs=5, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Use weighted loss function
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        ignore_index=-100
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for x, y, seq_len in train_loader:
            x, y, seq_len = x.to(device), y.to(device), seq_len.to(device)

            optimizer.zero_grad()
            logits = model(x, seq_len)

            # Calculate loss
            loss = criterion(logits.view(-1, num_classes), y.view(-1))

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)

        # Verification
        val_metrics, _ = evaluate_model(
            model,
            val_loader,
            device,
            criterion,
            num_classes
        )
        # Get validation loss and accuracy
        val_loss = val_metrics['avg_loss']
        val_acc = val_metrics['accuracy']

        print(f"Epoch {epoch + 1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

    print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")


def evaluate_model(model, loader, device, criterion=None, num_classes=5, class_names=None, plot_confusion=False):
    model.eval()
    total_loss = 0.0

    # Store all predictions and true labels
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y, seq_len in loader:
            x, y, seq_len = x.to(device), y.to(device), seq_len.to(device)
            logits = model(x, seq_len)

            # Calculate Loss
            if criterion is not None:
                loss = criterion(logits.view(-1, num_classes), y.view(-1))
                total_loss += loss.item() * x.size(0)

            # Get prediction results
            preds = torch.argmax(logits, dim=-1)
            mask = (y != -100)  # Only consider non-filled positions

            # Collect valid predictions and labels
            valid_preds = preds[mask]
            valid_labels = y[mask]

            all_preds.extend(valid_preds.cpu().numpy())
            all_labels.extend(valid_labels.cpu().numpy())

    # Make sure we have some samples
    if len(all_preds) == 0 or len(all_labels) == 0:
        print("Warning: No valid samples to evaluate")
        return {
            'accuracy': 0.0,
            'avg_loss': 0.0 if criterion is None else total_loss,
            'class_metrics': [],
            'macro_precision': 0.0,
            'macro_recall': 0.0,
            'macro_f1': 0.0,
            'weighted_precision': 0.0,
            'weighted_recall': 0.0,
            'weighted_f1': 0.0
        }, None

    # Calculate the overall accuracy
    accuracy = (np.array(all_preds) == np.array(all_labels)).sum() / len(all_labels)
    avg_loss = total_loss / len(loader.dataset) if criterion is not None else 0.0

    # Convert to numpy array for manipulation
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)

    # Initialize the indicator array for each category
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    support = np.zeros(num_classes, dtype=int)

    # Calculate the index for each category
    for i in range(num_classes):
        # The true label is the sample of the current category
        true_i = (all_labels_np == i)

        # Predict samples with labels of current categories
        pred_i = (all_preds_np == i)

        # TP
        true_positives = np.sum(true_i & pred_i)

        # FP
        false_positives = np.sum(pred_i & ~true_i)

        # FN
        false_negatives = np.sum(true_i & ~pred_i)

        # Support (number of samples)
        support[i] = np.sum(true_i)

        # Calculate precision (avoid division by zero)
        if true_positives + false_positives > 0:
            precision[i] = true_positives / (true_positives + false_positives)
        else:
            precision[i] = 0.0

        # Calculate recall (avoid division by zero)
        if true_positives + false_negatives > 0:
            recall[i] = true_positives / (true_positives + false_negatives)
        else:
            recall[i] = 0.0

        # Calculate F1 score (avoid division by zero)
        if precision[i] + recall[i] > 0:
            f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        else:
            f1[i] = 0.0

    # Calculating macro-average and weighted average
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)

    weighted_precision = np.sum(precision * support) / np.sum(support)
    weighted_recall = np.sum(recall * support) / np.sum(support)
    weighted_f1 = np.sum(f1 * support) / np.sum(support)

    # Create a result dictionary
    metrics = {
        'accuracy': accuracy,
        'avg_loss': avg_loss,
        'class_metrics': [],
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1
    }

    # Add indicators for each category
    for i in range(num_classes):
        class_metric = {
            'class_id': i,
            'class_name': class_names[i] if class_names else f'Class {i}',
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': support[i]
        }
        metrics['class_metrics'].append(class_metric)

    # Calculate the confusion matrix
    conf_matrix = None
    if num_classes > 1 and len(all_labels) > 0:
        conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

        # Filling the confusion matrix
        for true_label, pred_label in zip(all_labels, all_preds):
            conf_matrix[true_label][pred_label] += 1

        # if plot_confusion:
        #     plot_confusion_matrix(conf_matrix, class_names, normalize=True)

    return metrics, conf_matrix

# Draw confusion matrix
def plot_confusion_matrix(conf_matrix, class_names, normalize=False, cmap=plt.cm.Blues):
    """
    Draw a visualization chart of the confusion matrix
    """
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        title = 'Normalized confusion matrix'
    else:
        title = 'Unnormalized confusion matrix'

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save the image
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

# 8. Main function
def main():
    # Load and preprocess data
    file_path = "kddcup.data"  # Replace with the actual file path
    X_train, X_test, y_train, y_test, num_classes, label_to_id = load_and_preprocess_kdd99(file_path)

    print(f"\nDataset:")
    print(f"  Number of training set samples: {len(X_train)}")
    print(f"  Number of test set samples: {len(X_test)}")
    print(f"  Feature Dimension: {X_train.shape[1]}")
    print(f"  Number of categories: {num_classes}")

    # Calculate category weights
    class_weights = calculate_class_weights(y_train, num_classes)
    print(f"\nCategory weight: {class_weights.tolist()}")

    # Create a dataset
    max_seq_len = 20
    train_dataset = KDD99Dataset(X_train, y_train, max_seq_len)
    test_dataset = KDD99Dataset(X_test, y_test, max_seq_len)

    print(f"\nCreating a pseudo-sequence dataset:")
    print(f"  Number of training sequences: {len(train_dataset)}")
    print(f"  Number of test sequences: {len(test_dataset)}")
    print(f"  Maximum sequence length: {max_seq_len}")

    # Create a data loader
    batch_size = 64
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=kdd_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=kdd_collate_fn
    )

    # Initialize the model
    num_features = X_train.shape[1]
    model = KDDInformer(
        num_features=num_features,
        num_classes=num_classes,
        d_model=128,
        n_heads=8,
        e_layers=3
    )

    print("\nModel Architecture:")
    print(model)

    id_to_label = {v: k for k, v in label_to_id.items()}
    class_names = [id_to_label[i] for i in range(num_classes)]

    # Train the model
    train_model(
        model,
        train_loader,
        test_loader,
        num_classes=num_classes,
        class_weights=class_weights,
        epochs=5,
        lr=0.0005
    )

    # Final evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the best model
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    # Reverse the label_to_id mapping to get the category names
    id_to_label = {v: k for k, v in label_to_id.items()}
    class_names = [id_to_label[i] for i in range(num_classes)]

    # Use the test set for final evaluation
    test_metrics, test_conf_matrix = evaluate_model(
        model,
        test_loader,
        device,
        num_classes=num_classes,
        class_names=class_names,
        plot_confusion=True
    )

    # Print test results
    print("\n===== Final test results =====")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test loss: {test_metrics['avg_loss']:.4f}")
    print(f"Macro Average Precision: {test_metrics['macro_precision']:.4f}")
    print(f"Macro Average Recall: {test_metrics['macro_recall']:.4f}")
    print(f"Macro average F1 score: {test_metrics['macro_f1']:.4f}")
    print(f"Weighted Average Precision: {test_metrics['weighted_precision']:.4f}")
    print(f"Weighted Average Recall: {test_metrics['weighted_recall']:.4f}")
    print(f"Weighted average F1 score: {test_metrics['weighted_f1']:.4f}")

    # Print the metrics for each category
    print("\n===== Test indicators by category =====")
    for metric in test_metrics['class_metrics']:
        print(f"category {metric['class_id']} ({metric['class_name']}):")
        print(f"  Accuracy: {metric['precision']:.4f}")
        print(f"  Recall: {metric['recall']:.4f}")
        print(f"  F1 score: {metric['f1']:.4f}")
        print(f"  Number of samples: {metric['support']}")

if __name__ == "__main__":
    main()