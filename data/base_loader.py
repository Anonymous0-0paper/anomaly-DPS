import numpy as np
import torch
from torch.utils.data import Dataset

class AnomalyDetectionDatasets(Dataset):

    def __init__(self, features, labels, window_size=10, stride=5, num_classes=2):
        """
        Parameters:
        features: Feature array (n_samples, n_features)
        labels: Label array (n_samples,)
        window_size: Time window size
        stride: Sliding step size
        num_classes: Number of classes
        """
        self.features = features
        self.labels = labels
        self.window_size = window_size
        self.stride = stride
        self.num_classes = num_classes
        self.num_features = features.shape[1]

        # Create sample sequences
        self.samples = []
        for i in range(0, len(features) - window_size + 1, stride):
            self.samples.append(i)

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a single sample"""
        start = self.samples[idx]
        end = start + self.window_size

        # Feature sequence (window_size, num_features)
        seq_features = self.features[start:end]

        # Label sequence (window_size,)
        seq_labels = self.labels[start:end]

        # Determine the main label of the window
        if self.num_classes > 2:  # Multiclass
            label_counts = np.bincount(seq_labels, minlength=self.num_classes)
            label = np.argmax(label_counts)  # Select the most frequent class
        else:  # Binary classification
            label = 1 if np.sum(seq_labels) > self.window_size / 2 else 0

        return {
            'x': torch.tensor(seq_features, dtype=torch.float32),
            'y': torch.tensor(label, dtype=torch.long)
        }
