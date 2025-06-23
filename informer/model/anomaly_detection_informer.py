import torch
import torch.nn as nn
import torch.nn.functional as F

class AnomalyDetectionInformer(nn.Module):
    """Unified supervised/unsupervised anomaly detection model"""

    def __init__(self, num_features, num_classes=2, d_model=128, n_heads=8, e_layers=3, dropout=0.1):
        """
        Parameters:
        num_features: Input feature dimension
        num_classes: Number of output classes (used in supervised mode)
        d_model: Model dimension
        n_heads: Number of attention heads
        e_layers: Number of encoder layers
        dropout: Dropout probability
        """
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.d_model = d_model

        # Input embedding layer
        self.input_embedding = nn.Linear(num_features, d_model)

        # Positional encoding
        self.position_embedding = nn.Embedding(1000, d_model)

        # Informer encoder
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(e_layers)
        ])

        # Classification head (supervised mode)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

        # Reconstruction head (unsupervised mode)
        self.reconstructor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_features)
        )

        # Attention pooling
        self.attention_pool = nn.MultiheadAttention(
            d_model, 1, dropout=dropout, batch_first=True
        )

    def forward(self, x, training_mode='supervised'):
        """Forward propagation"""
        batch_size, seq_len, num_features = x.shape

        # Input embedding
        x_emb = self.input_embedding(x)

        # Positional embedding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)

        # Combined embedding
        combined = x_emb + pos_emb

        # Encoder processing
        encoder_output = combined
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output)

        # Attention pooling
        query = encoder_output.mean(dim=1, keepdim=True)
        pooled, _ = self.attention_pool(query, encoder_output, encoder_output)
        pooled = pooled.squeeze(1)

        # Return different outputs based on training mode
        if training_mode == 'unsupervised':
            # Unsupervised mode: reconstruct sequence
            reconstructed = self.reconstructor(pooled)
            reconstructed = reconstructed.unsqueeze(1).repeat(1, seq_len, 1)
            return None, reconstructed  # Classification output is None
        else:
            # Supervised mode: classification output
            logits = self.classifier(pooled)
            return logits, None  # Reconstruction output is None

    def predict_anomaly(self, x):
        """Predict anomaly (for unsupervised mode only)"""
        with torch.no_grad():
            _, reconstructed = self.forward(x, training_mode='unsupervised')
            # Calculate reconstruction error
            reconstruction_error = F.mse_loss(reconstructed, x, reduction='none')
            # Average error per sample (batch_size,)
            sample_errors = reconstruction_error.mean(dim=(1, 2))
            return sample_errors
