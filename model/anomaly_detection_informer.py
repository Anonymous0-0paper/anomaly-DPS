import torch
import torch.nn as nn

class AnomalyDetectionInformer(nn.Module):
    """Network security anomaly detection model based on Informer architecture"""

    def __init__(self, num_features, num_classes, d_model=128, n_heads=8, e_layers=3, dropout=0.1):
        """
        Parameters:
        num_features: Input feature dimension
        num_classes: Number of output classes
        d_model: Model dimension
        n_heads: Number of attention heads
        e_layers: Number of encoder layers
        dropout: Dropout probability
        """
        super().__init__()

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

        # Attention pooling layer
        self.attention_pool = nn.MultiheadAttention(
            d_model, 1, dropout=dropout, batch_first=True
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        """Forward propagation"""
        batch_size, seq_len, num_features = x.shape

        # Input embedding
        x_emb = self.input_embedding(x)

        # Positional embedding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)

        # Combined embedding
        combined = x_emb + pos_emb

        # Through encoder layers
        encoder_output = combined
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output)

        # Attention pooling
        query = encoder_output.mean(dim=1, keepdim=True)  # (batch_size, 1, d_model)
        pooled, _ = self.attention_pool(query, encoder_output, encoder_output)
        pooled = pooled.squeeze(1)  # (batch_size, d_model)

        # Classification prediction
        logits = self.classifier(pooled)
        return logits
