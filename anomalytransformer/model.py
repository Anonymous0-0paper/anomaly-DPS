from deepod.models.time_series import AnomalyTransformer as DeepOD_AnomalyTransformer
import numpy as np


class AnomalyDetectionPipeline:
    """Anomaly Detection Training and Prediction Pipeline for AnomalyTransformer"""

    def __init__(self, model_params, device='cuda'):
        self.model_params = model_params
        self.device = device
        # Extract inference batch size
        self.inference_batch_size = model_params.get('inference_batch_size', 10000)
        # Remove inference_batch_size from model parameters
        self.model_params = model_params.copy()
        self.model_params.pop('inference_batch_size', None)

        # AnomalyTransformer specific parameters
        self.k = model_params.get('k', 3)
        self.e_layers = model_params.get('e_layers', 3)  # 默认使用3层

    def initialize_model(self, n_features):
        """Initialize AnomalyTransformer model with proper parameters"""
        # 修复：使用正确的参数名和结构
        return DeepOD_AnomalyTransformer(
            seq_len=self.model_params.get('seq_len', 100),
            stride=self.model_params.get('stride', 1),
            lr=self.model_params.get('lr', 1e-4),
            epochs=self.model_params.get('epochs', 1),
            batch_size=self.model_params.get('batch_size', 32),
            device=self.device,
            k=self.k
        )

    def fit(self, train_data, test_data, test_labels, logger):
        # Initialize model
        self.model = self.initialize_model(train_data.shape[1])

        # 设置特征维度 (deepod库要求)
        self.model.n_features = train_data.shape[1]

        # Train model
        logger.info(f"Starting training for AnomalyTransformer with {train_data.shape[1]} features...")
        self.model.fit(train_data)

        # Process test data in batches
        logger.info("Generating anomaly scores...")
        n_test = len(test_data)
        scores = np.zeros(n_test)

        # Calculate anomaly scores in batches
        for i in range(0, n_test, self.inference_batch_size):
            end_idx = min(i + self.inference_batch_size, n_test)
            batch = test_data[i:end_idx]
            batch_scores = self.model.decision_function(batch)
            scores[i:end_idx] = batch_scores
            logger.info(f"Processed {end_idx}/{n_test} points")

        # Ensure labels are aligned
        aligned_labels = test_labels[:n_test]
        return scores, aligned_labels
