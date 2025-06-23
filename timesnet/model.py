from deepod.models.time_series import TimesNet as DeepOD_TimesNet
import numpy as np

class AnomalyDetectionPipeline:
    """Anomaly Detection Training and Prediction Pipeline"""

    def __init__(self, model_params, device='cuda'):
        self.model_params = model_params
        self.device = device
        # Separately extract inference batch size
        self.inference_batch_size = model_params.get('inference_batch_size', 10000)
        # Remove inference_batch_size from model parameters to avoid passing it to the model
        self.model_params = model_params.copy()
        self.model_params.pop('inference_batch_size', None)

    def initialize_model(self, n_features):
        return DeepOD_TimesNet(
            **self.model_params,
            device=self.device
        )

    def fit(self, train_data, test_data, test_labels, logger):
        # Initialize model
        self.model = self.initialize_model(train_data.shape[1])

        # Train model
        logger.info("Starting training...")
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
