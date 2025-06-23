import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc
import os
import logging

class Evaluator:
    """Evaluation and Visualization Tool"""

    def __init__(self, save_plots=True, plot_dir='results/plots', max_plot_points=5000):
        """
        Initialize the evaluator
        Parameters:
        save_plots: Whether to save plots
        plot_dir: Directory to save plots
        max_plot_points: Maximum number of plot points
        """
        # Force non-interactive backend
        matplotlib.use('Agg')  # Key setting: Ensure images can be saved in any environment
        plt.switch_backend('Agg')
        self.save_plots = save_plots
        self.plot_dir = plot_dir
        self.max_plot_points = max_plot_points

        # Ensure directory exists
        os.makedirs(self.plot_dir, exist_ok=True)

        # Set up logging
        self.logger = logging.getLogger('Evaluator')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(ch)

    def find_optimal_threshold(self, scores, labels):
        """
        Find the threshold that maximizes the F1 score
        Parameters:
        scores: Array of anomaly scores
        labels: Array of true labels
        Returns:
        best_threshold: Best threshold
        best_f1: Best F1 score
        """
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-9)

        # Find the threshold corresponding to the maximum F1 score
        max_idx = np.argmax(f1_scores)
        best_threshold = thresholds[max_idx] if max_idx < len(thresholds) else 0.5
        best_f1 = f1_scores[max_idx]
        return best_threshold, best_f1

    def calculate_metrics(self, scores, labels, threshold=None):
        """
        Calculate evaluation metrics
        Parameters:
        scores: Array of anomaly scores
        labels: Array of true labels
        threshold: Threshold (if None, it will be calculated automatically)
        Returns:
        metrics: Dictionary containing various metrics
        """
        if threshold is None:
            threshold, _ = self.find_optimal_threshold(scores, labels)
        pred_labels = (scores > threshold).astype(int)
        return {
            'f1': f1_score(labels, pred_labels),
            'auc_roc': roc_auc_score(labels, scores),
            'threshold': threshold
        }

    def plot_results(self, test_data, scores, labels, threshold, item_id):
        """
        Visualize results (ensure reliable saving)
        Parameters:
        test_data: Raw test data (n_timesteps, n_features)
        scores: Array of anomaly scores (n_timesteps,)
        labels: Array of true labels (n_timesteps,)
        threshold: Detection threshold
        item_id: Item ID
        """
        try:
            self.logger.info(f"Starting plot for {item_id}")
            n_points = len(scores)

            # Smart downsampling
            if n_points > self.max_plot_points:
                step = max(1, n_points // self.max_plot_points)
                indices = np.arange(0, n_points, step)
                scores_sampled = scores[indices]
                labels_sampled = labels[indices]
                test_data_sampled = test_data[indices]
                self.logger.info(f"Downsampled {n_points} to {len(indices)} points")
            else:
                indices = np.arange(n_points)
                scores_sampled = scores
                labels_sampled = labels
                test_data_sampled = test_data
                self.logger.info(f"Using all {n_points} points")

            # Create the plot
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))

            # 1. Raw data and anomaly regions
            ax1.plot(indices, test_data_sampled[:, 0], 'b-', alpha=0.7, linewidth=0.5)

            # Optimize anomaly region drawing
            anomaly_indices = np.where(labels_sampled == 1)[0]
            if len(anomaly_indices) > 0:
                # Find continuous anomaly regions
                diff = np.diff(anomaly_indices)
                breaks = np.where(diff > 1)[0] + 1
                segments = np.split(anomaly_indices, breaks)
                for seg in segments:
                    if len(seg) > 0:
                        start_idx = seg[0]
                        end_idx = seg[-1]
                        ax1.axvspan(start_idx, end_idx, color='red', alpha=0.3)
            ax1.set_title(f'{item_id} - Time Series (Feature 0)')
            ax1.grid(True, linestyle='--', alpha=0.3)
            ax1.set_xlim(indices[0], indices[-1])

            # 2. Anomaly scores
            ax2.plot(indices, scores_sampled, 'g-', alpha=0.7, linewidth=0.5)
            ax2.axhline(y=threshold, color='r', linestyle='--')
            ax2.set_title('Anomaly Scores')
            ax2.grid(True, linestyle='--', alpha=0.3)
            ax2.set_xlim(indices[0], indices[-1])

            # 3. Prediction results
            pred_labels = (scores_sampled > threshold).astype(int)
            ax3.step(indices, pred_labels, 'r-', where='post', linewidth=0.5)
            ax3.step(indices, labels_sampled, 'b-', alpha=0.5, where='post', linewidth=0.5)
            ax3.set_yticks([0, 1])
            ax3.set_yticklabels(['Normal', 'Anomaly'])
            f1 = f1_score(labels_sampled, pred_labels)
            ax3.set_title(f'Detection Results (F1={f1:.4f})')
            ax3.grid(True, linestyle='--', alpha=0.3)
            ax3.set_xlim(indices[0], indices[-1])

            plt.tight_layout()
            if self.save_plots:
                plot_path = os.path.join(self.plot_dir, f'{item_id}_results.png')
                # Multiple attempts to save
                for attempt in range(3):
                    try:
                        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                        self.logger.info(f"Saved plot to {plot_path}")
                        break
                    except Exception as e:
                        self.logger.error(f"Save attempt {attempt + 1} failed: {str(e)}")
                        if attempt == 2:
                            self.logger.error("Failed to save plot after 3 attempts")
            plt.close(fig)
            self.logger.info("Plot completed and closed")
            return True
        except Exception as e:
            self.logger.error(f"Error during plotting: {str(e)}")
            return False
