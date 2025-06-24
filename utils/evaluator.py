# main.py (优化版)
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, precision_score, recall_score
import os
import logging
import gc


class Evaluator:
    """Evaluation and Visualization Tool"""

    def __init__(self, model_name, save_plots=True,
                 plot_dir='results/plots', max_plot_points=5000):
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
        self.model_name = model_name
        self.save_plots = save_plots
        self.plot_dir = os.path.join(plot_dir, model_name)
        self.max_plot_points = max_plot_points

        # Ensure directory exists
        os.makedirs(self.plot_dir, exist_ok=True)

        # Set up logging
        self.logger = logging.getLogger(f'Evaluator.{model_name}')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
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

        max_idx = np.argmax(f1_scores)
        best_threshold = thresholds[max_idx] if max_idx < len(thresholds) else np.median(scores)
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
            'precision': precision_score(labels, pred_labels),
            'recall': recall_score(labels, pred_labels),
            'auc_roc': roc_auc_score(labels, scores),
            'threshold': threshold,
            'model': self.model_name
        }

    def plot_results(self, test_data, scores, labels, threshold, item_id,
                     additional_visualizations=None):
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
            fig = plt.figure(figsize=(15, 12))
            gs = fig.add_gridspec(4, 1)

            # 1. Raw data and anomaly regions
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(indices, test_data_sampled[:, 0], 'b-', alpha=0.7, linewidth=0.5)

            # Optimize anomaly region drawing
            anomaly_indices = np.where(labels_sampled == 1)[0]
            if len(anomaly_indices) > 0:
                diff = np.diff(anomaly_indices)
                breaks = np.where(diff > 1)[0] + 1
                segments = np.split(anomaly_indices, breaks)
                for seg in segments:
                    if len(seg) > 0:
                        start_idx = seg[0]
                        end_idx = seg[-1]
                        ax1.axvspan(start_idx, end_idx, color='red', alpha=0.3)
            ax1.set_title(f'{self.model_name} - {item_id} (Feature 0)')
            ax1.grid(True, linestyle='--', alpha=0.3)
            ax1.set_xlim(indices[0], indices[-1])

            # 2. Anomaly scores
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
            ax2.plot(indices, scores_sampled, 'g-', alpha=0.7, linewidth=0.5)
            ax2.axhline(y=threshold, color='r', linestyle='--')
            ax2.set_title('Anomaly scores')
            ax2.grid(True, linestyle='--', alpha=0.3)

            # 3. Prediction results
            ax3 = fig.add_subplot(gs[2], sharex=ax1)
            pred_labels = (scores_sampled > threshold).astype(int)
            ax3.step(indices, pred_labels, 'r-', where='post', linewidth=0.5)
            ax3.step(indices, labels_sampled, 'b-', alpha=0.5, where='post', linewidth=0.5)
            ax3.set_yticks([0, 1])
            ax3.set_yticklabels(['Normal', 'Anomaly'])
            f1 = f1_score(labels_sampled, pred_labels)
            ax3.set_title(f'Detection Results (F1={f1:.4f})')
            ax3.grid(True, linestyle='--', alpha=0.3)

            if additional_visualizations:
                ax4 = fig.add_subplot(gs[3], sharex=ax1)
                try:
                    additional_visualizations(ax4, indices, test_data_sampled, scores_sampled)
                    ax4.grid(True, linestyle='--', alpha=0.3)
                except Exception as e:
                    self.logger.error(f"模型特定可视化失败: {str(e)}")
                    plt.close(fig)
                    return False

            plt.tight_layout()

            if self.save_plots:
                plot_path = os.path.join(self.plot_dir, f'{item_id}_results.png')
                for attempt in range(3):
                    try:
                        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                        self.logger.info(f"Saved plot to {plot_path}")
                        break
                    except Exception as e:
                        self.logger.error(f"保存尝试 {attempt + 1}/3 失败: {str(e)}")

            plt.close(fig)
            gc.collect()
            return True

        except Exception as e:
            self.logger.error(f"Error during plotting: {str(e)}")
            return False