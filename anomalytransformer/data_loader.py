import os
import ast
import numpy as np
import pandas as pd
import glob
import logging
from datetime import datetime

class BaseDataLoader:
    """Base Data Loader"""
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.logger = logging.getLogger('DataLoader')

    def get_all_items(self):
        """Get all items (channels or machines)"""
        raise NotImplementedError

    def load_item(self, item_id):
        """Load data for the specified item"""
        raise NotImplementedError

class NASADataLoader(BaseDataLoader):
    """NASA Dataset Loader (supports MSL and SMAP)"""
    def __init__(self, root_dir, dataset='MSL'):
        super().__init__(root_dir)
        self.dataset = dataset
        self.label_file = os.path.join(root_dir, 'labeled_anomalies.csv')
        self.train_dir = os.path.join(root_dir, 'train')
        self.test_dir = os.path.join(root_dir, 'test')
        # Read the annotation file and filter the dataset
        self.anomaly_df = pd.read_csv(self.label_file)
        self.channels = self.anomaly_df[self.anomaly_df['spacecraft'] == dataset]['chan_id'].values
        self.logger.info(f"Loaded NASA {dataset} dataset with {len(self.channels)} channels")

    def get_all_items(self):
        """Get all channel IDs"""
        return self.channels

    def load_item(self, item_id):
        try:
            # Load training and test data
            train_file = os.path.join(self.train_dir, f"{item_id}.npy")
            test_file = os.path.join(self.test_dir, f"{item_id}.npy")
            # Load as 2D array (timesteps, channels)
            train_data_all = np.load(train_file)
            test_data_all = np.load(test_file)
            # Key modification: Select the first channel (NASA dataset has one channel per file)
            train_data = train_data_all[:, 0:1]  # Keep 2D structure (n,1)
            test_data = test_data_all[:, 0:1]  # Keep 2D structure (n,1)
            # Get annotation information
            channel_info = self.anomaly_df[self.anomaly_df['chan_id'] == item_id].iloc[0]
            anomaly_sequences = ast.literal_eval(channel_info['anomaly_sequences'])
            # Create test labels
            test_labels = np.zeros(len(test_data), dtype=int)
            for start, end in anomaly_sequences:
                start = max(0, min(start, len(test_data) - 1))
                end = max(0, min(end, len(test_data) - 1))
                test_labels[start:end + 1] = 1
            self.logger.info(
                f"Loaded NASA {item_id}: "
                f"Train={train_data.shape}, Test={test_data.shape}, "
                f"Anomalies={test_labels.sum()}"
            )
            return train_data, test_data, test_labels
        except Exception as e:
            self.logger.error(f"Error loading NASA data for {item_id}: {str(e)}")
            raise

class SMDDataLoader(BaseDataLoader):
    """SMD Dataset Loader"""
    def __init__(self, root_dir):
        super().__init__(root_dir)
        self.train_dir = os.path.join(root_dir, 'train')
        self.test_dir = os.path.join(root_dir, 'test')
        self.test_label_dir = os.path.join(root_dir, 'test_label')
        # Get all machine files
        self.train_files = sorted(glob.glob(os.path.join(self.train_dir, 'machine-*.txt')))
        self.machine_ids = [os.path.basename(f).replace('.txt', '') for f in self.train_files]
        self.logger.info(f"Loaded SMD dataset with {len(self.machine_ids)} machines")

    def get_all_items(self):
        """Get all machine IDs"""
        return self.machine_ids

    def load_item(self, machine_id):
        """
        Load data for the specified machine
        Returns:
        train_data: Training data (n_timesteps, n_features)
        test_data: Test data (n_timesteps, n_features)
        test_labels: Test labels (n_timesteps,)
        """
        try:
            # Load training data
            train_file = os.path.join(self.train_dir, f"{machine_id}.txt")
            self.logger.info(f"Loading SMD train data: {train_file}")
            train_data = np.genfromtxt(train_file, delimiter=',', dtype=np.float32)
            # Load test data
            test_file = os.path.join(self.test_dir, f"{machine_id}.txt")
            self.logger.info(f"Loading SMD test data: {test_file}")
            test_data = np.genfromtxt(test_file, delimiter=',', dtype=np.float32)
            # Load test labels
            label_file = os.path.join(self.test_label_dir, f"{machine_id}.txt")
            self.logger.info(f"Loading SMD labels: {label_file}")
            test_labels = np.genfromtxt(label_file, delimiter=',', dtype=np.int32)
            # Validate data shapes
            if len(train_data.shape) == 1:
                train_data = train_data.reshape(-1, 1)
            if len(test_data.shape) == 1:
                test_data = test_data.reshape(-1, 1)
            if len(test_labels.shape) > 1:
                test_labels = test_labels.squeeze()
            # Check data validity
            if np.isnan(train_data).any() or np.isnan(test_data).any():
                self.logger.warning(f"NaN values detected in {machine_id} data")
                train_data = np.nan_to_num(train_data)
                test_data = np.nan_to_num(test_data)
            self.logger.info(
                f"Loaded SMD {machine_id}: Train={train_data.shape}, Test={test_data.shape}, Anomalies={test_labels.sum()}")
            return train_data, test_data, test_labels
        except Exception as e:
            self.logger.error(f"Error loading SMD data for {machine_id}: {str(e)}")
            raise

class SWATDataLoader(BaseDataLoader):
    """SWAT Dataset Loader (supports new CSV format)"""
    def __init__(self, root_dir, config):
        super().__init__(root_dir)
        self.config = config
        self.train_path = os.path.join(root_dir, config['train_file'])
        self.test_path = os.path.join(root_dir, config['test_file'])
        self.logger = logging.getLogger('SWATLoader')
        self.system_id = 'swat_system'
        self.logger.info(f"Initializing SWAT loader for files: {self.train_path}, {self.test_path}")

    def get_all_items(self):
        return [self.system_id]

    def load_item(self, item_id):
        try:
            # 1. Load training data (normal operation)
            self.logger.info(f"Loading SWAT train data: {self.train_path}")
            train_df = pd.read_csv(self.train_path)
            # 2. Load test data (contains attacks)
            self.logger.info(f"Loading SWAT test data: {self.test_path}")
            test_df = pd.read_csv(self.test_path)
            # 3. Determine label column
            label_col = self._find_label_column(test_df.columns)
            self.logger.info(f"Using label column: {label_col}")
            # 4. Determine feature columns - exclude timestamp column
            if self.config['feature_columns']:
                feature_cols = self.config['feature_columns']
            else:
                # Use all columns except label column and timestamp
                feature_cols = [col for col in test_df.columns
                                if col != label_col and col.lower() != 'timestamp']
            # 5. Verify feature columns are all numeric
            numeric_cols = []
            for col in feature_cols:
                if pd.api.types.is_numeric_dtype(test_df[col]):
                    numeric_cols.append(col)
                else:
                    self.logger.warning(f"Ignoring non-numeric column: {col}")
            if len(numeric_cols) == 0:
                raise ValueError("No valid numeric features found")
            self.logger.info(f"Using {len(numeric_cols)} feature columns: {numeric_cols[:5]}...")
            # 6. Extract time features
            timestamp_col = self._find_timestamp_column(train_df.columns)
            if timestamp_col:
                self.logger.info(f"Found timestamp column: {timestamp_col}")
                train_df = self._add_time_features(train_df, timestamp_col)
                test_df = self._add_time_features(test_df, timestamp_col)
                # Add time features to numeric columns if they are numeric
                time_features = ['hour', 'dayofweek', 'dayofmonth']
                for feat in time_features:
                    if feat in train_df.columns and feat not in numeric_cols:
                        numeric_cols.append(feat)
                        self.logger.info(f"Added time feature: {feat}")
            # 7. Prepare training data (normal operation, no attacks)
            train_data = train_df[numeric_cols].values.astype(np.float32)
            # 8. Prepare test data and labels
            test_data = test_df[numeric_cols].values.astype(np.float32)
            test_labels = self._create_labels(test_df[label_col])
            # 9. Validate and process data
            train_data, test_data, test_labels = self._validate_and_process(
                train_data, test_data, test_labels
            )
            self.logger.info(f"SWAT data loaded: Train={train_data.shape}, Test={test_data.shape}, "
                             f"Anomalies={test_labels.sum()} ({test_labels.sum() / len(test_labels):.2%})")
            return train_data, test_data, test_labels
        except Exception as e:
            self.logger.error(f"Error loading SWAT data: {str(e)}")
            raise

    def _find_label_column(self, columns):
        """Determine the label column name (for test data)"""
        # Prefer the specified column in the config
        if self.config.get('label_column') and self.config['label_column'] in columns:
            return self.config['label_column']
        common_labels = ['Normal/Attack', 'Label', 'Status', 'Class', 'Anomaly']
        for col in common_labels:
            if col in columns:
                return col
        label_keywords = ['attack', 'normal', 'label', 'class', 'status']
        for col in columns:
            if any(kw in col.lower() for kw in label_keywords):
                return col
        # Use the last column as the label
        self.logger.warning("Using last column as label")
        return columns[-1]

    def _find_timestamp_column(self, columns):
        """Find the timestamp column"""
        for col in columns:
            if 'timestamp' in col.lower() or 'date' in col.lower() or 'time' in col.lower():
                return col
        return None

    def _add_time_features(self, df, timestamp_col):
        """Add time-based features from timestamp column"""
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df['hour'] = df[timestamp_col].dt.hour
        df['dayofweek'] = df[timestamp_col].dt.dayofweek
        df['dayofmonth'] = df[timestamp_col].dt.day
        # Normalize time features
        df['hour'] = df['hour'] / 24.0
        df['dayofweek'] = df['dayofweek'] / 7.0
        df['dayofmonth'] = df['dayofmonth'] / 31.0
        # Drop original timestamp column
        df.drop(columns=[timestamp_col], inplace=True)
        return df

    def _create_labels(self, label_series):
        """Create binary labels from the label column"""
        # Process string labels
        labels = np.zeros(len(label_series), dtype=int)
        # Create label mapping
        normal_vals = ['normal', '0', 'false']
        attack_vals = ['attack', '1', 'true']
        # Apply mapping
        for i, val in enumerate(label_series):
            val_str = str(val).strip().lower()
            if val_str in attack_vals:
                labels[i] = 1
            elif val_str in normal_vals:
                labels[i] = 0
            else:
                # Treat unknown values as normal by default
                labels[i] = 0
                self.logger.warning(f"Unknown label value: {val}, treating as normal")
        return labels

    def _validate_and_process(self, train_data, test_data, test_labels):
        """Validate and process data"""
        # Handle NaN values
        if np.isnan(train_data).any():
            self.logger.warning("NaN values in training data. Replacing with 0.")
            train_data = np.nan_to_num(train_data)
        if np.isnan(test_data).any():
            self.logger.warning("NaN values in test data. Replacing with 0.")
            test_data = np.nan_to_num(test_data)
        # Ensure 2D arrays
        if len(train_data.shape) == 1:
            train_data = train_data.reshape(-1, 1)
        if len(test_data.shape) == 1:
            test_data = test_data.reshape(-1, 1)
        if len(test_labels.shape) > 1:
            test_labels = test_labels.squeeze()
        return train_data, test_data, test_labels

class WADIDataLoader(BaseDataLoader):
    """WADI Dataset Loader"""
    def __init__(self, root_dir, config):
        super().__init__(root_dir)
        self.config = config
        self.train_path = os.path.join(root_dir, config['train_file'])
        self.test_path = os.path.join(root_dir, config['test_file'])
        self.label_col = config.get('label_column', 'attack_label')
        self.logger = logging.getLogger('WADILoader')
        self.system_id = 'wadi_system'
        self.logger.info(f"Initializing WADI loader for files: {self.train_path}, {self.test_path}")

    def get_all_items(self):
        return [self.system_id]

    def load_item(self, item_id):
        try:
            # 1. Load training data (normal operation)
            self.logger.info(f"Loading WADI train data: {self.train_path}")
            train_df = pd.read_csv(self.train_path)
            # 2. Load test data (contains attacks)
            self.logger.info(f"Loading WADI test data: {self.test_path}")
            test_df = pd.read_csv(self.test_path)
            # 3. Identify and process timestamp column
            timestamp_col = self._find_timestamp_column(train_df.columns)
            if timestamp_col:
                self.logger.info(f"Found timestamp column: {timestamp_col}")
                train_df = self._add_time_features(train_df, timestamp_col)
                test_df = self._add_time_features(test_df, timestamp_col)
            # 4. Check if label column exists
            label_exists = self.label_col in test_df.columns
            # 5. Prepare training data (normal operation, no attacks)
            # Training data has no label column, so no need to drop
            if label_exists:
                train_data = train_df.drop(columns=[self.label_col], errors='ignore').values.astype(np.float32)
            else:
                train_data = train_df.values.astype(np.float32)
            # 6. Prepare test data and labels
            if label_exists:
                test_data = test_df.drop(columns=[self.label_col]).values.astype(np.float32)
                test_labels = test_df[self.label_col].values.astype(int)
            else:
                # If label column doesn't exist, use the last column as label
                self.logger.warning(f"Label column '{self.label_col}' not found, using last column as label")
                test_data = test_df.iloc[:, :-1].values.astype(np.float32)
                test_labels = test_df.iloc[:, -1].values.astype(int)
            # 7. Validate data shapes
            if len(train_data.shape) == 1:
                train_data = train_data.reshape(-1, 1)
            if len(test_data.shape) == 1:
                test_data = test_data.reshape(-1, 1)
            if len(test_labels.shape) > 1:
                test_labels = test_labels.squeeze()
            # 8. Check data validity
            if np.isnan(train_data).any():
                self.logger.warning("NaN values in training data. Replacing with 0.")
                train_data = np.nan_to_num(train_data)
            if np.isnan(test_data).any():
                self.logger.warning("NaN values in test data. Replacing with 0.")
                test_data = np.nan_to_num(test_data)
            self.logger.info(f"WADI data loaded: Train={train_data.shape}, Test={test_data.shape}, "
                             f"Anomalies={test_labels.sum()} ({test_labels.sum() / len(test_labels):.2%})")
            return train_data, test_data, test_labels
        except Exception as e:
            self.logger.error(f"Error loading WADI data: {str(e)}")
            raise

    def _find_timestamp_column(self, columns):
        """Find the timestamp column"""
        for col in columns:
            if 'timestamp' in col.lower() or 'date' in col.lower() or 'time' in col.lower():
                return col
        return None

    def _add_time_features(self, df, timestamp_col):
        """Add time-based features from timestamp column"""
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df['hour'] = df[timestamp_col].dt.hour
        df['dayofweek'] = df[timestamp_col].dt.dayofweek
        df['dayofmonth'] = df[timestamp_col].dt.day
        # Normalize time features
        df['hour'] = df['hour'] / 24.0
        df['dayofweek'] = df['dayofweek'] / 7.0
        df['dayofmonth'] = df['dayofmonth'] / 31.0
        # Drop original timestamp column
        df.drop(columns=[timestamp_col], inplace=True)
        return df

def get_data_loader(dataset_type, root_dir, config=None, **kwargs):
    """Get data loader factory function"""
    logger = logging.getLogger('DataLoaderFactory')
    try:
        if dataset_type == 'nasa':
            # NASA dataset requires additional dataset parameter
            dataset_name = kwargs.get('dataset', 'MSL')
            logger.info(f"Creating NASA loader for {dataset_name}")
            return NASADataLoader(root_dir, dataset=dataset_name)
        elif dataset_type == 'smd':
            logger.info("Creating SMD loader")
            return SMDDataLoader(root_dir)
        elif dataset_type == 'swat':
            if not config:
                raise ValueError("SWAT data loader requires config")
            logger.info("Creating SWAT loader")
            return SWATDataLoader(root_dir, config)
        elif dataset_type == 'wadi':
            if not config:
                raise ValueError("WADI data loader requires config")
            logger.info("Creating WADI loader")
            return WADIDataLoader(root_dir, config)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
    except Exception as e:
        logger.error(f"Error creating data loader: {str(e)}")
        raise
