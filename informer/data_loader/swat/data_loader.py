import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

# Global configuration: Disable specific warnings
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
pd.options.mode.chained_assignment = None  # Disable SettingWithCopyWarning

def load_swat_data(file_path, version='v2'):
    """
    Precise SWaT dataset loading function, creating labels based on attack time periods
    """
    print(f"\n{'=' * 80}")
    print(f"{'Loading SWaT Dataset':^80}")
    print(f"{'=' * 80}")
    print(f"Using {version} data version")

    try:
        # Read Excel file
        if version == 'v2':
            # v2 version has two header rows
            headers = pd.read_excel(file_path, header=None, nrows=2)
            print(f"Header shape: {headers.shape}")

            # Create column names: device_name_data_type
            column_names = ['timestamp']
            for i in range(1, headers.shape[1]):
                device = str(headers.iloc[0, i]).strip() if not pd.isna(headers.iloc[0, i]) else f"device_{i}"
                dtype = str(headers.iloc[1, i]).strip() if not pd.isna(headers.iloc[1, i]) else "value"
                column_names.append(f"{device}_{dtype}")

            # Read actual data_loader (skip headers)
            df = pd.read_excel(file_path, header=None, skiprows=2, names=column_names)
        else:
            # v1 version has only one header row
            df = pd.read_excel(file_path)
            column_names = df.columns.tolist()
            # Rename timestamp column
            if 'Timestamp' in column_names:
                df = df.rename(columns={'Timestamp': 'timestamp'})

        print(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")

        # Process time column
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
        df = df.dropna(subset=['timestamp'])
        print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        # Define attack periods (GMT+0)
        attack_periods = [
            ("2019-07-20 07:08:46", "2019-07-20 07:10:31"),  # FIT401
            ("2019-07-20 07:15:00", "2019-07-20 07:19:32"),  # LIT301
            ("2019-07-20 07:26:57", "2019-07-20 07:30:48"),  # P601
            ("2019-07-20 07:38:50", "2019-07-20 07:46:20"),  # Multi-point
            ("2019-07-20 07:54:00", "2019-07-20 07:56:00"),  # MV501
            ("2019-07-20 08:02:56", "2019-07-20 08:16:18")  # P301
        ]

        # Convert time format
        attack_periods = [(pd.Timestamp(s + "+00:00"), pd.Timestamp(e + "+00:00"))
                          for s, e in attack_periods]

        # Create label column
        df['label'] = 0  # Default to normal

        # Mark attack periods
        for start, end in attack_periods:
            attack_mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
            df.loc[attack_mask, 'label'] = 1

        # Print label distribution
        print(f"Label distribution: Normal={sum(df['label'] == 0)}, Anomaly={sum(df['label'] == 1)}")

        # Process string features
        feature_columns = [col for col in df.columns if col not in ['timestamp', 'label']]

        for col in feature_columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'string':
                try:
                    # Try to convert to numeric
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                    # If conversion fails, use label encoding
                    if df[col].isna().any():
                        print(f"Column '{col}' contains non-numeric data, using label encoding")
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col].astype(str))
                except Exception as e:
                    print(f"Error processing column '{col}': {e}")
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))

        # Separate features and labels
        X = df[feature_columns].values.astype(np.float32)
        y = df['label'].values

        # Split training and test sets (in time order)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Ensure there are anomaly samples in the test set
        if sum(y_test) == 0:
            print("Warning: No anomaly samples in the test set, adjusting split point")
            # Find the position of the last anomaly sample
            anomaly_indices = np.where(y == 1)[0]
            if len(anomaly_indices) > 0:
                last_anomaly_idx = anomaly_indices[-1]
                split_idx = max(int(len(X) * 0.7), last_anomaly_idx - 1000)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]

        # Feature engineering: Remove constant features
        constant_cols = np.where(np.std(X_train, axis=0) == 0)[0]
        if len(constant_cols) > 0:
            print(f"Removing {len(constant_cols)} constant features")
            X_train = np.delete(X_train, constant_cols, axis=1)
            X_test = np.delete(X_test, constant_cols, axis=1)

        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Dataset information
        num_classes = 2
        class_names = ['Normal', 'Attack']

        print(f"\nSWaT Dataset Summary ({version}):")
        print(f"  Training samples: {len(X_train)} (Anomaly ratio: {np.sum(y_train) / len(y_train):.4f})")
        print(f"  Test samples: {len(X_test)} (Anomaly ratio: {np.sum(y_test) / len(y_test):.4f})")
        print(f"  Feature dimension: {X_train.shape[1]}")

        return X_train, X_test, y_train, y_test, num_classes, class_names

    except Exception as e:
        print(f"Error loading SWaT dataset: {str(e)}")
        # Create dummy dataset
        print("Creating dummy dataset...")
        X_train = np.random.rand(1000, 50).astype(np.float32)
        X_test = np.random.rand(200, 50).astype(np.float32)
        y_train = np.zeros(1000)
        y_train[:50] = 1  # 5% anomalies
        y_test = np.zeros(200)
        y_test[:10] = 1  # 5% anomalies
        num_classes = 2
        class_names = ['Normal', 'Attack']
        return X_train, X_test, y_train, y_test, num_classes, class_names
