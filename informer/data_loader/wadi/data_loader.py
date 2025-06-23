import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import re
import traceback

def load_wadi_data(base_path, version='A2'):
    """
    Load WADI.A2_19 Nov 2019 dataset
    :param base_path: Root directory path of the dataset
    :param version: Dataset version ('A2')
    :return: X_train, X_test, y_train, y_test, num_classes, class_names
    """
    print(f"\n{'=' * 80}")
    print(f"{'Loading WADI Dataset':^80}")
    print(f"{'=' * 80}")

    try:
        # 1. Define file paths
        train_file = os.path.join(base_path, 'WADI_14days_new.csv')
        test_file = os.path.join(base_path, 'WADI_attackdataLABLE.csv')

        print(f"  Training data_loader: {train_file}")
        print(f"  Test data_loader: {test_file}")

        # 2. Load training data_loader
        print("\nLoading training data_loader...")
        try:
            train_df = pd.read_csv(train_file, skiprows=4, low_memory=False)
            print("  Training data_loader loaded with skiprows=4")
        except Exception as e:
            print(f"  Warning: {str(e)}")
            print("  Trying without skipping rows...")
            train_df = pd.read_csv(train_file, low_memory=False)
            print("  Training data_loader loaded without skipping rows")

        # 3. Load test data_loader
        print("\nLoading test data_loader...")
        try:
            test_df = pd.read_csv(test_file, skiprows=1, low_memory=False)
            print("  Test data_loader loaded with skiprows=1")
        except Exception as e:
            print(f"  Warning: {str(e)}")
            print("  Trying without skipping rows...")
            test_df = pd.read_csv(test_file, low_memory=False)
            print("  Test data_loader loaded without skipping rows")

        # 4. Print initial data_loader shapes
        print(f"\nInitial data_loader shapes:")
        print(f"  Training data_loader: {train_df.shape}")
        print(f"  Test data_loader: {test_df.shape}")

        # 5. Data preprocessing function
        def preprocess_df(df, df_name):
            """Generic data_loader preprocessing function"""
            print(f"\nPreprocessing {df_name} data_loader...")
            print(f"  Initial shape: {df.shape}")

            # Drop all-empty columns
            df = df.dropna(axis=1, how='all')
            print(f"  After dropping all-NA columns: {df.shape}")

            # Drop time columns
            time_cols = ['Row', 'row', 'Date', 'date', 'Time', 'time']
            drop_cols = [col for col in time_cols if col in df.columns]

            if drop_cols:
                print(f"  Dropping time columns: {drop_cols}")
                df = df.drop(drop_cols, axis=1)
                print(f"  After dropping time columns: {df.shape}")

            # Handle special characters in numerical columns
            print("  Cleaning numerical columns...")
            for col in df.select_dtypes(include=['object']).columns:
                # Keep numbers, decimal points, and negative signs
                df[col] = df[col].replace(r'[^\d\.\-]', '', regex=True)
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass

            # Fill missing values
            print("  Filling missing values...")
            df = df.fillna(method='ffill').fillna(method='bfill')
            df = df.fillna(0)

            print(f"  Final shape: {df.shape}")
            return df

        # Preprocess training and test data_loader
        train_df = preprocess_df(train_df, "training")
        test_df = preprocess_df(test_df, "test")

        # 6. Extract test labels
        print("\nExtracting labels from test data_loader...")
        label_col = 'Attack LABLE (1:No Attack, -1:Attack)'

        if label_col not in test_df.columns:
            # Try to find the label column
            label_candidates = [col for col in test_df.columns
                                if 'LABLE' in col or 'LABEL' in col or 'Attack' in col]

            if label_candidates:
                label_col = label_candidates[0]
                print(f"  Using label column: {label_col}")
            else:
                print("  Warning: Label column not found, creating dummy labels")
                label_col = None
                y_test = np.zeros(len(test_df))

        if label_col:
            # Extract labels and drop the label column
            y_test = test_df[label_col].values
            test_df = test_df.drop(label_col, axis=1)
            print(f"  Test labels extracted. Test data_loader shape after drop: {test_df.shape}")

        # 7. Align feature columns - solve column name mismatch issues
        print("\nAligning feature columns...")

        # Print column name differences
        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)

        print(f"  Training columns count: {len(train_cols)}")
        print(f"  Test columns count: {len(test_cols)}")
        print(f"  Common columns count: {len(train_cols & test_cols)}")

        # Find different columns
        unique_train = train_cols - test_cols
        unique_test = test_cols - train_cols

        print(f"\n  Columns only in training data_loader ({len(unique_train)}):")
        print("    " + "\n    ".join(sorted(unique_train)[:5]) + ("..." if len(unique_train) > 5 else ""))

        print(f"\n  Columns only in test data_loader ({len(unique_test)}):")
        print("    " + "\n    ".join(sorted(unique_test)[:5]) + ("..." if len(unique_test) > 5 else ""))

        # Method 1: Try case-insensitive matching
        print("\nAttempting case-insensitive column matching...")
        train_cols_lower = [c.lower().strip() for c in train_df.columns]
        test_cols_lower = [c.lower().strip() for c in test_df.columns]

        # Create mapping dictionaries
        train_col_map = {c.lower().strip(): c for c in train_df.columns}
        test_col_map = {c.lower().strip(): c for c in test_df.columns}

        # Find common columns (case-insensitive)
        common_cols_lower = set(train_cols_lower) & set(test_cols_lower)

        if common_cols_lower:
            print(f"  Found {len(common_cols_lower)} common columns (case-insensitive)")

            # Create mapped column name lists
            mapped_train_cols = [train_col_map[col] for col in common_cols_lower]
            mapped_test_cols = [test_col_map[col] for col in common_cols_lower]

            # Ensure column order is consistent
            common_cols = sorted(mapped_train_cols)

            train_df = train_df[common_cols]
            test_df = test_df[mapped_test_cols]

            # Rename test set columns to match training set
            test_df.columns = common_cols
        else:
            # Method 2: Try partial matching (based on sensor ID)
            print("\nAttempting partial column matching...")

            def extract_sensor_id(col_name):
                """Extract sensor ID as matching basis"""
                # Try to match numeric prefix and sensor type
                match = re.search(r'(\d+_)?([A-Z]+\d*_?\d*)', col_name)
                if match:
                    return match.group(2)  # Return sensor ID part
                return col_name

            # Create sensor ID mappings
            train_sensor_ids = {extract_sensor_id(col): col for col in train_df.columns}
            test_sensor_ids = {extract_sensor_id(col): col for col in test_df.columns}

            common_ids = set(train_sensor_ids.keys()) & set(test_sensor_ids.keys())

            if common_ids:
                print(f"  Found {len(common_ids)} common sensor IDs")

                # Create mapped column name lists
                mapped_train_cols = [train_sensor_ids[id] for id in common_ids]
                mapped_test_cols = [test_sensor_ids[id] for id in common_ids]

                # Ensure column order is consistent
                common_cols = sorted(mapped_train_cols)

                train_df = train_df[common_cols]
                test_df = test_df[mapped_test_cols]

                # Rename test set columns to match training set
                test_df.columns = common_cols
            else:
                # Method 3: Use position-based matching
                print("\nAttempting position-based matching...")
                num_common = min(len(train_df.columns), len(test_df.columns))

                if num_common > 0:
                    print(f"  Matching first {num_common} columns by position")

                    # Get common columns
                    common_cols = train_df.columns[:num_common].tolist()

                    train_df = train_df.iloc[:, :num_common]
                    test_df = test_df.iloc[:, :num_common]

                    # Rename test set columns to match training set
                    test_df.columns = common_cols
                else:
                    raise ValueError("  Error: No common columns found after all matching attempts")

        print(f"\n  Final common columns: {len(common_cols)}")
        print("  Sample columns: " + ", ".join(common_cols[:3]) + "...")

        # 8. Feature engineering
        print("\nPerforming feature engineering...")

        def extract_sensor_type(col_name):
            """Extract sensor type from column name"""
            # Try to match common patterns
            patterns = [
                r'\d_([A-Z]{2,4})_\d+',  # Format: 1_AIT_001_PV
                r'([A-Z]{2,4})\d+_\w+',  # Format: FIT001_PV
                r'([A-Z]{2,4})_\d+',  # Format: AIT_001
            ]

            for pattern in patterns:
                match = re.search(pattern, col_name)
                if match:
                    sensor_type = match.group(1)
                    return sensor_type

            # Alternative: Extract all uppercase letter combinations
            matches = re.findall(r'[A-Z]{2,4}', col_name)
            if matches:
                return matches[0]

            return "Unknown"

        # Create sensor type features
        sensor_types = [extract_sensor_type(col) for col in common_cols]
        unique_types = sorted(set(sensor_types))
        print(f"  Detected sensor types: {unique_types}")

        sensor_type_mapping = {stype: idx for idx, stype in enumerate(unique_types)}
        sensor_type_features = [sensor_type_mapping[stype] for stype in sensor_types]

        # 9. Data standardization
        print("\nScaling data_loader with RobustScaler...")
        # Ensure correct data_loader types
        train_df = train_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        test_df = test_df.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Check if there are valid features
        if len(common_cols) == 0:
            raise ValueError("  Error: No features available after preprocessing")

        scaler = RobustScaler()
        X_train = scaler.fit_transform(train_df.values)
        X_test = scaler.transform(test_df.values)
        print(f"  Data scaled. Shapes: Train={X_train.shape}, Test={X_test.shape}")

        # Add sensor type as additional features
        print("  Adding sensor type features...")
        X_train = np.hstack([X_train, np.tile(sensor_type_features, (X_train.shape[0], 1))])
        X_test = np.hstack([X_test, np.tile(sensor_type_features, (X_test.shape[0], 1))])
        print(f"  Final feature dimensions: Train={X_train.shape}, Test={X_test.shape}")

        # 10. Label processing
        print("\nProcessing labels...")
        # Training set labels (all normal)
        y_train = np.zeros(len(X_train))

        # Convert test set labels
        # Original labels: -1 = attack, 1 = normal → Convert to: 1 = attack, 0 = normal
        if label_col:
            y_test = np.where(y_test == -1, 1, 0)
        else:
            print("  Using dummy labels (all normal)")
            y_test = np.zeros(len(X_test))

        # 11. Dataset information
        num_classes = 2
        class_names = ['Normal', 'Attack']

        # 12. Print dataset statistics
        print(f"\n{' WADI Dataset Summary ':-^80}")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Feature dimension: {X_train.shape[1]}")
        print(f"  Attack ratio in test set: {np.sum(y_test) / len(y_test):.4f}")
        print(f"  Sensor types: {len(sensor_type_mapping)}")
        print(f"{'-' * 80}")

        return X_train, X_test, y_train, y_test, num_classes, class_names

    except Exception as e:
        print(f"\n{' ERROR ':-^80}")
        print(f"Error loading WADI dataset: {str(e)}")
        print(traceback.format_exc())
        print(f"{'-' * 80}")
        raise
