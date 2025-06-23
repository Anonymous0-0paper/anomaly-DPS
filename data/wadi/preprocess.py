import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import os
import re
import glob
from datetime import datetime

def clean_column_names(columns):
    """Simplify lengthy column names"""
    cleaned = []
    for col in columns:
        # Extract the variable name part
        match = re.search(r'[^\\]+$', col)
        if match:
            clean_name = match.group(0).replace('\\', '_')
        else:
            clean_name = col
        # Replace special characters
        clean_name = clean_name.replace(',', '_').replace(';', '_')
        cleaned.append(clean_name)
    return cleaned

def parse_wadi_timestamp(date_str, time_str):
    """Parse WADI specific time format"""
    try:
        # Try parsing the date (format: m/d/yyyy)
        date_part = datetime.strptime(date_str, '%m/%d/%Y')
        # Parse the time (format: h:mm:ss.fff AM/PM)
        if '.' in time_str:
            time_part = datetime.strptime(time_str, '%I:%M:%S.%f %p')
        else:
            time_part = datetime.strptime(time_str, '%I:%M:%S %p')
        # Combine date and time
        return datetime.combine(date_part.date(), time_part.time())
    except Exception as e:
        print(f"Time parsing error: {date_str} {time_str} - {e}")
        return pd.NaT

def preprocess_wadi(input_path, output_path, is_train=False, scaler=None, attack_periods=None):
    """Main function to process the WADI dataset"""
    print(f"\nProcessing {'training set' if is_train else 'test set'}: {input_path}")

    # 1. Read raw data
    try:
        # Read the first 5 rows to determine the structure
        with open(input_path, 'r') as f:
            lines = [next(f) for _ in range(10)]
        # Find the data start row
        header_idx = None
        for i, line in enumerate(lines):
            if "Row,Date,Time" in line:
                header_idx = i
                break
        if header_idx is None:
            print("Warning: Standard header row not found, trying auto-detection")
            # Find the row with many columns
            for i, line in enumerate(lines):
                if len(line.split(',')) > 100:
                    header_idx = i
                    break
        if header_idx is None:
            header_idx = 3  # Default value
        print(f"Using header row: {header_idx + 1}")

        # Read data
        df = pd.read_csv(input_path, skiprows=header_idx, low_memory=False)
        print(f"Raw data shape: {df.shape}")

        # 2. Clean column names
        df.columns = clean_column_names(df.columns)
        print(f"Cleaned column names: {list(df.columns)[:5]}... total {len(df.columns)} columns")

        # 3. Process timestamp
        if 'Date' in df.columns and 'Time' in df.columns:
            # Apply custom time parser
            df['Timestamp'] = df.apply(
                lambda row: parse_wadi_timestamp(str(row['Date']), str(row['Time'])),
                axis=1
            )
            # Remove invalid timestamps
            initial_count = len(df)
            df = df.dropna(subset=['Timestamp'])
            df.set_index('Timestamp', inplace=True)
            print(f"Valid timestamp records: {len(df)}/{initial_count}")

            # Remove redundant columns
            df.drop(['Row', 'Date', 'Time'], axis=1, errors='ignore', inplace=True)
        else:
            print("Error: Missing Date or Time columns")
            return None

        # 4. Convert data types
        # Try to convert all columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 5. Handle missing values
        print("Handling missing values...")
        # Drop columns that are all NaN
        df.dropna(axis=1, how='all', inplace=True)
        # Drop rows that are all NaN
        df.dropna(axis=0, how='all', inplace=True)
        # Interpolate to fill remaining missing values
        df = df.interpolate(method='time', limit_direction='both')
        # Fill remaining NaN with column mean
        df = df.fillna(df.mean())
        print(f"Processed data shape: {df.shape}")

        # 6. Feature scaling (only for training set or when scaler is provided)
        if is_train or scaler:
            # Identify continuous variables (excluding attack labels)
            numeric_cols = [col for col in df.columns if col != 'attack_label']
            if numeric_cols:
                if is_train:
                    # Training set: create and fit scaler
                    scaler = RobustScaler()
                    scaled_values = scaler.fit_transform(df[numeric_cols])
                    print("Creating and fitting scaler")
                else:
                    # Test set: use training set's scaler
                    scaled_values = scaler.transform(df[numeric_cols])
                    print("Using training set scaler")
                # Update with scaled values
                df[numeric_cols] = scaled_values

        # 7. Add attack labels for the test set
        if not is_train and attack_periods is not None:
            print("Adding attack labels...")
            df['attack_label'] = 0
            for start, end in attack_periods:
                start_dt = pd.Timestamp(start)
                end_dt = pd.Timestamp(end)
                mask = (df.index >= start_dt) & (df.index <= end_dt)
                df.loc[mask, 'attack_label'] = 1
            attack_percent = df['attack_label'].mean() * 100
            print(f"Attack data proportion: {attack_percent:.2f}%")

        # 8. Save processed data
        print(f"Saving to: {output_path} ({len(df)} rows, {len(df.columns)} columns)")
        df.to_csv(output_path)
        return scaler if is_train else None
    except Exception as e:
        print(f"Processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Main processing workflow
if __name__ == "__main__":
    os.makedirs('processed_data', exist_ok=True)

    # Define attack periods
    attack_periods = [
        ('2017-10-09 19:25:00', '2017-10-09 19:50:16'),
        ('2017-10-09 20:38:00', '2017-10-09 20:53:44'),
        ('2017-10-09 21:35:00', '2017-10-09 21:50:16')
    ]

    # Find files
    base_dir = 'initialData/WADI.A1_9 Oct 2017'
    train_files = glob.glob(f"{base_dir}/*14days*.csv")
    test_files = glob.glob(f"{base_dir}/*attack*.csv")
    if not train_files or not test_files:
        print("Error: WADI files not found")
        exit()

    train_path = train_files[0]
    test_path = test_files[0]
    print(f"Training file: {train_path}")
    print(f"Test file: {test_path}")

    # Process training set
    print("\n" + "=" * 50)
    print("Processing training set")
    print("=" * 50)
    train_scaler = preprocess_wadi(
        train_path,
        'processed_data/train.csv',
        is_train=True
    )
    if train_scaler is None:
        print("Training set processing failed")
        exit()

    # Process test set
    print("\n" + "=" * 50)
    print("Processing test set")
    print("=" * 50)
    preprocess_wadi(
        test_path,
        'processed_data/test.csv',
        is_train=False,
        scaler=train_scaler,
        attack_periods=attack_periods
    )
    print("\nProcessing completed! Files saved in processed_data/ directory")
