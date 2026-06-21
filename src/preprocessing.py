import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_path="data/raw_data.csv", output_dir="data/processed/"):
    # 1. Create processed directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Load raw data
    print(f"Reading raw data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # 3. Handle column renaming safely
    if 'num' in df.columns:
        df.rename(columns={'num': 'target'}, inplace=True)
    elif 'target' not in df.columns:
        raise KeyError(f"Could not find a target column ('num' or 'target'). Available columns: {list(df.columns)}")
        
    # 4. Binary target conversion
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    
    # 5. Handle features and target
    X = df.drop(columns=['target'])
    y = df['target']
    
    # 6. Split data using random_state=42 and stratify to keep balance
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 🌟 NEW FIX: Impute missing values using training column medians
    # This prevents NaNs from propagating into scaling and model steps
    for col in X_train.columns:
        median_value = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_value)
        X_val[col] = X_val[col].fillna(median_value)
    
    # 7. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 8. Save processed datasets back to disk
    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    pd.DataFrame(X_val_scaled, columns=X.columns).to_csv(os.path.join(output_dir, "X_val.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_val.to_csv(os.path.join(output_dir, "y_val.csv"), index=False)
    
    print("Preprocessing complete! Saved split datasets to", output_dir)

if __name__ == "__main__":
    preprocess_data()