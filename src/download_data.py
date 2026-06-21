import os
import pandas as pd
from ucimlrepo import fetch_ucirepo 

def download_dataset():
    # Ensure the data directory exists
    os.makedirs("data", exist_ok=True)
    
    print("Fetching Heart Disease dataset from UCI...")
    # Fetch dataset (ID 45 is for Heart Disease)
    heart_disease = fetch_ucirepo(id=45) 

    # Map features and targets into a single DataFrame
    X = heart_disease.data.features 
    y = heart_disease.data.targets 
    df = pd.concat([X, y], axis=1)

    # Save to your pipeline's local data directory
    output_path = 'data/raw_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Successfully saved to {output_path}!")

if __name__ == "__main__":
    download_dataset()