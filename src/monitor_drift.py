import os
import sys
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
# Import the explicit metric class to access its structured results safely
from evidently.metrics import DatasetDriftMetric

def monitor_data_drift():
    # 1. Load your baseline training data as the reference dataset
    print("Loading reference training data...")
    if not os.path.exists("data/processed/X_train.csv"):
        print("❌ Error: Processed training data not found. Run preprocessing first.")
        sys.exit(1)
        
    reference_df = pd.read_csv("data/processed/X_train.csv")
    
    # 2. Simulate a 'production' dataset by adding slight noise to create synthetic drift
    print("Simulating production data stream...")
    production_df = reference_df.copy()
    
    if 'chol' in production_df.columns:
        production_df['chol'] = production_df['chol'] + 1.5
    if 'age' in production_df.columns:
        production_df['age'] = production_df['age'] + 0.8

    # 3. Initialize and run the Evidently Data Drift report using explicit metric objects
    print("Running Evidently feature drift detection checks...")
    drift_metric = DatasetDriftMetric()
    drift_report = Report(metrics=[drift_metric])
    drift_report.run(reference_data=reference_df, current_data=production_df)
    
   # 4. Programmatically extract metrics using the safe API object fields
    metric_results = drift_metric.get_result()
    
    number_of_drifted_features = metric_results.number_of_drifted_columns
    total_features = metric_results.number_of_columns
    drifted_features_share = number_of_drifted_features / total_features if total_features > 0 else 0.0
    dataset_drift_detected = metric_results.dataset_drift

    print("\n" + "="*40)
    print("📊 EVIDENTLY DRIFT ANALYSIS SUMMARY 📊")
    print(f"  - Total Features Tracked:   {total_features}")
    print(f"  - Number of Features Drifted: {number_of_drifted_features}")
    print(f"  - Share of Drifted Features:  {drifted_features_share:.2%}")
    print(f"  - Overall Dataset Drift Status: {'🚨 DRIFT DETECTED' if dataset_drift_detected else '✅ STABLE'}")
    print("="*40)

    # 5. Save the visual interactive HTML report to reports/
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/drift_report.html"
    drift_report.save_html(report_path)
    print(f"💾 Interactive drift dashboard saved to: {report_path}")

    # 6. Safety Check Guardrail: Exit with code 1 if more than 30% of features show drift
    DRIFT_THRESHOLD = 0.30
    if drifted_features_share > DRIFT_THRESHOLD:
        print(f"❌ ALERT: Drift share ({drifted_features_share:.2%}) exceeded safety threshold ({DRIFT_THRESHOLD:.2%})!")
        sys.exit(1)
    else:
        print("✅ Success: Feature metrics are within safe operational limits.")

if __name__ == "__main__":
    monitor_data_drift()