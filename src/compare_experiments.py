import mlflow
import pandas as pd

def find_best_run():
    # 1. Set the experiment name to search
    experiment_name = "Heart_Disease_Classification"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found. Make sure you've run the training script!")
        return

    # 2. Programmatically fetch all runs from this experiment using mlflow.search_runs
    print(f"Querying MLflow for runs in experiment: '{experiment_name}'...")
    runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if runs_df.empty:
        print("No runs found in this experiment.")
        return
    
    # 3. Sort the runs by your primary metric (accuracy) in descending order
    # The columns are automatically named 'metrics.accuracy', 'params.max_depth', etc.
    runs_df_sorted = runs_df.sort_values(by="metrics.accuracy", ascending=False)
    
    # 4. Extract the top run
    best_run = runs_df_sorted.iloc[0]
    
    print("\n" + "="*40)
    print("🏆 BEST RUN FOUND 🏆")
    print(f"Run ID: {best_run['run_id']}")
    print(f"Status: {best_run['status']}")
    print("-"*40)
    print("Top Metrics:")
    print(f"  - Accuracy:  {best_run['metrics.accuracy']:.4f}")
    print(f"  - Precision: {best_run['metrics.precision']:.4f}")
    print(f"  - Recall:    {best_run['metrics.recall']:.4f}")
    print("-"*40)
    print("Hyperparameters Used:")
    print(f"  - max_depth:    {best_run['params.max_depth']}")
    print(f"  - n_estimators: {best_run['params.n_estimators']}")
    print("="*40)

if __name__ == "__main__":
    find_best_run()