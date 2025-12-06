import json
import os
import pandas as pd
from datetime import datetime
import hashlib

class ExperimentTracker:
    """Track and compare ML experiments."""
    
    def __init__(self, experiments_dir="experiments"):
        self.experiments_dir = experiments_dir
        os.makedirs(experiments_dir, exist_ok=True)
        self.current_experiment = None
        
    def start_experiment(self, name=None, dataset_info=None):
        """Start a new experiment."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = hashlib.md5(f"{timestamp}_{name}".encode()).hexdigest()[:8]
        
        self.current_experiment = {
            "id": exp_id,
            "name": name or f"experiment_{timestamp}",
            "timestamp": timestamp,
            "dataset_info": dataset_info or {},
            "stages": {},
            "models": [],
            "metrics": {},
            "artifacts": []
        }
        
        return exp_id
    
    def log_stage(self, stage_name, data):
        """Log a pipeline stage."""
        if not self.current_experiment:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        self.current_experiment["stages"][stage_name] = {
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
    
    def log_model(self, model_name, metrics, params, artifacts=None):
        """Log a trained model."""
        if not self.current_experiment:
            raise ValueError("No active experiment.")
        
        model_entry = {
            "name": model_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "params": params,
            "artifacts": artifacts or []
        }
        
        self.current_experiment["models"].append(model_entry)
    
    def log_metric(self, key, value):
        """Log a single metric."""
        if not self.current_experiment:
            raise ValueError("No active experiment.")
        
        self.current_experiment["metrics"][key] = value
    
    def log_artifact(self, artifact_path, artifact_type="file"):
        """Log an artifact (file, plot, etc)."""
        if not self.current_experiment:
            raise ValueError("No active experiment.")
        
        self.current_experiment["artifacts"].append({
            "path": artifact_path,
            "type": artifact_type,
            "timestamp": datetime.now().isoformat()
        })
    
    def end_experiment(self):
        """Save and close current experiment."""
        if not self.current_experiment:
            return None
        
        exp_id = self.current_experiment["id"]
        exp_path = os.path.join(self.experiments_dir, f"{exp_id}.json")
        
        with open(exp_path, 'w') as f:
            json.dump(self.current_experiment, f, indent=2, default=str)
        
        result = self.current_experiment
        self.current_experiment = None
        return result
    
    def list_experiments(self):
        """List all experiments."""
        experiments = []
        for filename in os.listdir(self.experiments_dir):
            if filename.endswith('.json'):
                with open(os.path.join(self.experiments_dir, filename), 'r') as f:
                    experiments.append(json.load(f))
        
        return sorted(experiments, key=lambda x: x.get("timestamp", ""), reverse=True)
    
    def get_experiment(self, exp_id):
        """Get a specific experiment."""
        exp_path = os.path.join(self.experiments_dir, f"{exp_id}.json")
        if os.path.exists(exp_path):
            with open(exp_path, 'r') as f:
                return json.load(f)
        return None
    
    def compare_experiments(self, exp_ids):
        """Compare multiple experiments."""
        experiments = [self.get_experiment(exp_id) for exp_id in exp_ids]
        experiments = [e for e in experiments if e is not None]
        
        if not experiments:
            return None
        
        comparison = {
            "experiments": [],
            "best_models": {},
            "metric_comparison": {}
        }
        
        for exp in experiments:
            exp_summary = {
                "id": exp["id"],
                "name": exp["name"],
                "timestamp": exp["timestamp"],
                "n_models": len(exp.get("models", [])),
                "best_accuracy": max([m["metrics"].get("accuracy", 0) for m in exp.get("models", [])], default=0)
            }
            comparison["experiments"].append(exp_summary)
        
        return comparison
    
    def get_leaderboard(self, metric="accuracy"):
        """Get global leaderboard across all experiments."""
        all_models = []
        
        for exp in self.list_experiments():
            for model in exp.get("models", []):
                all_models.append({
                    "experiment_id": exp["id"],
                    "experiment_name": exp["name"],
                    "model_name": model["name"],
                    "timestamp": model["timestamp"],
                    **model["metrics"]
                })
        
        if not all_models:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_models)
        if metric in df.columns:
            df = df.sort_values(by=metric, ascending=False)
        
        return df
