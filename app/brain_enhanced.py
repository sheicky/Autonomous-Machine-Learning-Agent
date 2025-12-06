"""
Enhanced AutoML Brain with all improvements:
- Optuna integration for faster hyperparameter tuning
- Parallel model training
- Better error recovery
- SHAP explainability
- Advanced feature engineering
- Production-ready code generation
"""

import os
import json
import base64
import pandas as pd
from dotenv import load_dotenv
from daytona import Daytona, DaytonaConfig
import requests
import re
import certifi
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
import hashlib
import sys

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Fix for macOS SSL Certificate errors
os.environ['SSL_CERT_FILE'] = certifi.where()

# Load environment variables
load_dotenv()

class DaytonaExecutor:
    """Enhanced Daytona executor with retry logic and better error handling."""
    
    def __init__(self, api_key=None, language='python', max_retries=3):
        self.api_key = api_key or os.getenv("DAYTONA_API_KEY")
        if not self.api_key:
            raise ValueError("Daytona API Key is required.")
        
        self.config = DaytonaConfig(api_key=self.api_key)
        self.client = Daytona(self.config)
        self.sandbox = None
        self.language = language
        self.max_retries = max_retries

    def create_sandbox(self, retry_count=0):
        """Create sandbox with automatic retry."""
        try:
            self.sandbox = self.client.create()
            
            try:
                self.sandbox.start()
            except Exception as start_err:
                print(f"[DEBUG] sandbox.start() note: {start_err}")
            
            self.sandbox.wait_for_sandbox_start(timeout=60)
            return True, "Sandbox created successfully."
            
        except Exception as e:
            if retry_count < self.max_retries:
                wait_time = 2 ** retry_count
                print(f"Sandbox creation failed, retrying in {wait_time}s...")
                time.sleep(wait_time)
                return self.create_sandbox(retry_count + 1)
            return False, f"Failed to create sandbox after {self.max_retries} attempts: {str(e)}"

    def ensure_sandbox_ready(self):
        """Ensures sandbox exists and is started."""
        if not self.sandbox:
            success, msg = self.create_sandbox()
            if not success:
                raise RuntimeError(msg)
        
        try:
            self.sandbox.wait_for_sandbox_start(timeout=30)
        except Exception as e:
            try:
                self.sandbox.start()
                self.sandbox.wait_for_sandbox_start(timeout=60)
            except Exception as start_err:
                raise RuntimeError(f"Failed to start sandbox: {start_err}")

    def install_dependencies(self, packages):
        """Install Python packages in sandbox."""
        if not self.sandbox:
            return False, "Sandbox not initialized."
        
        install_cmd = f"pip install {' '.join(packages)}"
        return self.execute_command(install_cmd)

    def upload_data(self, df, filename='dataset.csv'):
        """Upload DataFrame to sandbox."""
        try:
            self.ensure_sandbox_ready()
        except RuntimeError as e:
            return {"exit_code": -1, "stdout": "", "stderr": str(e)}
        
        try:
            csv_string = df.to_csv(index=False)
            csv_bytes = csv_string.encode('utf-8')
            
            work_dir = self.sandbox.get_user_home_dir()
            dest_path = f"{work_dir}/{filename}"
            
            try:
                self.sandbox.fs.upload_file(csv_bytes, dest_path)
                method = "fs.upload_file"
            except Exception as fs_error:
                print(f"[DEBUG] fs.upload_file failed: {fs_error}")
                b64_data = base64.b64encode(csv_bytes).decode('utf-8')
                write_code = f"""
import base64
import os
os.chdir('{work_dir}')
csv_data = base64.b64decode('{b64_data}')
with open('{filename}', 'wb') as f:
    f.write(csv_data)
print(f'Wrote {{len(csv_data)}} bytes to {filename}')
"""
                response = self.sandbox.process.code_run(write_code)
                if response.exit_code != 0:
                    return {"exit_code": -1, "stdout": "", "stderr": f"Upload failed: {response.result}"}
                method = "base64 fallback"
            
            verify_code = f"""
import os
os.chdir('{work_dir}')
if os.path.exists('{filename}'):
    size = os.path.getsize('{filename}')
    print(f'VERIFIED: {filename} exists, size={{size}} bytes')
else:
    print(f'ERROR: {filename} not found')
"""
            verify_response = self.sandbox.process.code_run(verify_code)
            
            return {
                "exit_code": 0,
                "stdout": f"Upload via {method}. Verify: {verify_response.result}",
                "stderr": ""
            }
        except Exception as e:
            import traceback
            return {"exit_code": -1, "stdout": "", "stderr": f"Upload failed: {str(e)}\n{traceback.format_exc()}"}

    def execute_command(self, command):
        """Execute shell command in sandbox."""
        try:
            self.ensure_sandbox_ready()
        except RuntimeError as e:
            return {"exit_code": -1, "stdout": "", "stderr": str(e)}
            
        try:
            response = self.sandbox.process.code_run(command)
            result = {
                "exit_code": response.exit_code,
                "stdout": response.result,
                "stderr": "" 
            }
            if response.exit_code != 0:
                result["stderr"] = response.result
            return result
        except Exception as e:
            return {"exit_code": -1, "stdout": "", "stderr": str(e)}

    def execute_code(self, code_string):
        """Execute Python code in sandbox."""
        try:
            self.ensure_sandbox_ready()
        except RuntimeError as e:
            return {"exit_code": -1, "stdout": "", "stderr": str(e)}

        try:
            work_dir = self.sandbox.get_user_root_dir()
            script_name = "temp_script.py"
            script_path = f"{work_dir}/{script_name}"
            
            full_code = f"""#!/usr/bin/env python3
import os
import sys
os.chdir('{work_dir}')
sys.path.insert(0, '{work_dir}')
{code_string}
"""
            
            self.sandbox.fs.upload_file(full_code.encode('utf-8'), script_path)
            
            exec_response = self.sandbox.process.exec(
                f"python3 {script_name}", 
                cwd=work_dir,
                timeout=300
            )
            
            result = {
                "exit_code": exec_response.exit_code,
                "stdout": exec_response.result,
                "stderr": ""
            }
            if exec_response.exit_code != 0:
                result["stderr"] = f"Error (Exit Code {exec_response.exit_code}): {exec_response.result}"
            return result
        except Exception as e:
            try:
                work_dir = self.sandbox.get_user_root_dir()
                full_code = f"""
import os
os.chdir('{work_dir}')
{code_string}
"""
                response = self.sandbox.process.code_run(full_code)
                result = {
                    "exit_code": response.exit_code,
                    "stdout": response.result,
                    "stderr": ""
                }
                if response.exit_code != 0:
                    result["stderr"] = f"Error: {response.result}"
                return result
            except Exception as e2:
                return {"exit_code": -1, "stdout": "", "stderr": f"Both exec and code_run failed: {str(e)}, {str(e2)}"}

    def download_file(self, remote_path):
        """Download file from sandbox."""
        if not self.sandbox:
            return None
        
        try:
            work_dir = self.sandbox.get_user_root_dir()
            full_path = f"{work_dir}/{remote_path}"
            content = self.sandbox.fs.download_file(full_path)
            return content
        except Exception as e:
            print(f"Download failed: {e}")
            return None

    def cleanup(self):
        """Cleanup sandbox resources."""
        if self.sandbox:
            try:
                self.sandbox.delete()
                self.sandbox = None
                return True
            except Exception as e:
                print(f"Error cleaning up sandbox: {e}")
                return False
        return True



class Leaderboard:
    """Enhanced leaderboard with more metrics."""
    
    def __init__(self):
        self.records = []

    def add_entry(self, model_name, accuracy, precision, recall, params, 
                  feature_importance=None, train_accuracy=None, confusion_matrix=None,
                  f1_score=None, roc_auc=None, training_time=None):
        self.records.append({
            "Model": model_name,
            "Accuracy": round(float(accuracy), 4),
            "TrainAccuracy": round(float(train_accuracy), 4) if train_accuracy else None,
            "Precision": round(float(precision), 4),
            "Recall": round(float(recall), 4),
            "F1Score": round(float(f1_score), 4) if f1_score else None,
            "ROC_AUC": round(float(roc_auc), 4) if roc_auc else None,
            "TrainingTime": round(float(training_time), 2) if training_time else None,
            "Parameters": str(params),
            "FeatureImportance": feature_importance,
            "ConfusionMatrix": confusion_matrix
        })

    def get_dataframe(self):
        if not self.records:
            return pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall"])
        return pd.DataFrame(self.records).sort_values(by="Accuracy", ascending=False)


class EnhancedAutoMLAgent:
    """Enhanced AutoML Agent with all improvements."""
    
    def __init__(self, executor: DaytonaExecutor):
        self.executor = executor
        self.leaderboard = Leaderboard()
        self.history_file = "history.json"
        self.history = self.load_history()
        self._llm_cache = {}
        self.problem_type = None  # 'binary', 'multiclass', 'regression'
        
    def load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []

    def save_history(self, meta_data):
        self.history.append(meta_data)
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f)

    def detect_problem_type(self, target_series):
        """Auto-detect problem type."""
        n_unique = target_series.nunique()
        
        if target_series.dtype in ['float64', 'float32'] and n_unique > 20:
            return 'regression'
        elif n_unique == 2:
            return 'binary'
        elif n_unique <= 20:
            return 'multiclass'
        else:
            return 'regression'

    def call_llm_with_retry(self, prompt, retries=5, delay=20, use_cache=True):
        """Call OpenRouter API with retry and caching."""
        cache_key = hashlib.md5(prompt.encode()).hexdigest()
        if use_cache and cache_key in self._llm_cache:
            print(f"[CACHE HIT] Using cached LLM response")
            return self._llm_cache[cache_key]
        
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENROUTER_API_KEY")
        
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "AutoML Agent"
        }
        
        # Try multiple models in order of preference
        models = [
            os.getenv("OPENROUTER_MODEL", "anthropic/claude-opus-4.5"),
            "google/gemini-3-pro-preview",
            "google/gemini-2.0-flash-exp:free",
            "openai/gpt-5.1-codex-max",
            "anthropic/claude-opus-4.5"
        ]
        
        last_error = None
        
        for model in models:
            data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4000  # Limit response size
            }
            
            for attempt in range(min(retries, 3)):  # Max 3 attempts per model
                try:
                    print(f"[LLM] Trying {model} (attempt {attempt + 1})...")
                    response = requests.post(url, headers=headers, json=data, timeout=180)
                    
                    if response.status_code == 429:
                        # Rate limit - try next model instead of waiting too long
                        if attempt == 0:
                            print(f"Rate limit on {model}, trying next model...")
                            break
                        wait_time = min(delay * (2 ** attempt), 60)  # Cap at 60s
                        print(f"Rate limit (429). Waiting {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue
                    
                    if response.status_code == 408 or response.status_code == 504:
                        # Timeout - try next model
                        print(f"Timeout on {model}, trying next model...")
                        break
                        
                    response.raise_for_status()
                    result = response.json()['choices'][0]['message']['content']
                    
                    if use_cache:
                        self._llm_cache[cache_key] = result
                    
                    print(f"[LLM] Success with {model}")
                    return result
                    
                except requests.exceptions.Timeout as e:
                    last_error = e
                    print(f"Timeout on {model}, trying next model...")
                    break
                    
                except requests.exceptions.RequestException as e:
                    last_error = e
                    if attempt < min(retries, 3) - 1:
                        wait_time = min(delay * (2 ** attempt), 60)
                        print(f"API error: {str(e)[:100]}. Retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"Failed with {model}, trying next model...")
                        break
                        
                except Exception as e:
                    last_error = e
                    print(f"Error with {model}: {str(e)[:100]}")
                    break
        
        # All models failed
        error_msg = f"All LLM models failed. Last error: {str(last_error)}"
        print(f"[ERROR] {error_msg}")
        raise Exception(error_msg)

    def _clean_llm_code(self, code):
        """Clean LLM-generated code by removing explanatory text and markdown."""
        # Remove markdown code blocks
        if '```python' in code:
            code = code.split('```python')[1].split('```')[0]
        elif '```' in code:
            parts = code.split('```')
            for part in parts:
                if 'import' in part:
                    code = part
                    break
        
        # Find first import statement
        lines = code.split('\n')
        code_start = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                code_start = i
                break
        
        code = '\n'.join(lines[code_start:])
        
        # Remove trailing explanatory text
        code_lines = code.split('\n')
        clean_lines = []
        for line in code_lines:
            stripped = line.strip()
            # Stop at common explanatory phrases
            if any(stripped.startswith(phrase) for phrase in ['Here', 'This script', 'Note:', 'The above', 'This code']):
                break
            clean_lines.append(line)
        
        return '\n'.join(clean_lines).strip()
    
    def analyze_and_plan(self, data_summary, quality_report=None):
        """Enhanced planning with data quality insights."""
        
        prompt = f"""
You are an expert Data Scientist. Analyze this dataset and create a comprehensive ML plan.

Dataset Summary:
{json.dumps(data_summary, default=str)}

Data Quality Report:
{json.dumps(quality_report, default=str) if quality_report else "No quality issues detected"}

Create a plan that addresses data quality issues and optimizes for the problem type.

Output JSON format:
{{
    "plan_overview": "Brief explanation",
    "problem_type": "binary/multiclass/regression",
    "target_column": "column_name",
    "preprocessing_strategy": "Detailed strategy addressing quality issues",
    "feature_engineering_strategy": "Specific transformations (polynomial, datetime, interactions)",
    "feature_selection_strategy": "Method (SelectKBest, RFE, SelectFromModel)",
    "models": [
        {{
            "name": "ModelName",
            "params": {{"param": ["value1", "value2"]}},
            "regularization_params": {{"C": [0.01, 0.1, 1], "max_depth": [3, 5, 7]}}
        }}
    ],
    "evaluation_metrics": ["accuracy", "f1", "roc_auc"],
    "cross_validation_folds": 5,
    "handle_imbalance": "SMOTE/class_weight/none",
    "outlier_handling": "robust_scaler/winsorize/none"
}}
"""
        
        try:
            text = self.call_llm_with_retry(prompt)
            
            if text.startswith("```json"):
                text = text.split("```json")[1].split("```")[0]
            elif text.startswith("```"):
                text = text.split("```")[1].split("```")[0]
            
            text = text.strip()
            plan = json.loads(text)
            
            # Store problem type
            self.problem_type = plan.get('problem_type', 'binary')
            
            return plan
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse LLM response: {e}")
            return {"error": f"LLM returned invalid JSON: {str(e)}", "raw_response": text[:500]}
        except Exception as e:
            return {"error": str(e)}


    def generate_preprocessing_code(self, data_summary, target_col, prep_strategy, 
                                    eng_strategy, quality_report=None):
        """Generate enhanced preprocessing code with quality fixes."""
        
        quality_fixes = ""
        if quality_report and quality_report.get('recommendations'):
            quality_fixes = "\n".join([
                f"# Fix: {rec.get('action', '')}" 
                for rec in quality_report['recommendations'][:3]
            ])
        
        prompt = f"""
OUTPUT ONLY PYTHON CODE. NO EXPLANATIONS. START WITH IMPORTS.

Target column: '{target_col}'
Strategy: {prep_strategy}
Feature Engineering: {eng_strategy}
Quality Fixes: {quality_fixes}

Write preprocessing script that:
1. Loads 'dataset.csv'
2. Handles missing values
3. Handles outliers with RobustScaler if needed
4. Does feature engineering (polynomial, datetime, interactions, log transforms)
5. Encodes categoricals with OneHotEncoder
6. Scales with StandardScaler or RobustScaler
7. Handles class imbalance with SMOTE if ratio > 3:1
8. Splits train/test (80/20, stratified)
9. Saves: X_train_processed.csv, X_test_processed.csv, y_train.csv, y_test.csv, preprocessor.pkl
10. Prints JSON: {{"status": "complete", "n_features": N}}

CRITICAL: Start directly with: import pandas as pd
NO text before imports. NO explanations after code.
"""
        
        try:
            code = self.call_llm_with_retry(prompt)
            code = self._clean_llm_code(code)
            return code
        except Exception as e:
            return f"print('Error generating preprocessing code: {str(e)}')"

    def generate_feature_selection_code(self, target_col, strategy):
        """Generate feature selection code."""
        
        prompt = f"""
Write Python script for feature selection using {strategy}.

Requirements:
1. Load processed data files
2. Use {strategy} (SelectKBest with f_classif, RFE, or SelectFromModel)
3. Select top 50% of features or minimum 10 features
4. Transform train and test sets
5. Save: X_train_selected.csv, X_test_selected.csv, selector.pkl
6. Print JSON with selected feature indices and scores

Imports:
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib
import json
"""
        
        try:
            code = self.call_llm_with_retry(prompt)
            if code.startswith("```python"):
                code = code.split("```python")[1].split("```")[0]
            elif code.startswith("```"):
                code = code.split("```")[1].split("```")[0]
            return code.strip()
        except Exception as e:
            return f"print('Error: {str(e)}')"

    def generate_training_code_optuna(self, model_config, target_col, use_selected_features=False):
        """Generate training code with Optuna optimization."""
        
        data_prefix = "selected" if use_selected_features else "processed"
        
        prompt = f"""
Write Python script to train {model_config['name']} using OPTUNA for hyperparameter optimization.

Data: X_train_{data_prefix}.csv, X_test_{data_prefix}.csv, y_train.csv, y_test.csv

Steps:
1. Load data and preprocessor
2. Define Optuna objective function with these params: {model_config['params']}
3. Use optuna.create_study(direction='maximize') with 20 trials
4. Train best model
5. Calculate metrics: accuracy, precision, recall, f1, roc_auc, training_time
6. Calculate SHAP values for feature importance (top 20 features)
7. Get confusion matrix
8. Save model pipeline to '{model_config['name'].replace(' ', '_')}_model.pkl'
9. Print JSON with all metrics

Required imports:
import optuna
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import shap
import joblib
import json
import time

Example Optuna objective:
def objective(trial):
    params = {{
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200)
    }}
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20, show_progress_bar=False)
"""
        
        try:
            code = self.call_llm_with_retry(prompt)
            if code.startswith("```python"):
                code = code.split("```python")[1].split("```")[0]
            elif code.startswith("```"):
                code = code.split("```")[1].split("```")[0]
            return code.strip()
        except Exception as e:
            return f"print('Error: {str(e)}')"

    def train_models_parallel(self, models, target_col, use_selected_features=False, 
                             progress_callback=None):
        """Train multiple models in parallel."""
        
        results = []
        
        def train_single_model(model_config):
            try:
                if progress_callback:
                    progress_callback(f"Training {model_config['name']}...")
                
                code = self.generate_training_code_optuna(model_config, target_col, use_selected_features)
                res = self.executor.execute_code(code)
                
                if res['exit_code'] == 0:
                    metrics = self.extract_json(res['stdout'])
                    if metrics:
                        return {
                            'success': True,
                            'model': model_config['name'],
                            'metrics': metrics,
                            'output': res['stdout']
                        }
                
                return {
                    'success': False,
                    'model': model_config['name'],
                    'error': res['stderr']
                }
            except Exception as e:
                return {
                    'success': False,
                    'model': model_config['name'],
                    'error': str(e)
                }
        
        # Train models in parallel (max 3 concurrent)
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_model = {executor.submit(train_single_model, model): model for model in models}
            
            for future in as_completed(future_to_model):
                result = future.result()
                results.append(result)
                
                if result['success'] and progress_callback:
                    metrics = result['metrics']
                    progress_callback(f"✅ {result['model']} - Acc: {metrics.get('accuracy', 0):.4f}")
        
        return results


    def generate_ensemble_code(self, top_models, target_col, use_selected_features=False):
        """Generate ensemble code with stacking."""
        
        data_prefix = "selected" if use_selected_features else "processed"
        model_files = [f"{m['Model'].replace(' ', '_')}_model.pkl" for m in top_models]
        
        prompt = f"""
Write script to create an advanced ensemble using StackingClassifier.

Data: X_train_{data_prefix}.csv, X_test_{data_prefix}.csv, y_train.csv, y_test.csv
Models to load: {model_files}

Steps:
1. Load data and models
2. Create StackingClassifier with loaded models as base estimators
3. Use LogisticRegression as final estimator
4. Fit on train data
5. Evaluate on train and test (accuracy, f1, roc_auc, confusion_matrix)
6. Save to 'final_ensemble.pkl'
7. Print JSON metrics

Imports:
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import joblib
import json
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
"""
        
        try:
            code = self.call_llm_with_retry(prompt)
            if code.startswith("```python"):
                code = code.split("```python")[1].split("```")[0]
            elif code.startswith("```"):
                code = code.split("```")[1].split("```")[0]
            return code.strip()
        except Exception as e:
            return f"print('Error: {str(e)}')"

    def generate_fix_overfitting_code(self, model_name, current_params, train_acc, 
                                     test_acc, target_col, use_selected_features=False):
        """Generate code to fix overfitting with stronger regularization."""
        
        data_prefix = "selected" if use_selected_features else "processed"
        
        prompt = f"""
Model '{model_name}' is overfitting: Train={train_acc:.4f}, Test={test_acc:.4f}
Current params: {current_params}

Write script to retrain with STRONGER regularization:
- Random Forest: reduce max_depth by 50%, increase min_samples_leaf by 3x
- Gradient Boosting: reduce learning_rate by 50%, increase n_estimators
- Logistic Regression: reduce C by 10x
- Neural Network: increase alpha by 10x, add dropout

Use Optuna with 15 trials focusing on regularization parameters.
Calculate all metrics and save as '{model_name.replace(' ', '_')}_Fixed_model.pkl'
"""
        
        try:
            code = self.call_llm_with_retry(prompt)
            if code.startswith("```python"):
                code = code.split("```python")[1].split("```")[0]
            elif code.startswith("```"):
                code = code.split("```")[1].split("```")[0]
            return code.strip()
        except Exception as e:
            return f"print('Error: {str(e)}')"

    def generate_api_code(self, model_filename='final_ensemble.pkl'):
        """Generate production-ready FastAPI code with monitoring."""
        
        return f"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import pandas as pd
import joblib
import uvicorn
from datetime import datetime
import json
import os

app = FastAPI(title="AutoML Model API", version="1.0.0")

# Load Model
try:
    model = joblib.load("{model_filename}")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {{e}}")
    model = None

# Monitoring
predictions_log = []

class InferenceRequest(BaseModel):
    data: list  # List of dicts with feature values
    
class InferenceResponse(BaseModel):
    predictions: list
    probabilities: list = None
    timestamp: str
    model_version: str = "1.0.0"

@app.get("/")
def root():
    return {{
        "service": "AutoML Model API",
        "status": "healthy" if model else "model_not_loaded",
        "version": "1.0.0"
    }}

@app.get("/health")
def health_check():
    return {{
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
        "total_predictions": len(predictions_log)
    }}

@app.post("/predict", response_model=InferenceResponse)
def predict(req: InferenceRequest, background_tasks: BackgroundTasks):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        df = pd.DataFrame(req.data)
        predictions = model.predict(df)
        
        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df).tolist()
        
        # Log prediction
        log_entry = {{
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(df),
            "predictions": predictions.tolist()
        }}
        background_tasks.add_task(log_prediction, log_entry)
        
        return InferenceResponse(
            predictions=predictions.tolist(),
            probabilities=probabilities,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/batch_predict")
def batch_predict(req: InferenceRequest):
    \"\"\"Batch prediction endpoint for large datasets.\"\"\"
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        df = pd.DataFrame(req.data)
        predictions = model.predict(df)
        
        return {{
            "predictions": predictions.tolist(),
            "count": len(predictions),
            "timestamp": datetime.now().isoformat()
        }}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/stats")
def get_stats():
    \"\"\"Get prediction statistics.\"\"\"
    if not predictions_log:
        return {{"message": "No predictions yet"}}
    
    total_predictions = sum(log['n_samples'] for log in predictions_log)
    
    return {{
        "total_requests": len(predictions_log),
        "total_predictions": total_predictions,
        "first_prediction": predictions_log[0]['timestamp'] if predictions_log else None,
        "last_prediction": predictions_log[-1]['timestamp'] if predictions_log else None
    }}

def log_prediction(log_entry):
    \"\"\"Background task to log predictions.\"\"\"
    predictions_log.append(log_entry)
    
    # Keep only last 1000 entries
    if len(predictions_log) > 1000:
        predictions_log.pop(0)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

    def generate_docker_code(self):
        """Generate Dockerfile for deployment."""
        
        return """
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and API code
COPY final_ensemble.pkl .
COPY serve.py .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run API
CMD ["python", "serve.py"]
"""

    def extract_json(self, text):
        """Robustly extract JSON from text."""
        try:
            match = re.search(r'\\{.*"model":.*\\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end > start:
                json_candidate = text[start:end]
                try:
                    return json.loads(json_candidate)
                except json.JSONDecodeError:
                    pass
                    
            return None
        except Exception as e:
            print(f"[ERROR] extract_json failed: {e}")
            return None

    def get_data_summary_from_sandbox(self, filename='X_train_processed.csv'):
        """Get summary of processed data from sandbox."""
        
        code = f"""
import pandas as pd
import json
import numpy as np
import os

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

try:
    if not os.path.exists('{filename}'):
        print(json.dumps({{"error": "File not found"}}))
    else:
        df = pd.read_csv('{filename}')
        
        info = {{
            "rows": df.shape[0],
            "columns": df.shape[1],
            "columns_list": list(df.columns[:20]),
            "description": df.describe().to_dict(),
            "head": df.head().to_dict(orient='records')
        }}
        print(json.dumps(info, cls=NpEncoder))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
"""
        res = self.executor.execute_code(code)
        if res['exit_code'] == 0:
            try:
                return json.loads(res['stdout'])
            except:
                return {"error": f"Failed to parse: {res['stdout']}"}
        return {"error": res['stderr']}
