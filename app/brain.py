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

# Fix for macOS SSL Certificate errors
os.environ['SSL_CERT_FILE'] = certifi.where()

# Load environment variables
load_dotenv()

class DaytonaExecutor:
    def __init__(self, api_key=None, language='python'):
        self.api_key = api_key or os.getenv("DAYTONA_API_KEY")
        if not self.api_key:
            raise ValueError("Daytona API Key is required. Please set DAYTONA_API_KEY in .env or pass it explicitly.")
        
        self.config = DaytonaConfig(api_key=self.api_key)
        self.client = Daytona(self.config)
        self.sandbox = None
        self.language = language

    def create_sandbox(self):
        try:
            self.sandbox = self.client.create()
            
            # Ensure sandbox is started and ready
            try:
                self.sandbox.start()
            except Exception as start_err:
                # Might already be started or auto-started
                print(f"[DEBUG] sandbox.start() note: {start_err}")
            
            # Wait for sandbox to be fully ready
            self.sandbox.wait_for_sandbox_start(timeout=60)
            
            return True, "Sandbox created and started successfully."
        except Exception as e:
            return False, f"Failed to create sandbox: {str(e)}"

    def ensure_sandbox_ready(self):
        """Ensures sandbox exists and is started."""
        if not self.sandbox:
            success, msg = self.create_sandbox()
            if not success:
                raise RuntimeError(msg)
        
        # Make sure it's running
        try:
            self.sandbox.wait_for_sandbox_start(timeout=30)
        except Exception as e:
            # Try to start it
            try:
                self.sandbox.start()
                self.sandbox.wait_for_sandbox_start(timeout=60)
            except Exception as start_err:
                raise RuntimeError(f"Failed to start sandbox: {start_err}")

    def install_dependencies(self, packages):
        if not self.sandbox:
            return False, "Sandbox not initialized."
        
        # Only install if necessary.
        install_cmd = f"pip install {' '.join(packages)}"
        return self.execute_command(install_cmd)

    def upload_data(self, df, filename='dataset.csv'):
        try:
            self.ensure_sandbox_ready()
        except RuntimeError as e:
            return {"exit_code": -1, "stdout": "", "stderr": str(e)}
        
        try:
            # Convert DataFrame to CSV string
            csv_string = df.to_csv(index=False)
            csv_bytes = csv_string.encode('utf-8')
            
            # Get the working directory in sandbox
            work_dir = self.sandbox.get_user_home_dir()
            dest_path = f"{work_dir}/{filename}"
            
            print(f"[DEBUG] Uploading to: {dest_path}")
            print(f"[DEBUG] CSV size: {len(csv_bytes)} bytes")
            
            # Try using the filesystem API to upload the file
            try:
                self.sandbox.fs.upload_file(csv_bytes, dest_path)
                method = "fs.upload_file"
            except Exception as fs_error:
                print(f"[DEBUG] fs.upload_file failed: {fs_error}")
                # Fallback: Write via code execution using base64
                b64_data = base64.b64encode(csv_bytes).decode('utf-8')
                write_code = f"""
import base64
import os
os.chdir('{work_dir}')
csv_data = base64.b64decode('{b64_data}')
with open('{filename}', 'wb') as f:
    f.write(csv_data)
print(f'Wrote {{len(csv_data)}} bytes to {filename}')
print('Files:', os.listdir('.'))
"""
                response = self.sandbox.process.code_run(write_code)
                if response.exit_code != 0:
                    return {"exit_code": -1, "stdout": "", "stderr": f"Fallback upload failed: {response.result}"}
                method = "base64 fallback"
            
            # Verify the upload using code execution (more reliable than fs.list_files)
            verify_code = f"""
import os
os.chdir('{work_dir}')
if os.path.exists('{filename}'):
    size = os.path.getsize('{filename}')
    print(f'VERIFIED: {filename} exists, size={{size}} bytes')
else:
    print(f'ERROR: {filename} not found')
print('All files:', os.listdir('.'))
"""
            verify_response = self.sandbox.process.code_run(verify_code)
            
            return {
                "exit_code": 0,
                "stdout": f"Upload via {method}. Verify result: {verify_response.result}",
                "stderr": ""
            }
        except Exception as e:
            import traceback
            return {"exit_code": -1, "stdout": "", "stderr": f"Upload failed: {str(e)}\n{traceback.format_exc()}"}

    def execute_command(self, command):
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

    def write_file(self, filename, content):
        """
        Writes a file to the sandbox using the filesystem API.
        """
        if not self.sandbox:
            return False
        try:
            work_dir = self.sandbox.get_user_root_dir()
            dest_path = f"{work_dir}/{filename}"
            if isinstance(content, str):
                content = content.encode('utf-8')
            self.sandbox.fs.upload_file(content, dest_path)
            return True
        except Exception as e:
            print(f"[DEBUG] write_file failed: {e}")
            return False

    def execute_code(self, code_string):
        try:
            self.ensure_sandbox_ready()
        except RuntimeError as e:
            return {"exit_code": -1, "stdout": "", "stderr": str(e)}

        try:
            work_dir = self.sandbox.get_user_root_dir()
            
            # Write the code to a temporary Python file and execute it
            # This is more reliable than code_run for complex scripts
            script_name = "temp_script.py"
            script_path = f"{work_dir}/{script_name}"
            
            # Prepend working directory change and base imports
            full_code = f"""#!/usr/bin/env python3
import os
import sys
os.chdir('{work_dir}')
sys.path.insert(0, '{work_dir}')
{code_string}
"""
            
            # Upload the script file using fs API
            self.sandbox.fs.upload_file(full_code.encode('utf-8'), script_path)
            
            # Execute the script using process.exec with cwd parameter
            exec_response = self.sandbox.process.exec(
                f"python3 {script_name}", 
                cwd=work_dir,
                timeout=300  # 5 minute timeout for long-running operations
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
            # Fallback to code_run if exec fails for any reason
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
                    result["stderr"] = f"Error (Exit Code {response.exit_code}): {response.result}"
                return result
            except Exception as e2:
                return {"exit_code": -1, "stdout": "", "stderr": f"Both exec and code_run failed. exec error: {str(e)}, code_run error: {str(e2)}"}

    def download_file(self, remote_path):
        """
        Reads a file from the sandbox and returns its content.
        """
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
    def __init__(self):
        self.records = []

    def add_entry(self, model_name, accuracy, precision, recall, params, feature_importance=None, train_accuracy=None, confusion_matrix=None):
        self.records.append({
            "Model": model_name,
            "Accuracy": round(float(accuracy), 4),
            "TrainAccuracy": round(float(train_accuracy), 4) if train_accuracy else None,
            "Precision": round(float(precision), 4),
            "Recall": round(float(recall), 4),
            "Parameters": str(params),
            "FeatureImportance": feature_importance,
            "ConfusionMatrix": confusion_matrix
        })

    def get_dataframe(self):
        if not self.records:
            return pd.DataFrame(columns=["Model", "Accuracy", "TrainAccuracy", "Precision", "Recall", "Parameters"])
        return pd.DataFrame(self.records).sort_values(by="Accuracy", ascending=False)


class AutoMLAgent:
    def __init__(self, executor: DaytonaExecutor):
        self.executor = executor
        self.leaderboard = Leaderboard()
        self.history_file = "history.json"
        self.history = self.load_history()
        self._llm_cache = {}  # Cache LLM responses to avoid repeated calls

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

    def get_warm_start_recommendations(self, summary):
        """
        Phase 5: Meta-learning warm start.
        Looks at history to find similar datasets (by size/columns) and recommends models.
        """
        rows = summary.get('rows', 0)
        cols = summary.get('columns', 0)
        
        # Simple similarity: closest by rows/cols
        best_match = None
        min_dist = float('inf')
        
        for run in self.history:
            dist = abs(run['rows'] - rows) + abs(run['cols'] - cols)
            if dist < min_dist:
                min_dist = dist
                best_match = run
        
        if best_match:
            return f"Based on similar dataset ({best_match['rows']} rows), {best_match['best_model']} performed best."
        return ""

    def call_llm_with_retry(self, prompt, retries=5, delay=20, use_cache=True):
        """
        Calls OpenRouter API with retry logic and exponential backoff for rate limits.
        Includes caching to avoid redundant API calls.
        """
        # Check cache first
        import hashlib
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
        
        # Use a model with better rate limits - Llama 3.1 8B has higher free tier limits
        model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
        
        data = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        for attempt in range(retries):
            try:
                response = requests.post(url, headers=headers, json=data, timeout=120)
                
                # Check for rate limit specifically
                if response.status_code == 429:
                    wait_time = delay * (2 ** attempt) + random.uniform(1, 5)
                    print(f"Rate limit (429). Waiting {wait_time:.1f}s before retry {attempt+1}/{retries}...")
                    time.sleep(wait_time)
                    continue
                    
                response.raise_for_status()
                result = response.json()['choices'][0]['message']['content']
                
                # Cache the result
                if use_cache:
                    self._llm_cache[cache_key] = result
                    
                return result
            except requests.exceptions.HTTPError as e:
                if "429" in str(e) and attempt < retries - 1:
                    wait_time = delay * (2 ** attempt) + random.uniform(1, 5)
                    print(f"Rate limit hit. Waiting {wait_time:.1f}s before retry {attempt+1}/{retries}...")
                    time.sleep(wait_time)
                elif attempt < retries - 1:
                    wait_time = delay + random.uniform(0, 2)
                    print(f"API error: {e}. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    raise e
            except Exception as e:
                if attempt < retries - 1:
                    wait_time = delay + random.uniform(0, 2)
                    print(f"Error: {e}. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    raise e
        
        raise Exception(f"Max retries ({retries}) exceeded for OpenRouter API")

    def analyze_and_plan(self, data_summary):
        """
        Phase 2: LLM analyzes data and proposes a plan.
        """
        warm_start_info = self.get_warm_start_recommendations(data_summary)
        
        prompt = f"""
        You are an expert Data Scientist. Here is the summary of a dataset:
        {json.dumps(data_summary, default=str)}
        
        {warm_start_info}

        Please propose a machine learning plan to classify the target variable (assume the last column is target if not specified, or ask).
        
        Step 1: Suggest a Preprocessing Strategy (handling missing values, encoding categorical variables, scaling).
        Step 2: Suggest a Feature Engineering Strategy (creating new features, interaction terms, transformations).
        Step 3: Suggest a Feature Selection Strategy (e.g., SelectKBest, RFE, PCA, SelectFromModel) to select the most relevant features.
        Step 4: Suggest 3 different models to try from: Logistic Regression, Random Forest, Gradient Boosting, MLP, kNN.
        For each model, suggest 2-3 hyperparameters to tune, specifically including regularization parameters (e.g., C, max_depth, min_samples_leaf, alpha) to prevent overfitting.
        
        Output format (JSON):
        {{
            "plan_overview": "Brief explanation...",
            "target_column": "inferred_target_col",
            "preprocessing_strategy": "Description...",
            "feature_engineering_strategy": "Description...",
            "feature_selection_strategy": "Description...",
            "models": [
                {{"name": "ModelName", "params": {{"param_name": [value1, value2]}}}}
            ]
        }}
        """
        try:
            text = self.call_llm_with_retry(prompt)
            
            # Clean up potential markdown formatting
            if text.startswith("```json"):
                text = text.split("```json")[1].split("```")[0]
            elif text.startswith("```"):
                text = text.split("```")[1].split("```")[0]
            
            text = text.strip()
                
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse LLM response as JSON: {e}")
            print(f"[ERROR] LLM Response was:\n{text}")
            return {"error": f"LLM returned invalid JSON: {str(e)}", "raw_response": text[:500]}
        except Exception as e:
            return {"error": str(e)}

    def generate_preprocessing_code(self, data_summary, target_col, prep_strategy, eng_strategy):
        """
        Generates code to preprocess data and save artifacts. Includes Feature Engineering.
        """
        prompt = f"""
        Write a complete Python script to preprocess 'dataset.csv'.
        Target Column: '{target_col}'
        Preprocessing Strategy: {prep_strategy}
        Feature Engineering Strategy: {eng_strategy}
        
        Requirements:
        1. Import os and pandas. 
        2. Check if 'dataset.csv' exists. If not, print("ERROR: dataset.csv not found") and exit(1).
        3. Load 'dataset.csv'.
        4. Separate features (X) and target (y).
        5. Feature Engineering: Apply transformations (e.g., new columns) on X. 
           (Ensure these are robust. If using row-independent math, apply to X. If using stats, fits on train).
        6. Split into Train (80%) and Test (20%) sets using sklearn.model_selection.train_test_split(random_state=42).
        7. Create a sklearn ColumnTransformer/Pipeline ('preprocessor') to:
           - Handle missing values (SimpleImputer).
           - Encode Categorical columns {data_summary['categorical_columns']} (OneHotEncoder or LabelEncoder).
           - Scale Numerical columns (StandardScaler) if beneficial.
        8. Fit the preprocessor on X_train.
        9. Transform X_train and X_test.
        10. CRITICAL: Save the processed data to CSV files in the CURRENT directory (./).
           - If X_train_processed is a numpy array (e.g. from StandardScaler), convert it to DataFrame: 
             pd.DataFrame(X_train_processed).to_csv('X_train_processed.csv', index=False)
           - Do the same for 'X_test_processed.csv'.
           - Save 'y_train.csv' and 'y_test.csv'.
           - Save 'preprocessor.pkl' (The fitted ColumnTransformer object).
        11. Print JSON summary of processed data:
           {{"status": "complete", "n_features": X_train_processed.shape[1]}}
        12. Verify files exist: import os; print("Saved files in " + os.getcwd() + ":", os.listdir('.'))
        """
        
        try:
            code = self.call_llm_with_retry(prompt)
            if code.startswith("```python"):
                code = code.split("```python")[1].split("```")[0]
            elif code.startswith("```"):
                code = code.split("```")[1].split("```")[0]
            return code.strip()
        except Exception as e:
            error_msg = str(e).replace("'", "").replace('"', '').replace('\n', ' ')
            return f"print('Error generating preprocessing code: {error_msg}')"

    def generate_feature_selection_code(self, target_col, strategy):
        """
        Generates code to select features.
        """
        prompt = f"""
        Write a Python script to perform Feature Selection on processed data.
        Strategy: {strategy}
        
        Requirements:
        1. Load 'X_train_processed.csv', 'y_train.csv', 'X_test_processed.csv'.
        2. Initialize a Feature Selector (e.g., SelectKBest, RFE, SelectFromModel).
        3. Fit the selector on (X_train_processed, y_train).
        4. Transform X_train_processed and X_test_processed to keep only selected features.
        5. Save:
           - 'X_train_selected.csv'
           - 'X_test_selected.csv'
           - 'selector.pkl' (The fitted selector object)
        6. Print JSON summary:
           {{"status": "complete", "original_features": N, "selected_features": M, "kept_indices": [...]}}
        """
        
        try:
            code = self.call_llm_with_retry(prompt)
            if code.startswith("```python"):
                code = code.split("```python")[1].split("```")[0]
            elif code.startswith("```"):
                code = code.split("```")[1].split("```")[0]
            return code.strip()
        except Exception as e:
            error_msg = str(e).replace("'", "").replace('"', '').replace('\n', ' ')
            return f"print('Error generating feature selection code: {error_msg}')"

    def generate_training_code(self, model_config, target_col, use_selected_features=False):
        """
        Generates Python code to train a specific model using preprocessed (and optionally selected) data.
        """
        data_prefix = "selected" if use_selected_features else "processed"
        
        prompt = f"""
        Write a Python script to train {model_config['name']}.
        Data Source: 'X_train_{data_prefix}.csv', 'X_test_{data_prefix}.csv', 'y_train.csv', 'y_test.csv'.
        
        Steps:
        1. Load data files.
        2. Load 'preprocessor.pkl'.
        3. {'Load "selector.pkl" if it exists.' if use_selected_features else ''}
        4. Initialize {model_config['name']} with param grid: {model_config['params']}.
        5. Use GridSearchCV or RandomizedSearchCV (cv=3) to tune hyperparameters.
        6. Get the best estimator.
        7. Evaluate on Training Set (calculate accuracy).
        8. Evaluate on Test Set (calculate accuracy, precision, recall).
        9. Calculate Confusion Matrix on Test Set.
        10. Create a final Pipeline:
           - Steps: [('preprocessor', preprocessor)]
           - {'If "selector" loaded: steps.append(("selector", selector))' if use_selected_features else ''}
           - steps.append(('model', best_estimator))
           - Pipeline(steps)
        11. Save full pipeline to '{model_config['name'].replace(' ', '_')}_model.pkl'.
        12. Save model only to '{model_config['name'].replace(' ', '_')}_only.pkl'.
        13. PRINT results in JSON format:
        {{
            "model": "{model_config['name']}",
            "accuracy": 0.95,
            "train_accuracy": 0.98,
            "precision": 0.94,
            "recall": 0.93,
            "confusion_matrix": [[10, 2], [3, 15]],
            "best_params": {{...}},
            "feature_importance": {{...}}
        }}
        """
        
        try:
            code = self.call_llm_with_retry(prompt)
            
            if code.startswith("```python"):
                code = code.split("```python")[1].split("```")[0]
            elif code.startswith("```"):
                code = code.split("```")[1].split("```")[0]
            return code.strip()
        except Exception as e:
            error_msg = str(e).replace("'", "").replace('"', '').replace('\n', ' ')
            return f"print('Error generating code: {error_msg}')"
    
    def generate_fix_overfitting_code(self, model_name, current_params, train_acc, test_acc, target_col, use_selected_features=False):
        """
        Generates code to retrain a model with stronger regularization to fix overfitting.
        """
        data_prefix = "selected" if use_selected_features else "processed"
        
        prompt = f"""
        The model '{model_name}' is suffering from Overfitting.
        Current Performance -> Train Accuracy: {train_acc}, Test Accuracy: {test_acc}.
        Current Params: {current_params}
        
        Task: Write a Python script to retrain '{model_name}' with STRONGER Regularization to close the generalization gap.
        
        Strategies to apply based on model type:
        - Random Forest/Gradient Boosting: Reduce `max_depth`, increase `min_samples_leaf`, increase `min_samples_split`.
        - Logistic Regression/SVM: Decrease `C` (inverse regularization strength) or increase `alpha`.
        - MLP: Add/Increase Dropout, reduce hidden layer sizes, increase L2 penalty (`alpha`).
        - kNN: Increase `n_neighbors` (to smooth decision boundaries).
        
        Steps:
        1. Load data: 'X_train_{data_prefix}.csv', 'y_train.csv', 'X_test_{data_prefix}.csv', 'y_test.csv'.
        2. Load 'preprocessor.pkl'.
        3. {'Load "selector.pkl" if it exists.' if use_selected_features else ''}
        4. Initialize {model_name}.
        5. Define a new parameter grid that explores strictly MORE regularized values than current params.
        6. Use GridSearchCV (cv=3) to find the best regularized model.
        7. Evaluate on Train and Test sets.
        8. Calculate Confusion Matrix.
        9. Create final Pipeline (preprocessor + {'selector + ' if use_selected_features else ''} model).
        10. Save pipeline to '{model_name.replace(' ', '_')}_Fixed_model.pkl'.
        11. Save model only to '{model_name.replace(' ', '_')}_Fixed_only.pkl'.
        12. PRINT results in JSON format:
        {{
            "model": "{model_name} (Regularized)",
            "accuracy": 0.85,
            "train_accuracy": 0.86,
            "precision": 0.84,
            "recall": 0.83,
            "best_params": {{...}},
            "confusion_matrix": [[...]]
        }}
        """
        
        try:
            code = self.call_llm_with_retry(prompt)
            
            if code.startswith("```python"):
                code = code.split("```python")[1].split("```")[0]
            elif code.startswith("```"):
                code = code.split("```")[1].split("```")[0]
            return code.strip()
        except Exception as e:
            error_msg = str(e).replace("'", "").replace('"', '').replace('\n', ' ')
            return f"print('Error generating fix code: {error_msg}')"

    def generate_ensemble_code(self, top_models, target_col, use_selected_features=False):
        """
        Phase 5: Ensembling.
        """
        data_prefix = "selected" if use_selected_features else "processed"
        model_files = [f"{m['Model'].replace(' ', '_')}_only.pkl" for m in top_models]
        
        prompt = f"""
        Write a script to create a VotingClassifier ensemble.
        Data: 'X_train_{data_prefix}.csv', 'y_train.csv', 'X_test_{data_prefix}.csv', 'y_test.csv'.
        
        Steps:
        1. Load data.
        2. Load 'preprocessor.pkl'.
        3. {'Load "selector.pkl" if available.' if use_selected_features else ''}
        4. Load trained models: {model_files}.
        5. Create VotingClassifier(estimators=[...], voting='soft').
        6. Fit on Train data.
        7. Evaluate on Train data (accuracy).
        8. Evaluate on Test data (accuracy).
        9. Calculate Confusion Matrix on Test data.
        10. Create Full Pipeline (preprocessor + { 'selector + ' if use_selected_features else ''} ensemble).
        11. Save to 'final_ensemble.pkl'.
        12. Print JSON metrics: {{"model": "Ensemble", "accuracy": ..., "train_accuracy": ..., "confusion_matrix": ...}}
        """
        
        try:
            code = self.call_llm_with_retry(prompt)
            
            if code.startswith("```python"):
                code = code.split("```python")[1].split("```")[0]
            elif code.startswith("```"):
                code = code.split("```")[1].split("```")[0]
            return code.strip()
        except Exception as e:
            error_msg = str(e).replace("'", "").replace('"', '').replace('\n', ' ')
            return f"print('Error generating ensemble code: {error_msg}')"
    
    def generate_api_code(self, model_filename='final_ensemble.pkl'):
        """
        Phase 5: Export. Generates a FastAPI app code.
        """
        return f"""
from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import uvicorn
from pydantic import BaseModel

app = FastAPI(title="Autonomous ML Model API")

# Load Model
try:
    model = joblib.load("{model_filename}")
except:
    model = None

class InferenceRequest(BaseModel):
    data: list  # List of dicts or values

@app.post("/predict")
def predict(req: InferenceRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        df = pd.DataFrame(req.data)
        # Note: Preprocessing must match training. 
        # Ideally the pickle includes the full pipeline.
        predictions = model.predict(df)
        return {{"predictions": predictions.tolist()}}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

    def extract_json(self, text):
        """
        Robustly extracts JSON object from a text that might contain other logs.
        """
        try:
            # Search for pattern {"model": ... } which we enforce in prompts
            match = re.search(r'\{.*"model":.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError as e:
                    print(f"[WARN] JSON parsing failed on matched text: {e}")
                    print(f"[WARN] Matched text: {match.group(0)[:200]}...")
            
            # Fallback: find last bracket pair
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end > start:
                json_candidate = text[start:end]
                try:
                    return json.loads(json_candidate)
                except json.JSONDecodeError as e:
                    print(f"[WARN] JSON parsing failed on bracket extraction: {e}")
                    print(f"[WARN] Extracted text: {json_candidate[:200]}...")
                    
            print(f"[ERROR] No valid JSON found in text: {text[:300]}...")
            return None
        except Exception as e:
            print(f"[ERROR] extract_json failed with exception: {e}")
            return None

    def get_data_summary_from_sandbox(self, filename='X_train_processed.csv'):
        """
        Executes a script in the sandbox to get a summary of the processed/selected data.
        """
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
        files = os.listdir('.')
        print(json.dumps({{"error": f"File '{filename}' not found. Available files: {{files}}"}}))
    else:
        # Try loading with header, if fails or looks weird (all strings), might be numpy save
        df = pd.read_csv('{filename}')
        
        description = df.describe().to_dict()
        
        info = {{
            "rows": df.shape[0],
            "columns": df.shape[1],
            "columns_list": list(df.columns[:20]), # Limit for UI
            "description": description,
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
                return {"error": f"Failed to parse summary JSON: {res['stdout']}"}
        return {"error": res['stderr']}

    def run_experiment(self, df, progress_callback=None):
        """
        Legacy method: Runs the whole pipeline at once.
        """
        pass
