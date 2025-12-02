import os
import json
import base64
import pandas as pd
from dotenv import load_dotenv
from daytona import Daytona, DaytonaConfig
import google.generativeai as genai
import re
import certifi

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
            return True, "Sandbox created successfully."
        except Exception as e:
            return False, f"Failed to create sandbox: {str(e)}"

    def install_dependencies(self, packages):
        if not self.sandbox:
            return False, "Sandbox not initialized."
        
        # Only install if necessary.
        install_cmd = f"pip install {' '.join(packages)}"
        return self.execute_command(install_cmd)

    def upload_data(self, df, filename='dataset.csv'):
        if not self.sandbox:
             return False, "Sandbox not initialized."
        
        csv_data = df.to_csv(index=False)
        import base64
        encoded_data = base64.b64encode(csv_data.encode('utf-8')).decode('utf-8')
        
        write_script = f"""
import base64
data_b64 = "{encoded_data}"
with open('{filename}', 'wb') as f:
    f.write(base64.b64decode(data_b64))
print(f"Data written to {filename}")
"""
        return self.execute_code(write_script)

    def execute_command(self, command):
        if not self.sandbox:
            self.create_sandbox()
            
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
        if not self.sandbox:
             self.create_sandbox()

        try:
            response = self.sandbox.process.code_run(code_string)
            result = {
                "exit_code": response.exit_code,
                "stdout": response.result,
                "stderr": ""
            }
            if response.exit_code != 0:
                result["stderr"] = f"Error (Exit Code {response.exit_code}): {response.result}"
            return result
        except Exception as e:
            return {"exit_code": -1, "stdout": "", "stderr": str(e)}

    def download_file(self, remote_path):
        """
        Reads a file from the sandbox and returns its content.
        """
        if not self.sandbox:
            return None
        
        # Using python to cat the file content back
        read_script = f"""
import sys
import base64
try:
    with open('{remote_path}', 'rb') as f:
        print(base64.b64encode(f.read()).decode('utf-8'))
except Exception as e:
    print("FILE_NOT_FOUND")
"""
        res = self.execute_code(read_script)
        if res['exit_code'] == 0:
            try:
                output = res['stdout'].strip()
                if "FILE_NOT_FOUND" in output:
                    return None
                return base64.b64decode(output)
            except:
                return None
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

    def add_entry(self, model_name, accuracy, precision, recall, params, feature_importance=None):
        self.records.append({
            "Model": model_name,
            "Accuracy": round(float(accuracy), 4),
            "Precision": round(float(precision), 4),
            "Recall": round(float(recall), 4),
            "Parameters": str(params),
            "FeatureImportance": feature_importance
        })

    def get_dataframe(self):
        if not self.records:
            return pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "Parameters"])
        return pd.DataFrame(self.records).sort_values(by="Accuracy", ascending=False)


class AutoMLAgent:
    def __init__(self, executor: DaytonaExecutor):
        self.executor = executor
        self.leaderboard = Leaderboard()
        self.history_file = "history.json"
        self.history = self.load_history()

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
        Suggest 3 different models to try from: Logistic Regression, Random Forest, Gradient Boosting, MLP, kNN.
        For each model, suggest a key hyperparameter to tune.
        
        Output format (JSON):
        {{
            "plan_overview": "Brief explanation...",
            "target_column": "inferred_target_col",
            "models": [
                {{"name": "ModelName", "params": {{"param_name": [value1, value2]}}}}
            ]
        }}
        """
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key: return {"error": "Missing GEMINI_API_KEY"}
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            
            text = response.text
            # Clean up potential markdown formatting from Gemini
            if text.startswith("```json"):
                text = text.split("```json")[1].split("```")[0]
            elif text.startswith("```"):
                text = text.split("```")[1].split("```")[0]
                
            return json.loads(text)
        except Exception as e:
            return {"error": str(e)}

    def generate_code(self, model_config, data_summary, target_col):
        """
        Generates Python code to train a specific model.
        """
        prompt = f"""
        Write a complete Python script to:
        1. Read 'dataset.csv'.
        2. Preprocess data (handle missing values, encode '{data_summary['categorical_columns']}' using LabelEncoder or OneHotEncoder).
        3. Split into train/test (80/20).
        4. Train a {model_config['name']} with these hyperparameter options: {model_config['params']}.
        5. Use GridSearchCV or RandomizedSearchCV to find best params.
        6. Evaluate on test set.
        7. Calculate Feature Importance (if applicable) as a dictionary {{feature: score}}.
        8. Save the best model to a file named '{model_config['name'].replace(' ', '_')}_model.pkl'.
        9. PRINT the results in this exact JSON format at the end (CRITICAL: PRINT ONLY THE JSON):
        {{
            "model": "{model_config['name']}",
            "accuracy": 0.95,
            "precision": 0.94,
            "recall": 0.93,
            "best_params": {{...}},
            "feature_importance": {{ "col1": 0.5, "col2": 0.3 ... }}
        }}
        
        Ensure you import all necessary libraries (pandas, sklearn, numpy, json, pickle/joblib).
        """
        
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-3-pro-preview")
            response = model.generate_content(prompt)
            
            code = response.text
            if code.startswith("```python"):
                code = code.split("```python")[1].split("```")[0]
            elif code.startswith("```"):
                code = code.split("```")[1].split("```")[0]
            return code.strip()
        except Exception as e:
            return f"print('Error generating code: {str(e)}')"
    
    def generate_ensemble_code(self, top_models, target_col):
        """
        Phase 5: Ensembling.
        """
        model_names = [m['Model'] for m in top_models]
        files = [f"{m['Model'].replace(' ', '_')}_model.pkl" for m in top_models]
        
        prompt = f"""
        Write a script to create a VotingClassifier ensemble from these saved models: {files}.
        The dataset is 'dataset.csv', target is '{target_col}'.
        1. Load models from pickle files.
        2. Create VotingClassifier(estimators=[...], voting='soft').
        3. Train/Evaluate on 'dataset.csv' (split 80/20).
        4. Save ensemble to 'final_ensemble.pkl'.
        5. Print JSON metrics: {{"model": "Ensemble", "accuracy": ...}}
        """
        
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-3-pro-preview")
            response = model.generate_content(prompt)
            
            code = response.text
            if code.startswith("```python"):
                code = code.split("```python")[1].split("```")[0]
            elif code.startswith("```"):
                code = code.split("```")[1].split("```")[0]
            return code.strip()
        except Exception as e:
            return f"print('Error generating ensemble code: {str(e)}')"
    
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
            # Attempt to find the outermost JSON object
            # Regex looks for { ... } allowing for nested braces roughly
            # A simpler approach is to find the last occurrence of '}' and the first '{'
            
            # Search for pattern {"model": ... } which we enforce in prompts
            match = re.search(r'\{.*"model":.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            
            # Fallback: find last bracket pair
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end != -1:
                return json.loads(text[start:end])
            return None
        except:
            return None

    def run_experiment(self, df, progress_callback=None):
        logs = []
        
        def log(msg, type="info"):
            logs.append({"msg": msg, "type": type})
            if progress_callback: progress_callback(msg)

        log("Initializing Sandbox...")
        success, msg = self.executor.create_sandbox()
        if not success: 
            log(f"Failed to create sandbox: {msg}", "error")
            return logs
        
        log("Installing Dependencies (pandas, sklearn)...")
        # Only essential packages to save time/errors
        self.executor.install_dependencies(['pandas', 'scikit-learn', 'numpy', 'joblib'])
        
        log("Uploading Data...")
        self.executor.upload_data(df, 'dataset.csv')
        
        # Analyze
        summary = {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "cols_list": list(df.columns),
            "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist()
        }
        
        log("Planning Experiments...")
        plan = self.analyze_and_plan(summary)
        if "error" in plan: 
            log(f"LLM Error: {plan['error']}", "error")
            return logs
            
        target_col = plan.get('target_column', df.columns[-1])
        
        # Run Models
        for model in plan.get('models', []):
            log(f"Training {model['name']}...")
            code = self.generate_code(model, summary, target_col)
            
            # Execute
            result = self.executor.execute_code(code)
            
            # Store raw output in logs for debugging
            logs.append({"msg": f"--- Executing {model['name']} ---", "type": "debug", "details": result})

            if result['exit_code'] == 0:
                metrics = self.extract_json(result['stdout'])
                if metrics:
                    self.leaderboard.add_entry(
                        metrics['model'], metrics.get('accuracy', 0),
                        metrics.get('precision', 0), metrics.get('recall', 0),
                        metrics.get('best_params', {}),
                        metrics.get('feature_importance', None)
                    )
                    log(f"✅ {model['name']} Acc: {metrics.get('accuracy')}")
                else:
                    # If extraction failed, show part of stdout to help user debug
                    log(f"⚠️ {model['name']} finished but JSON missing.", "warning")
            else:
                log(f"❌ {model['name']} failed.", "error")
                logs.append({"msg": f"Error details for {model['name']}", "type": "error", "details": result['stderr']})

        # Phase 5: Ensemble
        if len(self.leaderboard.records) >= 2:
            log("Building Ensemble...")
            top_models = sorted(self.leaderboard.records, key=lambda x: x['Accuracy'], reverse=True)[:3]
            ensemble_code = self.generate_ensemble_code(top_models, target_col)
            ens_res = self.executor.execute_code(ensemble_code)
            
            logs.append({"msg": "--- Executing Ensemble ---", "type": "debug", "details": ens_res})
            
            if ens_res['exit_code'] == 0:
                log("✅ Ensemble created.")
                # Update history
                best_acc = top_models[0]['Accuracy']
                self.save_history({
                    "rows": df.shape[0], "cols": df.shape[1], 
                    "best_model": top_models[0]['Model'], "accuracy": best_acc
                })
            else:
                log(f"❌ Ensemble failed: {ens_res['stderr'][:200]}", "error")

        log("Experiment Completed.")
        return logs
