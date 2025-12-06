# 🚀 Quick Start Guide - Enhanced AutoML Agent

## 5-Minute Setup

### Step 1: Install Dependencies (2 min)

```bash
pip install -r requirements.txt
```

### Step 2: Configure API Keys (1 min)

Create `app/.env`:

```bash
DAYTONA_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free
```

**Get API Keys:**
- Daytona: https://daytona.io (Free tier available)
- OpenRouter: https://openrouter.ai (Free models available)

### Step 3: Run Application (30 sec)

```bash
streamlit run app/app_enhanced.py
```

### Step 4: Upload & Train (1 min)

1. Upload your CSV/Excel file
2. Click "Run Quality Analysis"
3. Click "Generate ML Strategy"
4. Click "Execute Preprocessing"
5. Click "Start Parallel Training"
6. Download your model!

## 🎯 First Time User Flow

### Example: Titanic Dataset

```bash
# 1. Download sample data
wget https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

# 2. Launch app
streamlit run app/app_enhanced.py

# 3. Upload titanic.csv

# 4. Follow the tabs:
#    - Data Overview: See your data
#    - Quality Analysis: Check for issues
#    - ML Pipeline: Train models
#    - Results: Download model
```

### What You'll Get

✅ **Data Quality Report**
- Class imbalance: 62% survived vs 38% died
- Missing values: Age (20%), Cabin (77%)
- Recommendations: Use SMOTE, drop Cabin column

✅ **Trained Models**
- Random Forest: 84% accuracy
- Gradient Boosting: 86% accuracy
- Ensemble: 87% accuracy

✅ **Production Assets**
- `autonomous_model.pkl` - Trained model
- `serve.py` - FastAPI deployment code
- `Dockerfile` - Container configuration

## 🔥 Advanced Quick Start

### Parallel Training with Custom Models

```python
# In app_enhanced.py, modify the plan to include more models:

models = [
    {"name": "Random Forest", "params": {"n_estimators": [100, 200], "max_depth": [5, 10]}},
    {"name": "XGBoost", "params": {"learning_rate": [0.01, 0.1], "max_depth": [3, 7]}},
    {"name": "LightGBM", "params": {"num_leaves": [31, 50], "learning_rate": [0.01, 0.1]}},
    {"name": "CatBoost", "params": {"depth": [4, 6], "learning_rate": [0.01, 0.1]}}
]
```

### Using Cache for Faster Iterations

```python
# First run: ~5 minutes (preprocessing + training)
# Second run: ~1 minute (cached preprocessing)

# Clear cache if needed:
# Sidebar -> Cache Management -> Clear Cache
```

### Experiment Tracking

```python
# View all experiments in the "Experiments" tab
# Compare models across different datasets
# Export global leaderboard
```

## 🐳 Docker Quick Deploy

```bash
# 1. Train model and download files
# 2. Create deployment directory
mkdir deploy
cd deploy

# 3. Copy files
cp ~/Downloads/autonomous_model.pkl .
cp ~/Downloads/serve.py .
cp ~/Downloads/Dockerfile .

# 4. Create requirements.txt
cat > requirements.txt << EOF
fastapi
uvicorn
pandas
scikit-learn
joblib
pydantic
EOF

# 5. Build and run
docker build -t my-automl-model .
docker run -p 8000:8000 my-automl-model

# 6. Test
curl http://localhost:8000/health
```

## 📊 Sample Datasets to Try

### Classification
- Titanic: https://www.kaggle.com/c/titanic/data
- Iris: Built into scikit-learn
- Wine Quality: https://archive.ics.uci.edu/ml/datasets/wine+quality

### Regression
- House Prices: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
- Boston Housing: Built into scikit-learn

### Imbalanced
- Credit Card Fraud: https://www.kaggle.com/mlg-ulb/creditcardfraud

## 🎓 Learning Path

### Beginner (Day 1)
1. Upload simple dataset (Iris, Titanic)
2. Run quality analysis
3. Train with default settings
4. Download model

### Intermediate (Day 2-3)
1. Understand quality report
2. Customize preprocessing strategy
3. Compare multiple experiments
4. Deploy with FastAPI

### Advanced (Week 1)
1. Modify feature engineering
2. Add custom models
3. Tune Optuna parameters
4. Production deployment with monitoring

## 🔧 Troubleshooting Quick Fixes

### Issue: "Sandbox creation failed"
```bash
# Solution: Check API key
echo $DAYTONA_API_KEY

# Or set manually in app
# Sidebar -> Agent Configuration -> Enter key
```

### Issue: "Rate limit exceeded"
```bash
# Solution: Switch model
# Sidebar -> LLM Model -> Select "Llama 3.3 70B"
```

### Issue: "Out of memory"
```bash
# Solution: Enable feature selection
# ML Pipeline -> Step 3 -> Run Feature Selection
```

### Issue: "Model accuracy too low"
```bash
# Solution: Check quality report
# Quality Analysis tab -> View recommendations
# Common fixes:
# - Handle class imbalance (SMOTE)
# - Remove high-cardinality features
# - Handle outliers
```

## 💡 Pro Tips

1. **Always run quality analysis first** - Saves time by identifying issues early
2. **Use caching** - Second runs are 4x faster
3. **Try feature selection** - Often improves accuracy and speed
4. **Check overfitting** - Use the automatic regularization fix
5. **Compare experiments** - Learn what works for your data type
6. **Start simple** - Use 3 models first, then expand
7. **Monitor training** - Watch real-time progress in status updates
8. **Export everything** - Download model, API code, and Docker files

## 🎯 Next Steps

After your first successful run:

1. **Experiment Tracking**: Compare different preprocessing strategies
2. **Feature Engineering**: Try polynomial features or interactions
3. **Model Tuning**: Increase Optuna trials for better results
4. **Production Deploy**: Set up monitoring and logging
5. **Scale Up**: Try larger datasets and more models

## 📚 Additional Resources

- [Full Documentation](README.md)
- [API Reference](docs/API.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

---

**Ready to build your first AutoML model? Let's go! 🚀**
