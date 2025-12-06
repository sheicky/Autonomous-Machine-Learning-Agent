# 🔄 Migration Guide: Original → Enhanced Version

This guide helps you transition from the original AutoML agent to the enhanced version.

## Quick Migration (5 minutes)

### Option 1: Run Enhanced Version Alongside Original

```bash
# Keep using original
streamlit run app/app.py --server.port 8501

# Try enhanced version
streamlit run app/app_enhanced.py --server.port 8502
```

Both versions can run simultaneously on different ports.

### Option 2: Switch to Enhanced Version

```bash
# Backup original (optional)
cp app/app.py app/app_original_backup.py

# Use enhanced version as default
streamlit run app/app_enhanced.py
```

## What's Different?

### User Interface Changes

| Feature | Original | Enhanced |
|---------|----------|----------|
| Tabs | 4 tabs | 6 tabs (added Quality Analysis & Experiments) |
| Progress Indicator | None | Visual pipeline stages |
| Cache Management | Hidden | Visible in sidebar |
| Model Comparison | Basic table | Interactive charts |
| Deployment | Code only | Code + Docker + K8s |

### Workflow Changes

**Original Workflow:**
1. Upload data
2. Generate plan
3. Preprocess
4. Train models (sequential)
5. Download model

**Enhanced Workflow:**
1. Upload data
2. **NEW:** Run quality analysis
3. Generate plan (with quality insights)
4. Preprocess (with caching)
5. **Optional:** Feature selection
6. Train models (parallel)
7. **NEW:** Auto-fix overfitting
8. Download model + deployment files

## Feature Mapping

### Data Quality Analysis (NEW)

```python
# Original: No automated checks
# You had to manually inspect data

# Enhanced: Automated analysis
from src.data_quality import DataQualityAnalyzer

analyzer = DataQualityAnalyzer(df, target_column='target')
report = analyzer.get_report()

# Get issues and recommendations
print(f"Issues: {report['summary']['total_issues']}")
print(f"Status: {report['summary']['status']}")
```

### Hyperparameter Tuning

```python
# Original: GridSearchCV
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(model, param_grid, cv=3)
grid_search.fit(X_train, y_train)

# Enhanced: Optuna (automatically used)
# No code changes needed - it's built into the agent
# Just click "Start Parallel Training"
```

### Caching

```python
# Original: Only LLM responses cached
# No preprocessing cache

# Enhanced: Full pipeline caching
from src.cache_manager import CacheManager

cache = CacheManager()
cache_key = cache._generate_key(df, params)

if cache.has(cache_key, "preprocessing"):
    result = cache.get(cache_key, "preprocessing")
else:
    result = preprocess(df)
    cache.set(cache_key, result, "preprocessing")
```

### Experiment Tracking

```python
# Original: Basic history.json file
# Limited tracking

# Enhanced: Full experiment tracking
from src.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker()
exp_id = tracker.start_experiment(name="My Experiment")

# Log stages
tracker.log_stage("preprocessing", data)
tracker.log_model("Random Forest", metrics, params)

# End and save
tracker.end_experiment()

# Compare experiments
experiments = tracker.list_experiments()
leaderboard = tracker.get_leaderboard()
```

### Parallel Training

```python
# Original: Sequential training
for model in models:
    train_model(model)  # One at a time

# Enhanced: Parallel training
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(train_model, m) for m in models]
    results = [f.result() for f in futures]
```

## API Changes

### Brain/Agent Class

```python
# Original
from brain import AutoMLAgent, DaytonaExecutor

executor = DaytonaExecutor()
agent = AutoMLAgent(executor)

# Enhanced (backward compatible)
from brain_enhanced import EnhancedAutoMLAgent, DaytonaExecutor

executor = DaytonaExecutor()
agent = EnhancedAutoMLAgent(executor)

# New methods available:
# - train_models_parallel()
# - generate_training_code_optuna()
# - generate_docker_code()
```

### Method Signatures

Most methods are backward compatible. New optional parameters:

```python
# analyze_and_plan()
# Original
plan = agent.analyze_and_plan(data_summary)

# Enhanced (quality_report is optional)
plan = agent.analyze_and_plan(data_summary, quality_report=None)

# generate_preprocessing_code()
# Original
code = agent.generate_preprocessing_code(summary, target, prep_strategy, eng_strategy)

# Enhanced (quality_report is optional)
code = agent.generate_preprocessing_code(
    summary, target, prep_strategy, eng_strategy, quality_report=None
)
```

## Configuration Changes

### Environment Variables

```bash
# Original .env
DAYTONA_API_KEY=xxx
OPENROUTER_API_KEY=xxx

# Enhanced .env (same, no changes needed)
DAYTONA_API_KEY=xxx
OPENROUTER_API_KEY=xxx
OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free  # Optional
```

### Dependencies

```bash
# Original requirements.txt
streamlit
pandas
scikit-learn
optuna  # Listed but not used

# Enhanced requirements.txt (additional packages)
streamlit
pandas
scikit-learn
optuna  # Now actively used!
shap  # NEW: For explainability
imbalanced-learn  # NEW: For SMOTE
xgboost  # NEW: Additional model
lightgbm  # NEW: Additional model
catboost  # NEW: Additional model
```

Install new dependencies:

```bash
pip install shap imbalanced-learn xgboost lightgbm catboost
```

## Data Migration

### Experiment History

```python
# Original: history.json
# Format: Simple list of experiments

# Enhanced: experiments/ directory
# Format: One JSON file per experiment with full details

# Migration script (if needed):
import json
import os
from src.experiment_tracker import ExperimentTracker

# Load old history
with open('history.json', 'r') as f:
    old_history = json.load(f)

# Convert to new format
tracker = ExperimentTracker()
for old_exp in old_history:
    exp_id = tracker.start_experiment(
        name=old_exp.get('name', 'Migrated'),
        dataset_info=old_exp.get('dataset_info', {})
    )
    # Add any available data
    if 'best_model' in old_exp:
        tracker.log_model(
            old_exp['best_model'],
            {'accuracy': old_exp.get('accuracy', 0)},
            {}
        )
    tracker.end_experiment()
```

### Cache Migration

```python
# Original: No cache files
# Enhanced: .cache/ directory

# No migration needed - cache builds automatically
```

## Troubleshooting Migration Issues

### Issue: Import errors

```bash
# Error: ModuleNotFoundError: No module named 'shap'
# Solution:
pip install -r requirements.txt
```

### Issue: Old experiments not showing

```bash
# Solution: Run migration script above
# Or start fresh (old history.json is preserved)
```

### Issue: Performance seems slower

```bash
# Possible causes:
# 1. First run (building cache) - subsequent runs will be faster
# 2. More comprehensive analysis - but provides better results
# 3. Check if parallel training is enabled

# Solution: Check settings in sidebar
```

### Issue: Different results than original

```bash
# This is expected! Enhanced version:
# - Uses Optuna (different optimization)
# - Applies quality fixes automatically
# - Uses stronger regularization
# - Results should be BETTER, not identical
```

## Gradual Migration Strategy

### Phase 1: Testing (Week 1)
- Run enhanced version on test datasets
- Compare results with original
- Familiarize team with new features

### Phase 2: Parallel Usage (Week 2-3)
- Use enhanced for new projects
- Keep original for ongoing projects
- Train team on new features

### Phase 3: Full Migration (Week 4)
- Switch all projects to enhanced
- Archive original version
- Update documentation

## Rollback Plan

If you need to rollback:

```bash
# 1. Stop enhanced version
# 2. Restart original version
streamlit run app/app.py

# 3. Your data is safe:
#    - Original history.json is untouched
#    - Enhanced uses separate experiments/ directory
#    - Cache is separate (.cache/ directory)
```

## Getting Help

### Common Questions

**Q: Can I use both versions?**
A: Yes! They don't interfere with each other.

**Q: Will my old models work?**
A: Yes! Model files (.pkl) are compatible.

**Q: Do I need to retrain everything?**
A: No, but retraining with enhanced version may improve results.

**Q: What if I don't want feature X?**
A: Most features are optional. You can skip quality analysis, feature selection, etc.

**Q: Is the enhanced version slower?**
A: First run may be slower (more analysis), but subsequent runs are faster (caching).

## Next Steps

1. ✅ Install new dependencies
2. ✅ Try enhanced version on sample dataset
3. ✅ Compare results with original
4. ✅ Read QUICKSTART.md for detailed usage
5. ✅ Explore new features (quality analysis, experiments tab)
6. ✅ Deploy with new Docker/K8s templates

## Support

- Check [README.md](README.md) for full documentation
- See [QUICKSTART.md](QUICKSTART.md) for usage examples
- Run [benchmark_comparison.py](benchmark_comparison.py) to see improvements
- Open GitHub issue for problems

---

**Happy migrating! 🚀**
