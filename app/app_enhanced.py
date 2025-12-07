import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_manager import DataManager
from src.data_quality import DataQualityAnalyzer
from src.experiment_tracker import ExperimentTracker
from src.cache_manager import CacheManager
from brain_enhanced import DaytonaExecutor, EnhancedAutoMLAgent

# Configure Streamlit
st.set_page_config(
    page_title="Enhanced Autonomous ML Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .status-good { color: #28a745; }
    .status-warning { color: #ffc107; }
    .status-critical { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🤖 Enhanced Autonomous ML Agent</h1>', unsafe_allow_html=True)
    st.markdown("""
    **Next-Gen AutoML**: Intelligent data analysis → AI-powered planning → Parallel training → 
    Production deployment with monitoring
    """)

    # Initialize session state
    if 'data_manager' not in st.session_state:
        st.session_state.data_manager = DataManager()
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'experiment_tracker' not in st.session_state:
        st.session_state.experiment_tracker = ExperimentTracker()
    if 'cache_manager' not in st.session_state:
        st.session_state.cache_manager = CacheManager()
    
    # Pipeline state
    if 'pipeline_stage' not in st.session_state:
        st.session_state.pipeline_stage = 'init'
    if 'plan' not in st.session_state:
        st.session_state.plan = None
    if 'quality_report' not in st.session_state:
        st.session_state.quality_report = None
    if 'current_experiment_id' not in st.session_state:
        st.session_state.current_experiment_id = None

    # Sidebar
    with st.sidebar:
        st.header("📁 Data Ingestion")
        uploaded_file = st.file_uploader("Upload Dataset", type=['csv', 'xlsx', 'parquet'])
        
        st.divider()
        st.header("⚙️ Agent Configuration")
        
        daytona_key = st.text_input("Daytona API Key", type="password", 
                                    value=os.getenv("DAYTONA_API_KEY", ""))
        openrouter_key = st.text_input("OpenRouter API Key", type="password",
                                       value=os.getenv("OPENROUTER_API_KEY", ""))
        
        model_options = {
            "Claude 4.5" : "anthropic/claude-opus-4.5",
            "Gemini 3 Pro" : "google/gemini-3-pro-preview",
            "Gemini 2.0 Flash": "google/gemini-2.0-flash-exp:free",
            "GPT-5 Codex" : "openai/gpt-5.1-codex-max",
            "CLaude Opus" : "anthropic/claude-opus-4.5",  
        }

        selected_model = st.selectbox("LLM Model", list(model_options.keys()), index=0)
        
        if daytona_key:
            os.environ['DAYTONA_API_KEY'] = daytona_key
        if openrouter_key:
            os.environ['OPENROUTER_API_KEY'] = openrouter_key
        os.environ['OPENROUTER_MODEL'] = model_options[selected_model]
        
        st.divider()
        st.header("🗂️ Cache Management")
        cache_stats = st.session_state.cache_manager.get_stats()
        st.metric("Cached Items", cache_stats.get('total_entries', 0))
        st.metric("Cache Size", f"{cache_stats.get('total_size_mb', 0):.2f} MB")
        
        if st.button("🗑️ Clear Cache"):
            st.session_state.cache_manager.invalidate()
            st.success("Cache cleared!")
            st.rerun()

    if uploaded_file:
        # Load data
        if 'current_file' not in st.session_state or st.session_state.current_file != uploaded_file.name:
            with st.spinner("Loading dataset..."):
                success, message = st.session_state.data_manager.load_data(uploaded_file)
                if success:
                    st.session_state.current_file = uploaded_file.name
                    st.session_state.pipeline_stage = 'uploaded'
                    
                    # Start new experiment
                    df = st.session_state.data_manager.get_dataframe()
                    exp_id = st.session_state.experiment_tracker.start_experiment(
                        name=f"Experiment_{uploaded_file.name}",
                        dataset_info={
                            "filename": uploaded_file.name,
                            "rows": len(df),
                            "columns": len(df.columns)
                        }
                    )
                    st.session_state.current_experiment_id = exp_id
                    st.success(f"✅ {message} | Experiment ID: {exp_id}")
                else:
                    st.error(message)
                    return

        dm = st.session_state.data_manager
        summary = dm.get_summary()
        df = dm.get_dataframe()

        if summary:
            # Create tabs
            tabs = st.tabs([
                "📊 Data Overview",
                "🔍 Quality Analysis",
                "📈 Visualizations",
                "🤖 ML Pipeline",
                "🏆 Results & Export",
                "📚 Experiments"
            ])

            # Tab 1: Data Overview
            with tabs[0]:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Rows", f"{summary['rows']:,}")
                col2.metric("Columns", summary['columns'])
                col3.metric("Missing Values", summary['total_missing'])
                col4.metric("Duplicates", summary['duplicate_rows'])
                
                st.subheader("Dataset Preview")
                st.dataframe(summary['preview'], width='stretch')
                
                st.subheader("Statistical Summary")
                st.dataframe(summary['description'], width='stretch')

            # Tab 2: Quality Analysis
            with tabs[1]:
                st.subheader("🔍 Data Quality Report")
                
                if st.button("🔬 Run Quality Analysis", type="primary"):
                    with st.spinner("Analyzing data quality..."):
                        # Infer target column (last column by default)
                        target_col = df.columns[-1]
                        
                        analyzer = DataQualityAnalyzer(df, target_column=target_col)
                        st.session_state.quality_report = analyzer.get_report()
                        
                        # Log to experiment
                        st.session_state.experiment_tracker.log_stage(
                            "quality_analysis",
                            st.session_state.quality_report
                        )
                        
                        st.success("Quality analysis complete!")
                        st.rerun()
                
                if st.session_state.quality_report:
                    report = st.session_state.quality_report
                    summary_data = report['summary']
                    
                    # Status indicator
                    status = summary_data['status']
                    status_colors = {
                        'good': '🟢',
                        'caution': '🟡',
                        'warning': '🟠',
                        'critical': '🔴'
                    }
                    
                    st.markdown(f"### Overall Status: {status_colors.get(status, '⚪')} {status.upper()}")
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Total Issues", summary_data['total_issues'])
                    col2.metric("Severity Score", f"{summary_data['severity_score']}/100")
                    
                    # Issues
                    if report['issues']:
                        st.subheader("⚠️ Detected Issues")
                        for issue in report['issues']:
                            severity_emoji = {
                                'critical': '🔴',
                                'high': '🟠',
                                'medium': '🟡',
                                'low': '🟢'
                            }
                            
                            with st.expander(
                                f"{severity_emoji.get(issue['severity'], '⚪')} "
                                f"{issue['type'].replace('_', ' ').title()} - "
                                f"{issue['severity'].upper()}"
                            ):
                                st.write(issue['message'])
                                if 'details' in issue:
                                    st.json(issue['details'])
                    
                    # Recommendations
                    if report['recommendations']:
                        st.subheader("💡 Recommendations")
                        for i, rec in enumerate(report['recommendations'], 1):
                            st.info(f"**{i}.** {rec['action']}")
                            if 'code_hint' in rec:
                                st.code(rec['code_hint'], language='python')

            # Tab 3: Visualizations
            with tabs[2]:
                st.subheader("📈 Data Visualizations")
                
                # Correlation heatmap
                if len(summary['numerical_columns']) > 1:
                    st.markdown("#### Correlation Matrix")
                    corr = df[summary['numerical_columns']].corr()
                    fig_corr = px.imshow(
                        corr,
                        text_auto='.2f',
                        aspect="auto",
                        color_continuous_scale='RdBu_r',
                        title="Feature Correlations"
                    )
                    st.plotly_chart(fig_corr, width='stretch')
                
                # Distribution plots
                col1, col2 = st.columns(2)
                
                with col1:
                    if summary['numerical_columns']:
                        selected_num = st.selectbox(
                            "Select Numerical Feature",
                            summary['numerical_columns']
                        )
                        fig_hist = px.histogram(
                            df,
                            x=selected_num,
                            title=f"Distribution of {selected_num}",
                            marginal="box"
                        )
                        st.plotly_chart(fig_hist, width='stretch')
                
                with col2:
                    if summary['categorical_columns']:
                        selected_cat = st.selectbox(
                            "Select Categorical Feature",
                            summary['categorical_columns']
                        )
                        value_counts = df[selected_cat].value_counts().head(10)
                        fig_bar = px.bar(
                            x=value_counts.index,
                            y=value_counts.values,
                            title=f"Top 10 Values in {selected_cat}",
                            labels={'x': selected_cat, 'y': 'Count'}
                        )
                        st.plotly_chart(fig_bar, width='stretch')

            # Tab 4: ML Pipeline
            with tabs[3]:
                render_ml_pipeline(df, summary)

            # Tab 5: Results & Export
            with tabs[4]:
                render_results_export()

            # Tab 6: Experiments
            with tabs[5]:
                render_experiments_tab()

    else:
        st.info("👆 Please upload a dataset to begin your ML journey!")
        
        # Show recent experiments
        experiments = st.session_state.experiment_tracker.list_experiments()
        if experiments:
            st.subheader("📚 Recent Experiments")
            for exp in experiments[:5]:
                with st.expander(f"🔬 {exp['name']} - {exp['timestamp']}"):
                    st.json(exp)



def render_ml_pipeline(df, summary):
    """Render the ML pipeline tab."""
    st.header("🤖 Autonomous ML Pipeline")
    
    # Progress indicator
    stages = ['uploaded', 'planned', 'preprocessed', 'selected', 'trained']
    current_stage_idx = stages.index(st.session_state.pipeline_stage) if st.session_state.pipeline_stage in stages else 0
    
    progress_cols = st.columns(len(stages))
    for idx, (col, stage) in enumerate(zip(progress_cols, stages)):
        if idx < current_stage_idx:
            col.markdown(f"✅ **{stage.title()}**")
        elif idx == current_stage_idx:
            col.markdown(f"🔄 **{stage.title()}**")
        else:
            col.markdown(f"⏳ {stage.title()}")
    
    st.divider()
    
    # Stage 1: Planning
    if st.session_state.pipeline_stage == 'uploaded':
        st.subheader("Step 1: AI Analysis & Planning")
        st.info("The AI will analyze your data quality and create an optimized ML strategy.")
        
        if st.button("🧠 Generate ML Strategy", type="primary", width='stretch'):
            with st.spinner("AI is analyzing your data..."):
                try:
                    # Initialize agent
                    if not st.session_state.agent:
                        executor = DaytonaExecutor()
                        st.session_state.agent = EnhancedAutoMLAgent(executor)
                        
                        # Create sandbox
                        with st.status("Initializing sandbox...") as status:
                            st.session_state.agent.executor.create_sandbox()
                            status.update(label="Installing dependencies...", state="running")
                            # Install core dependencies first
                            st.session_state.agent.executor.install_dependencies([
                                'pandas', 'numpy', 'scikit-learn', 'joblib'
                            ])
                            # Install ML libraries
                            st.session_state.agent.executor.install_dependencies([
                                'imbalanced-learn', 'optuna'
                            ])
                            # Install optional libraries (may fail, that's ok)
                            try:
                                st.session_state.agent.executor.install_dependencies([
                                    'shap', 'xgboost', 'lightgbm'
                                ])
                            except:
                                pass  # Optional packages
                            status.update(label="Uploading data...", state="running")
                            st.session_state.agent.executor.upload_data(df, 'dataset.csv')
                            status.update(label="Sandbox ready!", state="complete")
                    
                    # Generate plan
                    plan_summary = {
                        "rows": summary['rows'],
                        "columns": summary['columns'],
                        "categorical_columns": summary['categorical_columns'],
                        "numerical_columns": summary['numerical_columns'],
                        "cols_list": summary['column_types']
                    }
                    
                    st.session_state.plan = st.session_state.agent.analyze_and_plan(
                        plan_summary,
                        st.session_state.quality_report
                    )
                    
                    if 'error' in st.session_state.plan:
                        st.error(f"❌ Planning failed: {st.session_state.plan['error']}")
                        if 'raw_response' in st.session_state.plan:
                            with st.expander("🔍 View LLM Response"):
                                st.code(st.session_state.plan['raw_response'])
                        st.stop()
                    
                    # Log to experiment
                    st.session_state.experiment_tracker.log_stage("planning", st.session_state.plan)
                    
                    st.session_state.pipeline_stage = 'planned'
                    st.success("✅ ML Strategy generated!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    with st.expander("Full Error"):
                        st.code(traceback.format_exc())
    
    # Stage 2: Preprocessing
    if st.session_state.pipeline_stage in ['planned', 'preprocessed', 'selected', 'trained']:
        if st.session_state.plan:
            with st.expander("📋 View ML Strategy", expanded=(st.session_state.pipeline_stage == 'planned')):
                plan = st.session_state.plan
                
                col1, col2 = st.columns(2)
                col1.metric("Problem Type", plan.get('problem_type', 'Unknown'))
                col2.metric("Target Column", plan.get('target_column', 'Unknown'))
                
                st.markdown("**Overview:**")
                st.write(plan.get('plan_overview', 'No overview available'))
                
                st.markdown("**Preprocessing Strategy:**")
                st.info(plan.get('preprocessing_strategy', 'Standard preprocessing'))
                
                st.markdown("**Feature Engineering:**")
                st.info(plan.get('feature_engineering_strategy', 'Basic features'))
                
                st.markdown("**Models to Train:**")
                for model in plan.get('models', []):
                    st.write(f"- **{model['name']}** with params: {model.get('params', {})}")
        
        if st.session_state.pipeline_stage == 'planned':
            st.subheader("Step 2: Data Preprocessing & Feature Engineering")
            
            if st.button("🛠 Execute Preprocessing", type="primary", width='stretch'):
                with st.status("Preprocessing data...", expanded=True) as status:
                    try:
                        plan = st.session_state.plan
                        target_col = plan.get('target_column', df.columns[-1])
                        
                        # Check cache
                        cache_key = st.session_state.cache_manager._generate_key(df, plan)
                        cached_result = st.session_state.cache_manager.get(cache_key, "preprocessing")
                        
                        if cached_result:
                            st.info("📦 Using cached preprocessing results")
                            st.session_state.pipeline_stage = 'preprocessed'
                            st.rerun()
                        else:
                            # Generate and execute code
                            code = st.session_state.agent.generate_preprocessing_code(
                                summary,
                                target_col,
                                plan.get('preprocessing_strategy', ''),
                                plan.get('feature_engineering_strategy', ''),
                                st.session_state.quality_report
                            )
                            
                            status.update(label="Executing preprocessing code...", state="running")
                            res = st.session_state.agent.executor.execute_code(code)
                            
                            if res['exit_code'] == 0:
                                status.update(label="Preprocessing complete!", state="complete")
                                st.success("✅ Data preprocessed successfully!")
                                
                                # Cache result
                                st.session_state.cache_manager.set(cache_key, res, "preprocessing")
                                
                                # Log to experiment
                                st.session_state.experiment_tracker.log_stage("preprocessing", {
                                    "status": "success",
                                    "output": res['stdout'][:500]
                                })
                                
                                st.session_state.pipeline_stage = 'preprocessed'
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"❌ Preprocessing failed:\n{res['stderr']}")
                                with st.expander("Debug Code"):
                                    st.code(code, language='python')
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    # Stage 3: Feature Selection
    if st.session_state.pipeline_stage in ['preprocessed', 'selected', 'trained']:
        if st.session_state.pipeline_stage == 'preprocessed':
            st.subheader("Step 3: Feature Selection (Optional)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("✨ Run Feature Selection", width='stretch'):
                    with st.spinner("Selecting best features..."):
                        try:
                            plan = st.session_state.plan
                            target_col = plan.get('target_column', df.columns[-1])
                            
                            code = st.session_state.agent.generate_feature_selection_code(
                                target_col,
                                plan.get('feature_selection_strategy', 'SelectKBest')
                            )
                            
                            res = st.session_state.agent.executor.execute_code(code)
                            
                            if res['exit_code'] == 0:
                                st.success("✅ Feature selection complete!")
                                st.session_state.experiment_tracker.log_stage("feature_selection", {
                                    "status": "success"
                                })
                                st.session_state.pipeline_stage = 'selected'
                                st.rerun()
                            else:
                                st.error(f"Failed: {res['stderr']}")
                        except Exception as e:
                            st.error(f"Error: {e}")
            
            with col2:
                if st.button("⏩ Skip & Continue", width='stretch'):
                    st.session_state.pipeline_stage = 'selected'
                    st.rerun()
    
    # Stage 4: Training
    if st.session_state.pipeline_stage in ['selected', 'trained']:
        if st.session_state.pipeline_stage == 'selected':
            st.subheader("Step 4: Model Training with Optuna")
            st.info("🚀 Models will be trained in parallel using Optuna for hyperparameter optimization")
            
            if st.button("🚀 Start Parallel Training", type="primary", width='stretch'):
                with st.status("Training models in parallel...", expanded=True) as status:
                    try:
                        plan = st.session_state.plan
                        target_col = plan.get('target_column', df.columns[-1])
                        use_selection = (st.session_state.pipeline_stage == 'selected')
                        
                        # Progress container
                        progress_text = st.empty()
                        
                        def update_progress(msg):
                            progress_text.write(msg)
                        
                        # Train models in parallel
                        results = st.session_state.agent.train_models_parallel(
                            plan.get('models', []),
                            target_col,
                            use_selection,
                            update_progress
                        )
                        
                        # Process results
                        for result in results:
                            if result['success']:
                                metrics = result['metrics']
                                st.session_state.agent.leaderboard.add_entry(
                                    metrics['model'],
                                    metrics.get('accuracy', 0),
                                    metrics.get('precision', 0),
                                    metrics.get('recall', 0),
                                    metrics.get('best_params', {}),
                                    metrics.get('feature_importance', None),
                                    metrics.get('train_accuracy', None),
                                    metrics.get('confusion_matrix', None),
                                    metrics.get('f1_score', None),
                                    metrics.get('roc_auc', None),
                                    metrics.get('training_time', None)
                                )
                                
                                # Log to experiment
                                st.session_state.experiment_tracker.log_model(
                                    metrics['model'],
                                    metrics,
                                    metrics.get('best_params', {})
                                )
                        
                        # Build ensemble
                        if len(st.session_state.agent.leaderboard.records) >= 2:
                            status.update(label="Building ensemble...", state="running")
                            top_models = sorted(
                                st.session_state.agent.leaderboard.records,
                                key=lambda x: x['Accuracy'],
                                reverse=True
                            )[:3]
                            
                            ens_code = st.session_state.agent.generate_ensemble_code(
                                top_models, target_col, use_selection
                            )
                            ens_res = st.session_state.agent.executor.execute_code(ens_code)
                            
                            if ens_res['exit_code'] == 0:
                                ens_metrics = st.session_state.agent.extract_json(ens_res['stdout'])
                                if ens_metrics:
                                    st.session_state.agent.leaderboard.add_entry(
                                        "Ensemble",
                                        ens_metrics.get('accuracy', 0),
                                        0, 0,
                                        {},
                                        None,
                                        ens_metrics.get('train_accuracy', None),
                                        ens_metrics.get('confusion_matrix', None),
                                        ens_metrics.get('f1_score', None),
                                        ens_metrics.get('roc_auc', None)
                                    )
                        
                        status.update(label="Training complete!", state="complete")
                        st.session_state.pipeline_stage = 'trained'
                        
                        # Save experiment
                        st.session_state.experiment_tracker.end_experiment()
                        
                        time.sleep(1)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Training error: {e}")
                        import traceback
                        with st.expander("Full Error"):
                            st.code(traceback.format_exc())



def render_results_export():
    """Render results and export tab."""
    st.header("🏆 Results & Model Export")
    
    if st.session_state.pipeline_stage == 'trained' and st.session_state.agent:
        if st.session_state.agent.leaderboard.records:
            # Leaderboard
            st.subheader("📊 Model Leaderboard")
            lb_df = st.session_state.agent.leaderboard.get_dataframe()
            
            # Display main metrics
            display_cols = ['Model', 'Accuracy', 'F1Score', 'ROC_AUC', 'TrainingTime', 'TrainAccuracy']
            display_df = lb_df[[col for col in display_cols if col in lb_df.columns]]
            st.dataframe(display_df, width='stretch')
            
            # Best model insights
            st.divider()
            st.subheader("🎯 Best Model Analysis")
            
            best_record = lb_df.iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Best Model", best_record['Model'])
            col2.metric("Accuracy", f"{best_record['Accuracy']:.4f}")
            col3.metric("F1 Score", f"{best_record.get('F1Score', 0):.4f}")
            col4.metric("ROC AUC", f"{best_record.get('ROC_AUC', 0):.4f}")
            
            # Visualizations
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                st.markdown("#### 📊 Feature Importance")
                if best_record.get('FeatureImportance'):
                    fi_data = best_record['FeatureImportance']
                    if isinstance(fi_data, dict):
                        fi_df = pd.DataFrame(
                            list(fi_data.items()),
                            columns=['Feature', 'Importance']
                        ).sort_values('Importance', ascending=True).tail(20)
                        
                        fig_imp = px.bar(
                            fi_df,
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title="Top 20 Most Important Features"
                        )
                        st.plotly_chart(fig_imp, width='stretch')
                    else:
                        st.info("Feature importance not available for this model type")
                else:
                    st.info("Feature importance not available")
            
            with col_viz2:
                st.markdown("#### 🎯 Confusion Matrix")
                if best_record.get('ConfusionMatrix'):
                    cm_data = best_record['ConfusionMatrix']
                    if isinstance(cm_data, list):
                        fig_cm = px.imshow(
                            cm_data,
                            text_auto=True,
                            aspect="auto",
                            labels=dict(x="Predicted", y="Actual", color="Count"),
                            title=f"Confusion Matrix - {best_record['Model']}"
                        )
                        st.plotly_chart(fig_cm, width='stretch')
                else:
                    st.info("Confusion matrix not available")
            
            # Model comparison chart
            st.markdown("#### 📈 Model Comparison")
            comparison_df = lb_df[['Model', 'Accuracy', 'Precision', 'Recall']].head(5)
            fig_compare = go.Figure()
            
            for metric in ['Accuracy', 'Precision', 'Recall']:
                if metric in comparison_df.columns:
                    fig_compare.add_trace(go.Bar(
                        name=metric,
                        x=comparison_df['Model'],
                        y=comparison_df[metric]
                    ))
            
            fig_compare.update_layout(
                barmode='group',
                title="Model Performance Comparison",
                xaxis_title="Model",
                yaxis_title="Score"
            )
            st.plotly_chart(fig_compare, width='stretch')
            
            # Overfitting check
            st.divider()
            st.subheader("🔍 Overfitting Analysis")
            
            if best_record.get('TrainAccuracy') is not None:
                train_acc = best_record['TrainAccuracy']
                test_acc = best_record['Accuracy']
                diff = train_acc - test_acc
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Train Accuracy", f"{train_acc:.4f}")
                col2.metric("Test Accuracy", f"{test_acc:.4f}")
                col3.metric("Gap", f"{diff:.4f}", delta=f"{-diff:.4f}")
                
                if diff > 0.10:
                    st.warning(f"⚠️ High overfitting detected! Gap: {diff:.4f}")
                    
                    if st.button("🛡️ Apply Regularization Fix"):
                        with st.spinner("Retraining with stronger regularization..."):
                            try:
                                plan = st.session_state.plan
                                target_col = plan.get('target_column', '')
                                use_selection = True
                                
                                fix_code = st.session_state.agent.generate_fix_overfitting_code(
                                    best_record['Model'],
                                    best_record['Parameters'],
                                    train_acc,
                                    test_acc,
                                    target_col,
                                    use_selection
                                )
                                
                                res = st.session_state.agent.executor.execute_code(fix_code)
                                
                                if res['exit_code'] == 0:
                                    metrics = st.session_state.agent.extract_json(res['stdout'])
                                    if metrics:
                                        st.session_state.agent.leaderboard.add_entry(
                                            metrics['model'],
                                            metrics.get('accuracy', 0),
                                            metrics.get('precision', 0),
                                            metrics.get('recall', 0),
                                            metrics.get('best_params', {}),
                                            None,
                                            metrics.get('train_accuracy', None),
                                            metrics.get('confusion_matrix', None),
                                            metrics.get('f1_score', None),
                                            metrics.get('roc_auc', None)
                                        )
                                        st.success("✅ Regularized model trained!")
                                        st.rerun()
                                else:
                                    st.error(f"Fix failed: {res['stderr']}")
                            except Exception as e:
                                st.error(f"Error: {e}")
                
                elif diff < -0.05:
                    st.info(f"ℹ️ Model performs better on test set (gap: {diff:.4f})")
                else:
                    st.success("✅ Good generalization - low train-test gap")
            
            # Export section
            st.divider()
            st.subheader("📦 Export & Deployment")
            
            col_export1, col_export2, col_export3 = st.columns(3)
            
            with col_export1:
                st.markdown("##### 💾 Download Model")
                if st.button("📥 Download Ensemble (.pkl)", width='stretch'):
                    with st.spinner("Downloading..."):
                        content = st.session_state.agent.executor.download_file("final_ensemble.pkl")
                        if content:
                            st.download_button(
                                label="💾 Save Model File",
                                data=content,
                                file_name="autonomous_model.pkl",
                                mime="application/octet-stream",
                                width='stretch'
                            )
                        else:
                            st.warning("Model file not found")
            
            with col_export2:
                st.markdown("##### ⚡ FastAPI Code")
                if st.button("Generate API Code", width='stretch'):
                    api_code = st.session_state.agent.generate_api_code("autonomous_model.pkl")
                    st.code(api_code, language="python")
                    st.download_button(
                        "Download serve.py",
                        api_code,
                        "serve.py",
                        width='stretch'
                    )
            
            with col_export3:
                st.markdown("##### 🐳 Docker Setup")
                if st.button("Generate Dockerfile", width='stretch'):
                    docker_code = st.session_state.agent.generate_docker_code()
                    st.code(docker_code, language="dockerfile")
                    st.download_button(
                        "Download Dockerfile",
                        docker_code,
                        "Dockerfile",
                        width='stretch'
                    )
            
            # Deployment instructions
            with st.expander("📖 Deployment Instructions"):
                st.markdown("""
                ### Quick Deployment Guide
                
                **1. Local Testing:**
                ```bash
                python serve.py
                # API available at http://localhost:8000
                ```
                
                **2. Docker Deployment:**
                ```bash
                docker build -t automl-model .
                docker run -p 8000:8000 automl-model
                ```
                
                **3. API Usage:**
                ```python
                import requests
                
                response = requests.post(
                    "http://localhost:8000/predict",
                    json={"data": [{"feature1": value1, "feature2": value2}]}
                )
                print(response.json())
                ```
                
                **4. Health Check:**
                ```bash
                curl http://localhost:8000/health
                ```
                """)
            
            # Restart button
            st.divider()
            if st.button("🔄 Start New Experiment", width='stretch'):
                st.session_state.pipeline_stage = 'uploaded'
                st.session_state.plan = None
                st.session_state.agent = None
                st.rerun()
        
        else:
            st.info("No models trained yet. Complete the ML pipeline first.")
    else:
        st.info("Complete the ML pipeline to see results.")


def render_experiments_tab():
    """Render experiments comparison tab."""
    st.header("📚 Experiment History & Comparison")
    
    experiments = st.session_state.experiment_tracker.list_experiments()
    
    if not experiments:
        st.info("No experiments yet. Complete a full pipeline to create your first experiment!")
        return
    
    # Global leaderboard
    st.subheader("🏆 Global Model Leaderboard")
    global_lb = st.session_state.experiment_tracker.get_leaderboard()
    
    if not global_lb.empty:
        st.dataframe(global_lb.head(10), width='stretch')
    
    # Experiment list
    st.divider()
    st.subheader("🔬 All Experiments")
    
    for exp in experiments:
        with st.expander(f"📊 {exp['name']} - {exp['timestamp']}"):
            col1, col2, col3 = st.columns(3)
            
            dataset_info = exp.get('dataset_info', {})
            col1.metric("Dataset", dataset_info.get('filename', 'Unknown'))
            col2.metric("Rows", dataset_info.get('rows', 0))
            col3.metric("Models Trained", len(exp.get('models', [])))
            
            # Show stages
            if exp.get('stages'):
                st.markdown("**Pipeline Stages:**")
                for stage_name, stage_data in exp['stages'].items():
                    st.write(f"- ✅ {stage_name.replace('_', ' ').title()}")
            
            # Show best model
            if exp.get('models'):
                best_model = max(exp['models'], key=lambda m: m['metrics'].get('accuracy', 0))
                st.markdown(f"**Best Model:** {best_model['name']} - "
                          f"Accuracy: {best_model['metrics'].get('accuracy', 0):.4f}")
            
            # Full details
            with st.expander("View Full Details"):
                st.json(exp)


if __name__ == "__main__":
    main()
