import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os
import time

# Add the project root to sys.path to allow importing src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_manager import DataManager
from brain import DaytonaExecutor, AutoMLAgent

# Configure Streamlit page
st.set_page_config(page_title="Autonomous ML Agent", page_icon="🤖", layout="wide")

def main():
    st.title("🤖 Autonomous Machine Learning Agent")
    st.markdown("""
    **Autonomous Data Science**: Upload data -> Agent Plans -> Trains in Sandbox -> Optimizes -> Deploys.
    """)

    if 'data_manager' not in st.session_state:
        st.session_state.data_manager = DataManager()
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    
    # Pipeline State Management
    if 'pipeline_stage' not in st.session_state:
        st.session_state.pipeline_stage = 'init'  # init, uploaded, planned, preprocessed, selected, trained
    if 'plan' not in st.session_state:
        st.session_state.plan = None
    if 'processed_summary' not in st.session_state:
        st.session_state.processed_summary = None
    if 'selected_summary' not in st.session_state:
        st.session_state.selected_summary = None

    with st.sidebar:
        st.header("Data Ingestion")
        uploaded_file = st.file_uploader("Upload Dataset", type=['csv', 'xlsx', 'parquet'])
        
        st.divider()
        st.header("⚙️ Agent Settings")
        daytona_key = st.text_input("Daytona API Key", type="password", help="Set in .env or here")
        gemini_key = st.text_input("Gemini API Key", type="password", help="Set in .env or here")
        
        if daytona_key: os.environ['DAYTONA_API_KEY'] = daytona_key
        if gemini_key: os.environ['GEMINI_API_KEY'] = gemini_key

    if uploaded_file:
        if 'current_file' not in st.session_state or st.session_state.current_file != uploaded_file.name:
            success, message = st.session_state.data_manager.load_data(uploaded_file)
            if success:
                st.session_state.current_file = uploaded_file.name
                st.session_state.pipeline_stage = 'uploaded'
                st.success(message)
            else:
                st.error(message)
                return

        dm = st.session_state.data_manager
        summary = dm.get_summary()
        df = dm.get_dataframe()

        if summary:
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🔍 Deep Dive", "📈 Visualizations", "🤖 Agent & Export"])

            with tab1:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Rows", summary['rows'])
                col2.metric("Columns", summary['columns'])
                col3.metric("Missing Values", summary['total_missing'])
                col4.metric("Duplicates", summary['duplicate_rows'])
                st.dataframe(summary['preview'], use_container_width=True)

            with tab2:
                st.dataframe(summary['description'], use_container_width=True)

            with tab3:
                if len(summary['numerical_columns']) > 1:
                    corr = df[summary['numerical_columns']].corr()
                    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
                    st.plotly_chart(fig_corr, use_container_width=True)

            with tab4:
                st.header("🤖 AutoML Orchestrator")
                
                # Step 1: Initialize Agent & Plan
                if st.session_state.pipeline_stage == 'uploaded':
                    if st.button("🧠 Analyze & Plan Strategy", type="primary"):
                        try:
                            if not st.session_state.agent:
                                executor = DaytonaExecutor()
                                st.session_state.agent = AutoMLAgent(executor)
                                
                                # Create sandbox & upload raw data once
                                with st.spinner("Initializing Sandbox & Uploading Data..."):
                                    st.session_state.agent.executor.create_sandbox()
                                    st.session_state.agent.executor.install_dependencies(['pandas', 'scikit-learn', 'numpy', 'joblib'])
                                    st.session_state.agent.executor.upload_data(df, 'dataset.csv')
                            
                            with st.spinner("Generating ML Plan..."):
                                # Summary needs to include column types for the LLM
                                plan_summary = {
                                    "rows": summary['rows'], "columns": summary['columns'],
                                    "categorical_columns": summary['categorical_columns'],
                                    "cols_list": summary['column_types']
                                }
                                st.session_state.plan = st.session_state.agent.analyze_and_plan(plan_summary)
                                st.session_state.pipeline_stage = 'planned'
                                st.rerun()
                        except Exception as e:
                            st.error(f"Planning Failed: {e}")

                # Step 2: Review Plan & Execute Preprocessing
                if st.session_state.pipeline_stage in ['planned', 'preprocessed', 'selected', 'trained']:
                    if st.session_state.plan:
                        with st.expander("📋 View Machine Learning Plan", expanded=False):
                            st.json(st.session_state.plan)
                            
                    if st.session_state.pipeline_stage == 'planned':
                        st.info("Plan generated. Next: Clean data and Engineer features.")
                        if st.button("🛠 Run Preprocessing & Feature Engineering"):
                            with st.spinner("Executing Preprocessing in Sandbox..."):
                                plan = st.session_state.plan
                                target_col = plan.get('target_column', df.columns[-1])
                                prep_strategy = plan.get('preprocessing_strategy', '')
                                eng_strategy = plan.get('feature_engineering_strategy', '')
                                
                                # Code Gen & Exec
                                code = st.session_state.agent.generate_preprocessing_code(summary, target_col, prep_strategy, eng_strategy)
                                res = st.session_state.agent.executor.execute_code(code)
                                
                                if res['exit_code'] == 0:
                                    st.success("Preprocessing Complete!")
                                    # Fetch summary of processed data
                                    st.session_state.processed_summary = st.session_state.agent.get_data_summary_from_sandbox("X_train_processed.csv")
                                    st.session_state.pipeline_stage = 'preprocessed'
                                    st.rerun()
                                else:
                                    st.error(f"Preprocessing Failed:\n{res['stderr']}")
                                    st.expander("Debug Code").code(code)

                # Step 3: Review Processed Data & Feature Selection
                if st.session_state.pipeline_stage in ['preprocessed', 'selected', 'trained']:
                    if st.session_state.processed_summary:
                        with st.expander("📊 View Processed Data Summary", expanded=False):
                            st.json(st.session_state.processed_summary)
                            
                    if st.session_state.pipeline_stage == 'preprocessed':
                        col_fs1, col_fs2 = st.columns(2)
                        with col_fs1:
                            if st.button("✨ Run Feature Selection"):
                                with st.spinner("Selecting Features..."):
                                    plan = st.session_state.plan
                                    target_col = plan.get('target_column', df.columns[-1])
                                    sel_strategy = plan.get('feature_selection_strategy', 'SelectKBest')
                                    
                                    code = st.session_state.agent.generate_feature_selection_code(target_col, sel_strategy)
                                    res = st.session_state.agent.executor.execute_code(code)
                                    
                                    if res['exit_code'] == 0:
                                        st.success("Feature Selection Complete!")
                                        st.session_state.selected_summary = st.session_state.agent.get_data_summary_from_sandbox("X_train_selected.csv")
                                        st.session_state.pipeline_stage = 'selected'
                                        st.rerun()
                                    else:
                                        st.error(f"Feature Selection Failed:\n{res['stderr']}")
                        
                        with col_fs2:
                             if st.button("⏩ Skip Selection & Train"):
                                 st.session_state.pipeline_stage = 'selected' # Treat as selected but use processed
                                 st.session_state.selected_summary = None # Indicator to use processed
                                 st.rerun()

                # Step 4: Training
                if st.session_state.pipeline_stage in ['selected', 'trained']:
                    if st.session_state.selected_summary:
                        with st.expander("🎯 View Selected Features Summary", expanded=False):
                            st.json(st.session_state.selected_summary)
                    
                    if st.session_state.pipeline_stage == 'selected':
                        if st.button("🚀 Start Model Training Loop", type="primary"):
                            status_container = st.container()
                            with status_container:
                                with st.status("Training Models...", expanded=True) as status:
                                    plan = st.session_state.plan
                                    target_col = plan.get('target_column', df.columns[-1])
                                    use_selection = (st.session_state.selected_summary is not None)
                                    
                                    # Loop through models
                                    for model in plan.get('models', []):
                                        status.write(f"Training {model['name']}...")
                                        code = st.session_state.agent.generate_training_code(model, target_col, use_selected_features=use_selection)
                                        res = st.session_state.agent.executor.execute_code(code)
                                        
                                        if res['exit_code'] == 0:
                                            metrics = st.session_state.agent.extract_json(res['stdout'])
                                            if metrics:
                                                st.session_state.agent.leaderboard.add_entry(
                                                    metrics['model'], metrics.get('accuracy', 0),
                                                    metrics.get('precision', 0), metrics.get('recall', 0),
                                                    metrics.get('best_params', {}),
                                                    metrics.get('feature_importance', None),
                                                    metrics.get('train_accuracy', None),
                                                    metrics.get('confusion_matrix', None)
                                                )
                                                status.write(f"✅ {model['name']} (Acc: {metrics.get('accuracy')})")
                                            else:
                                                status.write(f"⚠️ {model['name']} finished but no metrics.")
                                        else:
                                            status.write(f"❌ {model['name']} failed.")
                                            st.error(res['stderr'])
                                    
                                    # Ensembling
                                    if len(st.session_state.agent.leaderboard.records) >= 2:
                                        status.write("Building Ensemble...")
                                        top_models = sorted(st.session_state.agent.leaderboard.records, key=lambda x: x['Accuracy'], reverse=True)[:3]
                                        ens_code = st.session_state.agent.generate_ensemble_code(top_models, target_col, use_selected_features=use_selection)
                                        ens_res = st.session_state.agent.executor.execute_code(ens_code)
                                        if ens_res['exit_code'] == 0:
                                            # Extract ensemble metrics to add to leaderboard properly
                                            ens_metrics = st.session_state.agent.extract_json(ens_res['stdout'])
                                            if ens_metrics:
                                                st.session_state.agent.leaderboard.add_entry(
                                                    "Ensemble", 
                                                    ens_metrics.get('accuracy', 0),
                                                    0, 0, # Precision/Recall might be missing in simple ensemble print
                                                    {}, 
                                                    None,
                                                    ens_metrics.get('train_accuracy', None),
                                                    ens_metrics.get('confusion_matrix', None)
                                                )
                                            status.write("✅ Ensemble Created.")
                                        else:
                                            status.write("❌ Ensemble Failed.")

                                    status.update(label="Training Complete!", state="complete", expanded=False)
                            st.session_state.pipeline_stage = 'trained'
                            st.rerun()

                # Step 5: Results & Export
                if st.session_state.pipeline_stage == 'trained':
                     if st.session_state.agent and st.session_state.agent.leaderboard.records:
                        st.subheader("🏆 Leaderboard")
                        lb_df = st.session_state.agent.leaderboard.get_dataframe()
                        
                        # Show leaderboard without complex columns first
                        display_df = lb_df.drop(columns=['FeatureImportance', 'ConfusionMatrix'], errors='ignore')
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Model Insights Section
                        st.divider()
                        st.subheader("🧠 Best Model Insights")
                        
                        best_record = lb_df.iloc[0]
                        st.markdown(f"**Best Model:** {best_record['Model']}")
                        
                        col_insight1, col_insight2 = st.columns(2)
                        
                        with col_insight1:
                             st.markdown("#### Feature Importance")
                             if best_record.get('FeatureImportance'):
                                fi_data = best_record['FeatureImportance']
                                if isinstance(fi_data, dict):
                                    fi_df = pd.DataFrame(list(fi_data.items()), columns=['Feature', 'Importance'])
                                    fi_df = fi_df.sort_values(by='Importance', ascending=True) # Sort for bar chart
                                    
                                    fig_imp = px.bar(fi_df, x='Importance', y='Feature', orientation='h')
                                    st.plotly_chart(fig_imp, use_container_width=True)
                                else:
                                    st.info("Feature Importance format not recognized.")
                             else:
                                st.info("No feature importance available for this model type.")

                        with col_insight2:
                            st.markdown("#### Confusion Matrix")
                            if best_record.get('ConfusionMatrix'):
                                cm_data = best_record['ConfusionMatrix']
                                if isinstance(cm_data, list):
                                    # Create a heatmap
                                    fig_cm = px.imshow(cm_data, text_auto=True, aspect="auto",
                                                       labels=dict(x="Predicted", y="Actual", color="Count"),
                                                       x=['Class 0', 'Class 1'], y=['Class 0', 'Class 1'],
                                                       title=f"Confusion Matrix ({best_record['Model']})")
                                    st.plotly_chart(fig_cm, use_container_width=True)
                                else:
                                    st.info("Invalid Confusion Matrix format.")
                            else:
                                st.info("Confusion Matrix not available.")

                        # Overfitting Check
                        if best_record.get('TrainAccuracy') is not None:
                            train_acc = best_record['TrainAccuracy']
                            test_acc = best_record['Accuracy']
                            diff = train_acc - test_acc
                            st.markdown(f"**Overfitting Check:** Train Acc: `{train_acc}` vs Test Acc: `{test_acc}`")
                            if diff > 0.10:
                                st.warning(f"⚠️ High Overfitting detected! (gap: {diff:.2f}).")
                                if st.button(f"🛡️ Fix Overfitting for {best_record['Model']}"):
                                    with st.spinner(f"Applying Stronger Regularization to {best_record['Model']}..."):
                                        plan = st.session_state.plan
                                        target_col = plan.get('target_column', df.columns[-1])
                                        use_selection = (st.session_state.selected_summary is not None)
                                        
                                        # Generate Fix Code
                                        fix_code = st.session_state.agent.generate_fix_overfitting_code(
                                            best_record['Model'], 
                                            best_record['Parameters'],
                                            train_acc,
                                            test_acc,
                                            target_col,
                                            use_selection
                                        )
                                        
                                        # Execute
                                        res = st.session_state.agent.executor.execute_code(fix_code)
                                        
                                        if res['exit_code'] == 0:
                                            metrics = st.session_state.agent.extract_json(res['stdout'])
                                            if metrics:
                                                st.session_state.agent.leaderboard.add_entry(
                                                    metrics['model'], metrics.get('accuracy', 0),
                                                    metrics.get('precision', 0), metrics.get('recall', 0),
                                                    metrics.get('best_params', {}),
                                                    None, # No feature importance needed for this update
                                                    metrics.get('train_accuracy', None),
                                                    metrics.get('confusion_matrix', None)
                                                )
                                                st.success("✅ Regularized Model Trained! Check Leaderboard.")
                                                st.rerun()
                                            else:
                                                st.warning("Model trained but metrics parsing failed.")
                                        else:
                                            st.error(f"Fix failed:\n{res['stderr']}")

                            elif diff < -0.05:
                                st.info(f"ℹ️ Model performed better on Test set (gap: {diff:.2f}). Potentially data distribution mismatch.")
                            else:
                                st.success("✅ Good generalization (low train-test gap).")

                        
                        col_dl1, col_dl2 = st.columns(2)
                        
                        with col_dl1:
                            st.markdown("### Download Model")
                            if st.button("📥 Download Ensemble Model (.pkl)"):
                                with st.spinner("Downloading from Sandbox..."):
                                    content = st.session_state.agent.executor.download_file("final_ensemble.pkl")
                                    if content:
                                        st.download_button(
                                            label="Click to Save Model",
                                            data=content,
                                            file_name="autonomous_model.pkl",
                                            mime="application/octet-stream"
                                        )
                                    else:
                                        st.warning("Model file not found in sandbox (maybe training failed?)")

                        with col_dl2:
                            st.markdown("### Generate API Code")
                            if st.button("⚡ Generate FastAPI Code"):
                                code = st.session_state.agent.generate_api_code("autonomous_model.pkl")
                                st.code(code, language="python")
                                st.download_button("Download serve.py", code, "serve.py")
                        
                        if st.button("🔄 Restart Experiment"):
                            st.session_state.pipeline_stage = 'uploaded'
                            st.session_state.plan = None
                            st.session_state.agent = None
                            st.rerun()

    else:
        st.info("Please upload a dataset to begin.")

if __name__ == "__main__":
    main()
