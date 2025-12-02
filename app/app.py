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
                
                col_start, col_spacer = st.columns([1, 3])
                with col_start:
                    start_btn = st.button("🚀 Start Autonomous Training", type="primary")
                
                status_container = st.container()
                
                if start_btn:
                    try:
                        executor = DaytonaExecutor()
                        st.session_state.agent = AutoMLAgent(executor)
                    except Exception as e:
                        st.error(f"Failed to initialize Agent. Check API Keys. Error: {e}")
                        return

                    with status_container:
                        with st.status("Agent Working...", expanded=True) as status:
                            def update_status(msg): status.write(msg)
                            logs = st.session_state.agent.run_experiment(df, progress_callback=update_status)
                            
                            for log_item in logs:
                                msg = log_item.get("msg", "")
                                type_ = log_item.get("type", "info")
                                
                                if type_ == "error":
                                    st.error(msg)
                                elif type_ == "warning":
                                    st.warning(msg)
                                elif type_ == "info":
                                    if "✅" in msg: st.success(msg)
                                    elif "❌" in msg: st.error(msg)
                                    else: st.info(msg)
                                    
                            status.update(label="Training Complete!", state="complete", expanded=False)

                    with st.expander("🔍 View Execution Logs & Errors"):
                        for log_item in logs:
                            st.markdown(f"**{log_item.get('msg')}**")
                            if "details" in log_item:
                                st.json(log_item["details"])

                if st.session_state.agent and st.session_state.agent.leaderboard.records:
                    st.subheader("🏆 Leaderboard")
                    lb_df = st.session_state.agent.leaderboard.get_dataframe()
                    
                    # Show leaderboard without complex columns first
                    display_df = lb_df.drop(columns=['FeatureImportance'], errors='ignore')
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Model Insights Section
                    st.divider()
                    st.subheader("🧠 Best Model Insights")
                    
                    best_record = lb_df.iloc[0]
                    st.markdown(f"**Best Model:** {best_record['Model']} (Accuracy: {best_record['Accuracy']})")
                    
                    if best_record.get('FeatureImportance'):
                        fi_data = best_record['FeatureImportance']
                        if isinstance(fi_data, dict):
                            fi_df = pd.DataFrame(list(fi_data.items()), columns=['Feature', 'Importance'])
                            fi_df = fi_df.sort_values(by='Importance', ascending=True) # Sort for bar chart
                            
                            fig_imp = px.bar(fi_df, x='Importance', y='Feature', orientation='h', 
                                             title=f"Feature Importance ({best_record['Model']})")
                            st.plotly_chart(fig_imp, use_container_width=True)
                        else:
                            st.info("Feature Importance format not recognized.")
                    else:
                        st.info("No feature importance available for this model type.")

                    
                    st.divider()
                    st.subheader("📦 Deployment & Artifacts")
                    
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

    else:
        st.info("Please upload a dataset to begin.")

if __name__ == "__main__":
    main()
