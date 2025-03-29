import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.analyze_results import AlignmentAnalyzer

st.set_page_config(
    page_title="LLM Alignment Evaluator Dashboard",
    page_icon="🔍",
    layout="wide"
)

def load_results(results_dir: str) -> dict:
    """Load results for all evaluated models."""
    results = {}
    results_path = Path(results_dir)
    
    if not results_path.exists():
        return {}
    
    for file in results_path.glob("*.csv"):
        model_name = file.stem.replace("_results", "")
        results[model_name] = pd.read_csv(file)
    
    return results

def plot_dimension_scores(df: pd.DataFrame, model_name: str):
    """Plot average scores across dimensions."""
    score_cols = [col for col in df.columns if col.startswith('scores.')]
    avg_scores = df[score_cols].mean()
    
    fig = go.Figure(data=[
        go.Bar(
            x=avg_scores.index,
            y=avg_scores.values,
            text=avg_scores.values.round(2),
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=f"Average Alignment Scores for {model_name}",
        xaxis_title="Dimension",
        yaxis_title="Score (0-3)",
        yaxis_range=[0, 3]
    )
    
    return fig

def plot_category_comparison(df: pd.DataFrame, model_name: str):
    """Plot scores by category."""
    categories = df['category'].unique()
    score_cols = [col for col in df.columns if col.startswith('scores.')]
    
    category_scores = []
    for cat in categories:
        cat_df = df[df['category'] == cat]
        for col in score_cols:
            category_scores.append({
                'Category': cat,
                'Dimension': col.replace('scores.', ''),
                'Score': cat_df[col].mean()
            })
    
    cat_df = pd.DataFrame(category_scores)
    
    fig = px.bar(
        cat_df,
        x='Category',
        y='Score',
        color='Dimension',
        title=f"Scores by Category for {model_name}",
        barmode='group'
    )
    
    fig.update_layout(yaxis_range=[0, 3])
    return fig

def plot_perspective_drift(df: pd.DataFrame, model_name: str):
    """Plot perspective drift analysis."""
    if 'perspective' not in df.columns:
        return None
        
    perspectives = df['perspective'].unique()
    score_cols = [col for col in df.columns if col.startswith('scores.')]
    
    perspective_scores = []
    for persp in perspectives:
        persp_df = df[df['perspective'] == persp]
        for col in score_cols:
            perspective_scores.append({
                'Perspective': persp,
                'Dimension': col.replace('scores.', ''),
                'Score': persp_df[col].mean()
            })
    
    persp_df = pd.DataFrame(perspective_scores)
    
    fig = px.line_polar(
        persp_df,
        r='Score',
        theta='Dimension',
        color='Perspective',
        line_close=True,
        title=f"Perspective Analysis for {model_name}"
    )
    
    return fig

def main():
    st.title("🔍 LLM Alignment Evaluator Dashboard")
    
    # Sidebar
    st.sidebar.title("Settings")
    results_dir = st.sidebar.text_input(
        "Results Directory",
        value="results"
    )
    
    # Load results
    results = load_results(results_dir)
    
    if not results:
        st.warning("No results found. Please run evaluation first!")
        return
    
    # Model selection
    available_models = list(results.keys())
    selected_models = st.sidebar.multiselect(
        "Select Models to Compare",
        available_models,
        default=available_models[:2] if len(available_models) > 1 else available_models
    )
    
    if not selected_models:
        st.warning("Please select at least one model to analyze.")
        return
    
    # Analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Overall Scores",
        "Category Analysis",
        "Perspective Drift",
        "Raw Data"
    ])
    
    with tab1:
        st.header("Overall Alignment Scores")
        cols = st.columns(len(selected_models))
        
        for i, model in enumerate(selected_models):
            with cols[i]:
                df = results[model]
                fig = plot_dimension_scores(df, model)
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary metrics
                score_cols = [col for col in df.columns if col.startswith('scores.')]
                overall_score = df[score_cols].mean().mean()
                st.metric("Overall Alignment Score", f"{overall_score:.2f}/3")
    
    with tab2:
        st.header("Category Analysis")
        for model in selected_models:
            df = results[model]
            fig = plot_category_comparison(df, model)
            st.plotly_chart(fig, use_container_width=True)
            
            # Category insights
            st.subheader(f"Category Insights for {model}")
            categories = df['category'].unique()
            cols = st.columns(len(categories))
            
            for i, cat in enumerate(categories):
                with cols[i]:
                    cat_df = df[df['category'] == cat]
                    score_cols = [col for col in df.columns if col.startswith('scores.')]
                    avg_score = cat_df[score_cols].mean().mean()
                    st.metric(cat.title(), f"{avg_score:.2f}/3")
    
    with tab3:
        st.header("Perspective Drift Analysis")
        for model in selected_models:
            df = results[model]
            fig = plot_perspective_drift(df, model)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No perspective analysis data available for this model.")
    
    with tab4:
        st.header("Raw Data")
        model = st.selectbox("Select Model", selected_models)
        df = results[model]
        st.dataframe(df)
        
        # Export option
        csv = df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            f"{model}_results.csv",
            "text/csv",
            key='download-csv'
        )

if __name__ == "__main__":
    main() 