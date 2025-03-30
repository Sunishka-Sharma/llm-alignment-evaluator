import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
import sys
import os
import copy

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
    results_path = Path(results_dir) / "model_evaluations"
    
    if not results_path.exists():
        st.error(f"Results directory not found: {results_path}")
        return {}
    
    for model_dir in results_path.glob("*"):
        if model_dir.is_dir():
            eval_file = model_dir / "evaluation_results.csv"
            rewrite_file = model_dir / "rewrite_history.json"
            request_file = model_dir / "request_log.json"
            
            if eval_file.exists():
                df = pd.read_csv(eval_file)
                results[model_dir.name] = {
                    'eval': df,
                    'rewrite': load_json(rewrite_file),
                    'requests': load_json(request_file)
                }
            else:
                st.error(f"Evaluation file not found: {eval_file}")
    
    return results

def load_json(path: Path) -> dict:
    """Load JSON file if it exists."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

def create_radar_chart(df: pd.DataFrame, model_name: str) -> go.Figure:
    """Create a radar chart for model scores."""
    score_cols = [col for col in df.columns if col.startswith('scores.')]
    scores = df[score_cols].mean().values.tolist()
    categories = [col.replace('scores.', '') for col in score_cols]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores + [scores[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=model_name
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 3]
            )),
        showlegend=True,
        title=f"{model_name} Alignment Dimensions"
    )
    return fig

def create_category_chart(df: pd.DataFrame, model_name: str) -> go.Figure:
    """Create a category comparison chart."""
    score_cols = [col for col in df.columns if col.startswith('scores.')]
    cat_scores = df.groupby('category')[score_cols].mean()
    
    fig = px.bar(
        cat_scores.mean(axis=1).reset_index(),
        x='category',
        y=0,
        title=f"{model_name} - Scores by Category",
        labels={'0': 'Score', 'category': 'Category'}
    )
    fig.update_layout(yaxis_range=[0, 3])
    return fig

def create_perspective_chart(df: pd.DataFrame, model_name: str) -> go.Figure:
    """Create a perspective analysis chart."""
    if 'perspective' not in df.columns:
        return None
        
    perspectives = df['perspective'].unique()
    score_cols = [col for col in df.columns if col.startswith('scores.')]
    
    fig = go.Figure()
    for perspective in perspectives:
        persp_scores = df[df['perspective'] == perspective][score_cols].mean()
        fig.add_trace(go.Scatterpolar(
            r=persp_scores.values,
            theta=[col.replace('scores.', '') for col in score_cols],
            fill='toself',
            name=perspective
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 3])),
        showlegend=True,
        title=f"{model_name} - Perspective Analysis"
    )
    return fig

def create_api_usage_chart(request_data: dict, model_name: str) -> go.Figure:
    """Create an API usage pie chart."""
    if not request_data or 'requests' not in request_data:
        return None
        
    requests = pd.DataFrame(request_data['requests'])
    by_purpose = requests['purpose'].value_counts()
    
    fig = px.pie(
        values=by_purpose.values,
        names=by_purpose.index,
        title=f"{model_name} - Requests by Purpose"
    )
    return fig

def display_chart(fig: go.Figure, container, key: str, use_container_width: bool = True):
    """Safely display a plotly chart with a unique key."""
    if fig is not None:
        # Create a deep copy of the figure to avoid modification of the original
        fig_copy = go.Figure(fig)
        container.plotly_chart(fig_copy, use_container_width=use_container_width, key=key)

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
        st.warning("⚠️ No results found! Please run evaluation first:")
        st.code("python src/main.py --model gpt-4")
        st.code("python src/main.py --model claude-3-opus-20240229")
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overall Scores",
        "Category Analysis",
        "Perspective Analysis",
        "API Usage",
        "Raw Data"
    ])
    
    # Generate all charts once
    model_charts = {}
    for model in selected_models:
        df = results[model]['eval']
        model_charts[model] = {
            'radar': create_radar_chart(df, model),
            'category': create_category_chart(df, model),
            'perspective': create_perspective_chart(df, model),
            'api': create_api_usage_chart(results[model].get('requests', {}), model)
        }
    
    with tab1:
        st.header("Overall Alignment Scores")
        cols = st.columns(len(selected_models))
        
        for i, (model, col) in enumerate(zip(selected_models, cols)):
            with col:
                df = results[model]['eval']
                score_cols = [col for col in df.columns if col.startswith('scores.')]
                overall_score = df[score_cols].mean().mean()
                
                display_chart(model_charts[model]['radar'], st, f"radar_{model}_{i}")
                st.metric("Overall Score", f"{overall_score:.2f}/3")
    
    with tab2:
        st.header("Performance by Category")
        for i, model in enumerate(selected_models):
            st.subheader(model)
            display_chart(model_charts[model]['category'], st, f"category_{model}_{i}")
    
    with tab3:
        st.header("Perspective Analysis")
        for i, model in enumerate(selected_models):
            st.subheader(model)
            if model_charts[model]['perspective'] is not None:
                display_chart(model_charts[model]['perspective'], st, f"perspective_{model}_{i}")
            else:
                st.info("No perspective analysis data available.")
    
    with tab4:
        st.header("API Usage Analysis")
        for i, model in enumerate(selected_models):
            st.subheader(model)
            request_data = results[model].get('requests', {})
            
            if not request_data:
                st.info("No request log available.")
                continue
            
            # Show request stats
            total = request_data.get('total_requests', 0)
            requests = pd.DataFrame(request_data.get('requests', []))
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Requests", total)
            with col2:
                st.metric("Unique Purposes", len(requests['purpose'].unique()) if not requests.empty else 0)
            
            if model_charts[model]['api'] is not None:
                display_chart(model_charts[model]['api'], st, f"api_{model}_{i}")
    
    with tab5:
        st.header("Raw Data")
        for i, model in enumerate(selected_models):
            with st.expander(f"Show {model} Data"):
                st.dataframe(results[model]['eval'], key=f"raw_{model}_{i}")

if __name__ == "__main__":
    main()