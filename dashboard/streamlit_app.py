import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
import sys
import os

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
        return {}
    
    for model_dir in results_path.glob("*"):
        if model_dir.is_dir():
            model_name = model_dir.name
            eval_file = model_dir / "evaluation_results.csv"
            rewrite_file = model_dir / "rewrite_history.json"
            request_file = model_dir / "request_log.json"
            
            if eval_file.exists():
                results[model_name] = {
                    'eval': pd.read_csv(eval_file),
                    'rewrite': load_json(rewrite_file),
                    'requests': load_json(request_file),
                    'plots': {
                        'dimension_scores': str(Path(results_dir) / "plots" / "model_specific" / model_name / "dimension_scores.png"),
                        'comparison': str(Path(results_dir) / "plots" / "comparison" / "dimension_scores_comparison.png")
                    }
                }
    
    return results

def load_json(path: Path) -> dict:
    """Load JSON file if it exists."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

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
    
    # Save plot
    if not os.path.exists('results/plots'):
        os.makedirs('results/plots')
    fig.write_image(f"results/plots/{model_name}_dimension_scores.png")
    
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
        st.warning("⚠️ No results found! Please run evaluation first:")
        st.code("python src/main.py --run-all")
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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overall Scores",
        "Category Analysis",
        "Perspective Analysis",
        "Prompt Rewrites",
        "API Usage",
        "Raw Data"
    ])
    
    with tab1:
        st.header("Overall Alignment Scores")
        
        # Show comparison plot
        comparison_plot = Path(results[available_models[0]]['plots']['comparison'])
        if comparison_plot.exists():
            st.image(str(comparison_plot), caption="Model Comparison")
        
        # Show individual model scores
        cols = st.columns(len(selected_models))
        for i, model in enumerate(selected_models):
            with cols[i]:
                df = results[model]['eval']
                score_cols = [col for col in df.columns if col.startswith('scores.')]
                overall_score = df[score_cols].mean().mean()
                
                st.metric(f"{model} Overall Score", f"{overall_score:.2f}/3")
                
                # Show dimension scores plot
                dimension_plot = Path(results[model]['plots']['dimension_scores'])
                if dimension_plot.exists():
                    st.image(str(dimension_plot), caption=f"{model} Dimensions")
    
    with tab2:
        st.header("Performance by Category")
        for model in selected_models:
            st.subheader(model)
            df = results[model]['eval']
            
            # Group by category
            score_cols = [col for col in df.columns if col.startswith('scores.')]
            cat_scores = df.groupby('category')[score_cols].mean()
            
            # Create bar chart
            fig = px.bar(
                cat_scores.mean(axis=1).reset_index(),
                x='category',
                y=0,
                title=f"{model} - Scores by Category",
                labels={'0': 'Score', 'category': 'Category'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Perspective Analysis")
        for model in selected_models:
            st.subheader(model)
            df = results[model]['eval']
            
            if 'perspective' in df.columns:
                perspectives = df['perspective'].unique()
                score_cols = [col for col in df.columns if col.startswith('scores.')]
                
                # Create radar chart for perspectives
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
                    title=f"{model} - Perspective Analysis"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No perspective analysis data available.")
    
    with tab4:
        st.header("Prompt Rewrite Analysis")
        for model in selected_models:
            st.subheader(model)
            rewrite_data = results[model].get('rewrite', {})
            
            if rewrite_data:
                total_rewrites = len([r for r in rewrite_data if r.get("improved", False)])
                st.metric("Total Rewrites", total_rewrites)
                
                # Show rewrite examples
                for i, rewrite in enumerate(rewrite_data):
                    if rewrite.get("improved", False):
                        with st.expander(f"Rewrite {i+1}: {', '.join(rewrite['rules_triggered'])}"):
                            st.text("Original:")
                            st.code(rewrite['original_prompt'])
                            st.text("Rewritten:")
                            st.code(rewrite['final_prompt'])
            else:
                st.info("No rewrite data available. Run with --rewrite flag to enable.")
    
    with tab5:
        st.header("API Usage Analysis")
        for model in selected_models:
            st.subheader(model)
            request_data = results[model].get('requests', {})
            
            if request_data:
                total = request_data.get('total_requests', 0)
                st.metric("Total API Calls", total)
                
                # Analyze request purposes
                if 'requests' in request_data:
                    purposes = {}
                    for req in request_data['requests']:
                        purpose = req.get('purpose', 'unknown')
                        purposes[purpose] = purposes.get(purpose, 0) + 1
                    
                    # Create pie chart
                    fig = px.pie(
                        values=list(purposes.values()),
                        names=list(purposes.keys()),
                        title="API Calls by Purpose"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No API usage data available.")
    
    with tab6:
        st.header("Raw Data")
        for model in selected_models:
            with st.expander(f"Show {model} Data"):
                st.dataframe(results[model]['eval'])

if __name__ == "__main__":
    main() 