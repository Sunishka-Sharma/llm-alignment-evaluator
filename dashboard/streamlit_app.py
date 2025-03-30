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
    results_path = Path(results_dir)
    
    if not results_path.exists():
        return {}
    
    for model_dir in results_path.glob("*"):
        if model_dir.is_dir():
            eval_file = model_dir / "evaluation_results.csv"
            if eval_file.exists():
                results[model_dir.name] = {
                    'eval': pd.read_csv(eval_file),
                    'rewrite': load_json(model_dir / "rewrite_history.json"),
                    'requests': load_json(model_dir / "request_log.json")
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
    
    # Create plots directory if it doesn't exist
    os.makedirs('results/plots', exist_ok=True)
    
    # Analysis tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overall Scores",
        "Category Analysis",
        "Prompt Rewrites",
        "API Usage",
        "Raw Data"
    ])
    
    with tab1:
        st.header("Overall Alignment Scores")
        cols = st.columns(len(selected_models))
        
        for i, model in enumerate(selected_models):
            with cols[i]:
                df = results[model]['eval']
                score_cols = [col for col in df.columns if col.startswith('scores.')]
                overall_score = df[score_cols].mean().mean()
                
                # Radar chart
                categories = [col.replace('scores.', '') for col in score_cols]
                scores = df[score_cols].mean().values.tolist()
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=scores + [scores[0]],
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=model
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 3]
                        )),
                    showlegend=True,
                    title=f"{model} Alignment Dimensions"
                )
                
                # Save radar plot
                fig.write_image(f"results/plots/{model}_radar.png")
                
                st.plotly_chart(fig, use_container_width=True, key=f"radar_{model}")
                st.metric("Overall Score", f"{overall_score:.2f}/3")
    
    with tab2:
        st.header("Performance by Category")
        
        for i, model in enumerate(selected_models):
            st.subheader(model)
            df = results[model]['eval']
            
            # Group by category
            score_cols = [col for col in df.columns if col.startswith('scores.')]
            cat_scores = df.groupby('category')[score_cols].mean()
            
            # Plot
            fig = px.bar(
                cat_scores.mean(axis=1).reset_index(),
                x='category',
                y=0,
                title=f"{model} - Scores by Category",
                labels={'0': 'Score', 'category': 'Category'}
            )
            
            # Save category plot
            fig.write_image(f"results/plots/{model}_category_scores.png")
            
            st.plotly_chart(fig, use_container_width=True, key=f"cat_{model}")
    
    with tab3:
        st.header("Prompt Rewrite Analysis")
        
        for i, model in enumerate(selected_models):
            st.subheader(model)
            rewrite_data = results[model].get('rewrite', {})
            
            if not rewrite_data:
                st.info("No rewrite data available. Run evaluation with --rewrite flag.")
                continue
            
            # Show rewrite stats
            rewrites = pd.DataFrame(rewrite_data)
            st.metric("Total Rewrites", len(rewrites))
            
            if not rewrites.empty:
                st.write("Sample Rewrites:")
                for _, row in rewrites.iterrows():
                    with st.expander(f"Rewrite due to: {', '.join(row['rules_triggered'])}"):
                        st.text("Original:")
                        st.code(row['original_prompt'])
                        st.text("Rewritten:")
                        st.code(row['final_prompt'])
    
    with tab4:
        st.header("API Usage Analysis")
        
        for i, model in enumerate(selected_models):
            st.subheader(model)
            request_data = results[model].get('requests', {})
            
            if not request_data:
                st.info("No request log available.")
                continue
            
            # Show request stats
            requests = pd.DataFrame(request_data.get('requests', []))
            total = request_data.get('total_requests', 0)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Requests", total)
            
            if not requests.empty:
                with col2:
                    by_purpose = requests['purpose'].value_counts()
                    st.metric("Unique Purposes", len(by_purpose))
                
                # Plot requests by purpose
                fig = px.pie(
                    values=by_purpose.values,
                    names=by_purpose.index,
                    title="Requests by Purpose"
                )
                
                # Save API usage plot
                fig.write_image(f"results/plots/{model}_api_usage.png")
                
                st.plotly_chart(fig, use_container_width=True, key=f"api_{model}")
    
    with tab5:
        st.header("Raw Data")
        
        for i, model in enumerate(selected_models):
            with st.expander(f"Show {model} Data"):
                st.dataframe(results[model]['eval'])

if __name__ == "__main__":
    main() 