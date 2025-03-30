import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
import sys
import os
import copy
import numpy as np

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
            cross_eval_file = model_dir / "cross_evaluation_results.json"
            
            if eval_file.exists():
                df = pd.read_csv(eval_file)
                results[model_dir.name] = {
                    'eval': df,
                    'rewrite': load_json(rewrite_file),
                    'requests': load_json(request_file),
                    'cross_eval': load_json(cross_eval_file)
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
    """Create an API usage pie chart including cross-model evaluation calls."""
    if not request_data or 'requests' not in request_data:
        return None
        
    requests = pd.DataFrame(request_data['requests'])
    by_purpose = requests['purpose'].value_counts()
    
    # Ensure cross-evaluation calls are included
    cross_eval_count = requests['purpose'].str.contains('cross_eval').sum()
    by_purpose['Cross Evaluation'] = cross_eval_count
    
    fig = px.pie(
        values=by_purpose.values,
        names=by_purpose.index,
        title=f"{model_name} - Requests by Purpose"
    )
    return fig

def create_comparison_plot(results: dict, selected_models: list) -> go.Figure:
    """Create a comparison plot for all selected models."""
    fig = go.Figure()
    
    for model in selected_models:
        df = results[model]['eval']
        score_cols = [col for col in df.columns if col.startswith('scores.')]
        scores = df[score_cols].mean().values.tolist()
        categories = [col.replace('scores.', '') for col in score_cols]
        
        # Add trace for each model
        fig.add_trace(go.Scatterpolar(
            r=scores + [scores[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=model
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 3])),
        showlegend=True,
        title="Model Comparison - Alignment Dimensions"
    )
    return fig

def display_rewrite_analysis(results: dict, model: str) -> None:
    """Display rewrite analysis for a model."""
    rewrite_data = results[model].get('rewrite', [])  # Changed from {} to [] since it's a list
    if not rewrite_data:
        st.info("No rewrite data available for this model.")
        return
        
    # Count improved rewrites
    improved_rewrites = [r for r in rewrite_data if r.get("improved", False)]
    total_rewrites = len(improved_rewrites)
    st.metric("Prompts Successfully Rewritten", f"{total_rewrites}/{len(rewrite_data)}")
    
    # Rules triggered analysis
    rules_triggered = {}
    for entry in rewrite_data:
        for rule in entry.get("rules_triggered", []):
            rules_triggered[rule] = rules_triggered.get(rule, 0) + 1
        # Also check iterations for rules
        for iteration in entry.get("iterations", []):
            rule = iteration.get("rule_applied")
            if rule:
                rules_triggered[rule] = rules_triggered.get(rule, 0) + 1
    
    if rules_triggered:
        st.subheader("Constitutional Rules Applied")
        rules_df = pd.DataFrame(
            {"Rule": list(rules_triggered.keys()), 
             "Times Applied": list(rules_triggered.values())}
        ).sort_values("Times Applied", ascending=False)
        
        # Create bar chart for rules
        fig = px.bar(
            rules_df,
            x="Rule",
            y="Times Applied",
            title="Constitutional Rules Usage"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show example rewrites
        st.subheader("Example Rewrites")
        for entry in improved_rewrites[:3]:  # Show first 3 successful rewrites
            rules = entry.get("rules_triggered", [])
            iterations = entry.get("iterations", [])
            if iterations:  # If we have iteration data
                rule_names = [it.get("rule_applied") for it in iterations if it.get("rule_applied")]
            else:
                rule_names = rules
                
            with st.expander(f"Rewrite Example - Rules: {', '.join(rule_names)}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original Prompt:**")
                    st.text(entry.get("original_prompt", ""))
                with col2:
                    st.markdown("**Rewritten Prompt:**")
                    st.text(entry.get("final_prompt", ""))
                    
                # Show iteration steps if available
                if iterations:
                    st.markdown("**Rewrite Steps:**")
                    for i, step in enumerate(iterations, 1):
                        st.markdown(f"Step {i}: Applied rule '{step.get('rule_applied')}'")

def display_chart(fig: go.Figure, container, key: str, use_container_width: bool = True):
    """Safely display a plotly chart with a unique key."""
    if fig is not None:
        # Create a deep copy of the figure to avoid modification of the original
        fig_copy = go.Figure(fig)
        container.plotly_chart(fig_copy, use_container_width=use_container_width, key=key)

def display_request_metrics(results: dict, model: str) -> None:
    """Display detailed request metrics for a model."""
    request_data = results[model].get('requests', {})
    if not request_data:
        st.info("No request data available.")
        return
        
    total_requests = request_data.get('total_requests', 0)
    requests = pd.DataFrame(request_data.get('requests', []))
    
    # Calculate request metrics
    metrics = {
        'Total Requests': total_requests,
        'Base Evaluation': len(requests[requests['purpose'] == 'evaluation']),
        'Rewrites': len(requests[requests['purpose'] == 'rewrite']),
        'Perspective Tests': len(requests[requests['purpose'] == 'perspective']),
        'Cross-Evaluation': len(requests[requests['purpose'] == 'cross_eval'])
    }
    
    # Display metrics in columns
    cols = st.columns(len(metrics))
    for col, (metric, value) in zip(cols, metrics.items()):
        col.metric(metric, value)
    
    # Show request timeline
    if not requests.empty and 'timestamp' in requests.columns:
        requests['timestamp'] = pd.to_datetime(requests['timestamp'])
        fig = px.line(requests.groupby('timestamp').size().reset_index(), 
                     x='timestamp', y=0, title='Request Timeline')
        st.plotly_chart(fig)

def display_model_evaluation_metrics(results: dict, model: str) -> None:
    """Display model evaluation metrics including cross-evaluation scores."""
    eval_data = results[model].get('eval')
    
    if eval_data is not None:
        st.subheader("Model Performance Metrics")
        
        # Calculate average scores
        score_cols = [col for col in eval_data.columns if col.startswith('scores.')]
        avg_scores = eval_data[score_cols].mean()
        
        # Get cross-evaluation score if available
        cross_eval = results[model].get('cross_eval')
        if cross_eval:
            dimensions = ["helpfulness", "safety", "ethical_judgment", "clarity"]
            cross_eval_df = pd.DataFrame(cross_eval)
            cross_eval_scores = [
                cross_eval_df[dim].apply(lambda x: x['score']).mean()
                for dim in dimensions
            ]
            cross_eval_avg = sum(cross_eval_scores) / len(cross_eval_scores)
        else:
            cross_eval_avg = None
        
        # Display metrics in columns
        num_cols = len(score_cols) + (1 if cross_eval_avg else 0)
        cols = st.columns(num_cols)
        
        # Display dimension scores
        for i, (metric, score) in enumerate(avg_scores.items()):
            cols[i].metric(
                metric.replace('scores.', '').replace(' (0-3)', ''),
                f"{score:.2f}/3"
            )
        
        # Display cross-evaluation score if available
        if cross_eval_avg is not None:
            cols[-1].metric(
                "Cross-Eval Score",
                f"{cross_eval_avg:.2f}/3"
            )

def display_cross_evaluation_metrics(results: dict, model: str) -> None:
    """Display cross-evaluation metrics for a model."""
    if not results[model].get('cross_eval'):
        st.info("No cross-evaluation data available for this model.")
        return
        
    eval_data = pd.DataFrame(results[model]['cross_eval'])
    dimensions = ["helpfulness", "safety", "ethical_judgment", "clarity"]
    
    # Calculate average scores
    avg_scores = pd.Series({
        f'scores.{dim} (0-3)': eval_data[dim].apply(lambda x: x['score']).mean()
        for dim in dimensions
    })
    
    # Display overall score
    overall_score = avg_scores.mean()
    st.metric("Cross-Evaluation Score", f"{overall_score:.2f}/3")
    
    # Show dimension scores in columns
    st.subheader("Dimension Scores")
    cols = st.columns(len(dimensions))
    for i, dim in enumerate(dimensions):
        score = avg_scores[f'scores.{dim} (0-3)']
        cols[i].metric(dim.replace('_', ' ').title(), f"{score:.2f}/3")
    
    # Show category breakdown
    st.subheader("Performance by Category")
    for category in eval_data['category'].unique():
        st.write(f"**Category: {category}**")
        cat_evals = eval_data[eval_data['category'] == category]
        
        # Calculate category averages
        cat_scores = {
            dim: cat_evals[dim].apply(lambda x: x['score']).mean()
            for dim in dimensions
        }
        
        # Display in columns
        cols = st.columns(len(dimensions))
        for i, (dim, score) in enumerate(cat_scores.items()):
            cols[i].metric(
                dim.replace('_', ' ').title(),
                f"{score:.2f}/3"
            )
        st.write("---")  # Add separator between categories

def calculate_agreement_stats(results: dict, model: str, all_models: list) -> dict:
    """Calculate agreement statistics between models."""
    other_models = [m for m in all_models if m != model]
    dimensions = ["helpfulness", "harmlessness", "ethical_judgment", "honesty"]
    
    agreements = []
    dimension_agreements = {dim: [] for dim in dimensions}
    
    for other_model in other_models:
        df1 = results[model]['eval']
        df2 = results[other_model]['eval']
        
        for dim in dimensions:
            scores1 = df1[f'scores.{dim} (0-3)']
            scores2 = df2[f'scores.{dim} (0-3)']
            agreement = np.mean(np.abs(scores1 - scores2) <= 0.5) * 100
            agreements.append(agreement)
            dimension_agreements[dim].append(agreement)
    
    # Calculate average agreement for each dimension
    dim_agreement_df = pd.DataFrame({
        'Dimension': dimensions,
        'Agreement %': [np.mean(dimension_agreements[dim]) for dim in dimensions]
    }).sort_values('Agreement %', ascending=False)
    
    return {
        'avg_agreement': np.mean(agreements),
        'max_agreement': np.max(agreements),
        'min_agreement': np.min(agreements),
        'dimension_agreement': dim_agreement_df
    }

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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Overall Scores",
        "Model Comparison",
        "Category Analysis",
        "Perspective Analysis",
        "Constitutional Rewrites",
        "API Usage",
        "Cross-Evaluation"
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
        for model in selected_models:
            st.subheader(model)
            display_model_evaluation_metrics(results, model)
            display_chart(model_charts[model]['radar'], st, f"radar_{model}")
    
    with tab2:
        st.header("Model Comparison")
        if len(selected_models) > 1:
            # Add self vs cross-evaluation comparison
            st.subheader("Self vs Cross-Evaluation Comparison")
            comparison_data = []
            
            for model in selected_models:
                df = results[model]['eval']
                cross_eval = results[model].get('cross_eval', [])
                
                # Self-evaluation scores
                score_cols = [col for col in df.columns if col.startswith('scores.')]
                self_scores = {col.replace('scores.', '').replace(' (0-3)', ''): df[col].mean() for col in score_cols}
                
                # Cross-evaluation scores
                if cross_eval:
                    cross_eval_df = pd.DataFrame(cross_eval)
                    dimensions = ["helpfulness", "safety", "ethical_judgment", "clarity"]
                    cross_scores = {
                        dim: cross_eval_df[dim].apply(lambda x: x['score']).mean()
                        for dim in dimensions
                    }
                else:
                    cross_scores = {}
                
                # Combine scores
                row = {
                    'Model': model,
                    **{f"Self-{k}": v for k, v in self_scores.items()},
                    **{f"Cross-{k}": v for k, v in cross_scores.items()}
                }
                comparison_data.append(row)
            
            comparison_df = pd.DataFrame(comparison_data).set_index('Model')
            
            # Create comparison plot
            fig = go.Figure()
            for col in comparison_df.columns:
                fig.add_trace(go.Bar(
                    name=col,
                    x=comparison_df.index,
                    y=comparison_df[col],
                    text=comparison_df[col].round(2),
                    textposition='auto',
                ))
            
            fig.update_layout(
                title="Self vs Cross-Evaluation Scores",
                yaxis_title="Score (0-3)",
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show the comparison table
            st.dataframe(comparison_df.round(2), use_container_width=True)
            
            # Original comparison plot
            comparison_fig = create_comparison_plot(results, selected_models)
            st.plotly_chart(comparison_fig, use_container_width=True)
        else:
            st.info("Please select multiple models to see comparison.")
    
    with tab3:
        st.header("Performance by Category")
        for i, model in enumerate(selected_models):
            st.subheader(model)
            display_chart(model_charts[model]['category'], st, f"category_{model}_{i}")
    
    with tab4:
        st.header("Perspective Analysis")
        for i, model in enumerate(selected_models):
            st.subheader(model)
            if model_charts[model]['perspective'] is not None:
                display_chart(model_charts[model]['perspective'], st, f"perspective_{model}_{i}")
            else:
                st.info("No perspective analysis data available.")
    
    with tab5:
        st.header("Constitutional Rewrite Analysis")
        for model in selected_models:
            st.subheader(f"{model} Rewrite Analysis")
            display_rewrite_analysis(results, model)
    
    with tab6:
        st.header("API Usage Analysis")
        for i, model in enumerate(selected_models):
            st.subheader(model)
            request_data = results[model].get('requests', {})
            
            if not request_data:
                st.info("No request log available.")
                continue
            
            # Show request stats with cross-evaluation included
            total = request_data.get('total_requests', 0)
            requests = pd.DataFrame(request_data.get('requests', []))
            
            # Update request purposes to include cross-evaluation
            if not requests.empty:
                cross_eval_requests = requests[requests['purpose'].str.contains('cross_eval', na=False)]
                eval_requests = requests[requests['purpose'] == 'evaluation']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Requests", total)
                with col2:
                    st.metric("Evaluation Requests", len(eval_requests))
                with col3:
                    st.metric("Cross-Evaluation Requests", len(cross_eval_requests))
            
            if model_charts[model]['api'] is not None:
                display_chart(model_charts[model]['api'], st, f"api_{model}_{i}")

    with tab7:
        st.header("Cross-Model Evaluation Analysis")
        
        # Add explanation of scoring
        st.markdown("""
        ### Understanding the Scores
        
        The evaluation system uses two types of scores:
        1. **Self-Evaluation Scores**: These are scores (0-3) that the model assigns to itself across dimensions like helpfulness, harmlessness, ethical judgment, and honesty.
        2. **Cross-Evaluation Scores**: These are scores (0-3) given by other models when evaluating this model's responses. The dimensions are helpfulness, safety, ethical judgment, and clarity.
        
        The scoring ranges from 0 (poor) to 3 (excellent) in both cases, but the cross-evaluation provides an external perspective on the model's performance.
        """)
        
        # Display cross-model evaluation plots
        st.subheader("Model-to-Model Evaluation")
        
        # Load and display the cross-model evaluation plot
        cross_eval_plot = os.path.join(results_dir, "plots", "comparison", "cross_model_evaluation.png")
        if os.path.exists(cross_eval_plot):
            st.image(cross_eval_plot, use_container_width=True)
        else:
            st.info("Cross-model evaluation plot not available. Run evaluation with multiple models first.")
        
        # Show detailed metrics for each model
        st.subheader("Detailed Cross-Evaluation Metrics")
        for model in selected_models:
            with st.expander(f"{model} Cross-Evaluation Details"):
                display_cross_evaluation_metrics(results, model)
                
                # Show agreement statistics
                if len(selected_models) > 1:
                    st.subheader("Agreement Analysis")
                    agreement_stats = calculate_agreement_stats(results, model, selected_models)
                    
                    # Display agreement metrics
                    cols = st.columns(3)
                    cols[0].metric("Average Agreement", f"{agreement_stats['avg_agreement']:.1f}%")
                    cols[1].metric("Highest Agreement", f"{agreement_stats['max_agreement']:.1f}%")
                    cols[2].metric("Lowest Agreement", f"{agreement_stats['min_agreement']:.1f}%")
                    
                    # Show dimension-wise agreement
                    st.write("Agreement by Dimension:")
                    st.dataframe(agreement_stats['dimension_agreement'])

if __name__ == "__main__":
    main()