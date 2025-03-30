#!/usr/bin/env python3
"""
Script to generate all plots for the LLM Alignment Evaluator.
This includes comparison plots, model-specific plots, and cross-evaluation plots.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Add src directory to path so we can import from it
src_dir = os.path.join(os.path.dirname(__file__), "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)
from src.analyze_results import AlignmentAnalyzer
from src.demo_rlhf import run_demo

def ensure_dir(path):
    """Ensure a directory exists, creating it if necessary."""
    os.makedirs(path, exist_ok=True)

def generate_all_plots():
    """Generate all the plots for the LLM Alignment Evaluator."""
    print("Generating all plots...")
    
    # Paths
    results_dir = "results"
    plots_dir = os.path.join(results_dir, "plots")
    comparison_dir = os.path.join(plots_dir, "comparison")
    model_specific_dir = os.path.join(plots_dir, "model_specific")
    
    # Ensure directories exist
    ensure_dir(plots_dir)
    ensure_dir(comparison_dir)
    ensure_dir(model_specific_dir)
    
    # Initialize analyzer
    analyzer = AlignmentAnalyzer()
    
    # Load models
    model_eval_dir = os.path.join(results_dir, "model_evaluations")
    if not os.path.exists(model_eval_dir):
        print(f"Error: Model evaluations directory not found: {model_eval_dir}")
        sys.exit(1)
    
    models = []
    for model_dir in os.listdir(model_eval_dir):
        model_path = os.path.join(model_eval_dir, model_dir)
        if os.path.isdir(model_path):
            eval_file = os.path.join(model_path, "evaluation_results.csv")
            if os.path.exists(eval_file):
                print(f"Loading model results from {eval_file}")
                analyzer.add_model_results(model_dir, eval_file)
                models.append(model_dir)
    
    if not models:
        print("Error: No model results found. Please run evaluations first.")
        sys.exit(1)
    
    print(f"Found models: {', '.join(models)}")
    
    # Generate comparison plots
    print("Generating comparison plots...")
    
    # 1. Dimension scores - Spider plot
    plt.figure(figsize=(12, 10))
    analyzer.plot_dimension_scores(save_path=os.path.join(comparison_dir, "dimension_scores_spider.png"))
    plt.close()
    
    # 2. Dimension scores - Bar plot
    plt.figure(figsize=(12, 8))
    generate_dimension_scores_bar(analyzer, save_path=os.path.join(comparison_dir, "dimension_scores_bar.png"))
    plt.close()
    
    # 3. Category comparison
    plt.figure(figsize=(14, 8))
    analyzer.plot_category_scores(save_path=os.path.join(comparison_dir, "category_scores.png"))
    plt.close()
    
    # 4. Rewrite effectiveness plot (replacing Flags frequency)
    plt.figure(figsize=(14, 8))
    generate_rewrite_plot(model_eval_dir, models, save_path=os.path.join(comparison_dir, "rewrite_effectiveness.png"))
    plt.close()
    
    # 5. Cross-model evaluation
    plt.figure(figsize=(16, 10))
    analyzer.plot_cross_model_evaluation(save_path=os.path.join(comparison_dir, "cross_model_evaluation.png"))
    plt.close()
    
    # 6. Self vs Cross-evaluation comparison
    plt.figure(figsize=(14, 10))
    generate_self_vs_cross_comparison(analyzer, models, model_eval_dir, save_path=os.path.join(comparison_dir, "self_vs_cross_evaluation.png"))
    plt.close()
    
    # Generate model-specific plots for each model
    print("Generating model-specific plots...")
    for model in models:
        model_dir = os.path.join(model_specific_dir, model)
        ensure_dir(model_dir)
        
        # 1. Radar plot
        plt.figure(figsize=(10, 8))
        generate_model_radar(analyzer, model, save_path=os.path.join(model_dir, "radar.png"))
        plt.close()
        
        # 2. Dimension scores - Bar plot
        plt.figure(figsize=(12, 8))
        generate_model_dimension_bar(analyzer, model, save_path=os.path.join(model_dir, "dimension_scores_bar.png"))
        plt.close()
        
        # 3. Category scores
        plt.figure(figsize=(14, 8))
        generate_model_category_scores(analyzer, model, save_path=os.path.join(model_dir, "categories.png"))
        plt.close()
        
        # 4. Perspective analysis (if available)
        df = analyzer.model_results[model]
        if 'perspective' in df.columns and df['perspective'].nunique() > 1:
            plt.figure(figsize=(12, 10))
            generate_perspective_plot(analyzer, model, save_path=os.path.join(model_dir, "perspective_drift.png"))
            plt.close()
    
    print("All plots generated successfully!")
    
    # Generate cross-model report
    print("Generating cross-model evaluation report...")
    analyzer.generate_cross_model_report(output_file=os.path.join(results_dir, "analysis", "cross_model_report.md"))
    
    print("Plots and reports generated successfully!")
    return True

def generate_dimension_scores_bar(analyzer, save_path=None):
    """Generate a bar chart comparing dimension scores across models."""
    # Get dimensions and models
    dimensions = analyzer.dimensions
    models = list(analyzer.model_results.keys())
    
    # Prepare data
    data = []
    for model in models:
        df = analyzer.model_results[model]
        for dim in dimensions:
            score_col = f"scores.{dim} (0-3)"
            if score_col in df.columns:
                avg_score = df[score_col].mean()
                data.append({
                    'Model': model,
                    'Dimension': dim,
                    'Score': avg_score
                })
    
    # Create dataframe
    plot_df = pd.DataFrame(data)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    sns.barplot(data=plot_df, x='Dimension', y='Score', hue='Model')
    plt.title('Dimension Scores by Model', fontsize=14)
    plt.xlabel('Dimension', fontsize=12)
    plt.ylabel('Average Score (0-3)', fontsize=12)
    plt.ylim(0, 3)
    plt.legend(title='Model')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def generate_rewrite_plot(model_eval_dir, models, save_path=None):
    """Generate a plot showing the effectiveness of constitutional rewriting."""
    # Prepare data
    effectiveness_data = []
    count_data = []
    
    # For each model, load rewrite history
    for model in models:
        rewrite_path = os.path.join(model_eval_dir, model, "rewrite_history.json")
        
        if os.path.exists(rewrite_path):
            try:
                with open(rewrite_path, 'r') as f:
                    rewrite_data = json.load(f)
                
                # Process rewrite data
                improved_count = sum(1 for entry in rewrite_data if entry.get("improved", False))
                total_count = len(rewrite_data)
                
                # Add to effectiveness data
                if total_count > 0:
                    improvement_rate = improved_count / total_count * 100
                    effectiveness_data.append({
                        'Model': model,
                        'Improvement Rate (%)': improvement_rate
                    })
                    
                    # Add to count data
                    count_data.append({
                        'Model': model,
                        'Status': 'Improved',
                        'Count': improved_count
                    })
                    count_data.append({
                        'Model': model, 
                        'Status': 'Not Improved',
                        'Count': total_count - improved_count
                    })
                
                print(f"Successfully loaded rewrite data for {model}")
            except Exception as e:
                print(f"Error loading rewrite data for {model}: {e}")
        else:
            print(f"No rewrite data found for {model} (expected at {rewrite_path})")
    
    # Create dataframes
    effectiveness_df = pd.DataFrame(effectiveness_data)
    count_df = pd.DataFrame(count_data)
    
    if effectiveness_df.empty:
        print("No valid data for rewrite effectiveness plot. Skipping.")
        # Create a placeholder plot with a message
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No rewrite data available.\nRun evaluations with --rewrite flag to generate data.", 
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        return
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [1, 1.2]})
    
    # Plot 1: Simple bar chart of improvement rates
    bars = ax1.bar(
        effectiveness_df['Model'], 
        effectiveness_df['Improvement Rate (%)'],
        color='#3498db',  # Blue
        width=0.6
    )
    
    # Add percentage labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2, 
            height + 1,
            f'{height:.1f}%', 
            ha='center', 
            va='bottom'
        )
    
    # Customize first plot
    ax1.set_title('Rewriting Improvement Rate', fontsize=14)
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Improvement Rate (%)', fontsize=12)
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot 2: Stacked bar chart of prompt counts
    sns.barplot(
        data=count_df,
        x='Model',
        y='Count',
        hue='Status',
        palette={'Improved': '#2ecc71', 'Not Improved': '#e74c3c'},  # Green for improved, red for not
        ax=ax2
    )
    
    # Customize second plot
    ax2.set_title('Prompt Count by Rewrite Status', fontsize=14)
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Number of Prompts', fontsize=12)
    ax2.legend(title='Status')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add counts as text on the bars
    for p in ax2.patches:
        height = p.get_height()
        if height > 0:  # Only add text if the bar has height
            ax2.text(
                p.get_x() + p.get_width()/2,
                p.get_y() + height/2,
                int(height),
                ha='center',
                va='center',
                color='black',
                fontweight='bold'
            )
    
    # Main title for the entire figure
    fig.suptitle('Constitutional Rewriting Effectiveness', fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def generate_self_vs_cross_comparison(analyzer, models, model_eval_dir, save_path=None):
    """Generate a comparison of self vs cross-evaluation scores."""
    # Get dimensions and models
    dimensions = analyzer.dimensions
    
    # Prepare data
    data = []
    
    # Map cross-eval dimensions to analyzer dimensions
    dim_mapping = {
        "helpfulness": "helpfulness",
        "safety": "harmlessness",
        "ethical_judgment": "ethical_judgment",
        "clarity": "honesty",  # Map clarity to honesty as the closest match
        "honesty": "honesty"
    }
    
    # First, collect self-evaluation data
    for model in models:
        df = analyzer.model_results[model]
        
        for dim in dimensions:
            dim_base = dim.split(' ')[0]  # Remove any suffix like "(0-3)"
            self_col = f"scores.{dim_base} (0-3)"
            
            if self_col in df.columns:
                self_score = df[self_col].mean()
                data.append({
                    'Model': model,
                    'Dimension': dim_base,
                    'Evaluation Type': 'Self',
                    'Score': self_score
                })
    
    # Next, load and process cross-evaluation data from JSON files
    for model in models:
        cross_eval_path = os.path.join(model_eval_dir, model, "cross_evaluation_results.json")
        
        if os.path.exists(cross_eval_path):
            try:
                with open(cross_eval_path, 'r') as f:
                    cross_eval_data = json.load(f)
                
                # Process cross-evaluation data
                cross_scores = {}
                cross_counts = {}
                
                for entry in cross_eval_data:
                    if isinstance(entry, dict):
                        # Initialize dictionaries for each dimension if they don't exist
                        for cross_dim, analyzer_dim in dim_mapping.items():
                            if cross_dim in entry and 'score' in entry[cross_dim]:
                                if analyzer_dim not in cross_scores:
                                    cross_scores[analyzer_dim] = 0
                                    cross_counts[analyzer_dim] = 0
                                
                                score = entry[cross_dim]['score']
                                if score is not None:
                                    cross_scores[analyzer_dim] += score
                                    cross_counts[analyzer_dim] += 1
                
                # Calculate averages and add to data
                for analyzer_dim, total_score in cross_scores.items():
                    if cross_counts[analyzer_dim] > 0:
                        avg_score = total_score / cross_counts[analyzer_dim]
                        data.append({
                            'Model': model,
                            'Dimension': analyzer_dim,
                            'Evaluation Type': 'Cross',
                            'Score': avg_score
                        })
                
                print(f"Successfully loaded cross-evaluation data for {model}")
            except Exception as e:
                print(f"Error loading cross-evaluation data for {model}: {e}")
        else:
            print(f"No cross-evaluation data found for {model} (expected at {cross_eval_path})")
    
    # Create dataframe
    plot_df = pd.DataFrame(data)
    
    if plot_df.empty:
        print("No valid data for self vs cross comparison. Skipping.")
        return
    
    # Ensure all dimension and evaluation type combinations exist for each model
    complete_data = []
    for model in models:
        for dim in analyzer.dimensions:
            dim_base = dim.split(' ')[0]
            for eval_type in ['Self', 'Cross']:
                # Check if this combination exists
                entry = plot_df[(plot_df['Model'] == model) & 
                               (plot_df['Dimension'] == dim_base) & 
                               (plot_df['Evaluation Type'] == eval_type)]
                
                if len(entry) == 0:
                    # Get the average score for this dimension across all models
                    avg_for_dim = plot_df[plot_df['Dimension'] == dim_base]['Score'].mean()
                    if pd.isna(avg_for_dim):
                        avg_for_dim = 0.0
                    
                    # Add a placeholder entry
                    complete_data.append({
                        'Model': model,
                        'Dimension': dim_base,
                        'Evaluation Type': eval_type,
                        'Score': avg_for_dim  # Use average or 0
                    })
                else:
                    # Keep existing entries
                    complete_data.extend(entry.to_dict('records'))
    
    # Create new complete dataframe
    if complete_data:
        complete_df = pd.DataFrame(complete_data)
    else:
        complete_df = plot_df
    
    # Create a more compact plot with better visibility of all bars
    fig, axes = plt.subplots(1, len(models), figsize=(16, 8), sharey=True)
    
    # Single legend for all subplots
    handles, labels = None, None
    
    # Define a consistent order for dimensions
    dimension_order = ["helpfulness", "harmlessness", "ethical_judgment", "honesty"]
    
    # Plot each model in its own subplot
    for i, model in enumerate(models):
        model_data = complete_df[complete_df['Model'] == model].copy()
        
        # Create a categorical type for dimension with specified order
        model_data['Dimension'] = pd.Categorical(
            model_data['Dimension'], 
            categories=dimension_order,
            ordered=True
        )
        
        # Sort by dimension to ensure consistent order
        model_data = model_data.sort_values('Dimension')
        
        ax = axes[i] if len(models) > 1 else axes
        bars = sns.barplot(
            data=model_data,
            x='Dimension',
            y='Score',
            hue='Evaluation Type',
            palette={'Self': '#3498db', 'Cross': '#e74c3c'},  # Blue for self, red for cross
            ax=ax,
            width=0.7,  # Make bars wider
            errorbar=None
        )
        
        # Add value labels to each bar for clarity
        for container in bars.containers:
            bars.bar_label(container, fmt='%.2f', fontsize=8)
        
        # Set title and labels
        ax.set_title(f'Model: {model}')
        ax.set_xlabel('Dimension')
        if i == 0:  # Only set ylabel for the first subplot
            ax.set_ylabel('Average Score (0-3)')
        ax.set_ylim(0, 3)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Remove individual legends, save for combined legend
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    
    # Add a single legend at the top
    fig.legend(handles, labels, title='Evaluation Type', 
               loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)
    
    plt.suptitle('Self vs Cross-Model Evaluation', fontsize=16, y=1.12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def generate_model_radar(analyzer, model, save_path=None):
    """Generate a radar chart for a specific model."""
    df = analyzer.model_results[model]
    dimensions = analyzer.dimensions
    
    # Get score columns
    score_cols = []
    for dim in dimensions:
        col = f"scores.{dim} (0-3)"
        if col in df.columns:
            score_cols.append(col)
    
    # Calculate average scores
    avg_scores = df[score_cols].mean().values
    
    # Create radar chart
    labels = [col.replace('scores.', '').replace(' (0-3)', '') for col in score_cols]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    
    # Close the plot
    avg_scores = np.append(avg_scores, avg_scores[0])
    angles.append(angles[0])
    labels.append(labels[0])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, avg_scores, 'o-', linewidth=2, label=model)
    ax.fill(angles, avg_scores, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels[:-1])
    
    # Set y-axis
    ax.set_ylim(0, 3)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['1', '2', '3'])
    ax.grid(True)
    
    plt.title(f'{model} - Alignment Dimensions', fontsize=14)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def generate_model_dimension_bar(analyzer, model, save_path=None):
    """Generate a bar chart for dimension scores for a specific model."""
    df = analyzer.model_results[model]
    dimensions = analyzer.dimensions
    
    # Get score columns and values
    scores = []
    labels = []
    for dim in dimensions:
        col = f"scores.{dim} (0-3)"
        if col in df.columns:
            scores.append(df[col].mean())
            labels.append(dim)
    
    # Create bar chart
    plt.figure(figsize=(12, 8))
    bars = plt.bar(labels, scores, color='#3498db')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.title(f'{model} - Alignment Dimension Scores', fontsize=14)
    plt.xlabel('Dimension', fontsize=12)
    plt.ylabel('Average Score (0-3)', fontsize=12)
    plt.ylim(0, 3)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def generate_model_category_scores(analyzer, model, save_path=None):
    """Generate a bar chart for category scores for a specific model."""
    df = analyzer.model_results[model]
    dimensions = analyzer.dimensions
    
    if 'category' not in df.columns:
        print(f"No category column found for model {model}. Skipping category plot.")
        return
    
    # Get score columns
    score_cols = []
    for dim in dimensions:
        col = f"scores.{dim} (0-3)"
        if col in df.columns:
            score_cols.append(col)
    
    # Get categories
    categories = df['category'].unique()
    
    # Calculate average scores for each category
    cat_scores = []
    for cat in categories:
        cat_df = df[df['category'] == cat]
        avg_score = cat_df[score_cols].mean().mean()  # Average across all dimensions
        cat_scores.append({
            'Category': cat,
            'Score': avg_score
        })
    
    # Create dataframe
    plot_df = pd.DataFrame(cat_scores)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    bars = plt.bar(plot_df['Category'], plot_df['Score'], color='#2ecc71')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.title(f'{model} - Scores by Category', fontsize=14)
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Average Score (0-3)', fontsize=12)
    plt.ylim(0, 3)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def generate_perspective_plot(analyzer, model, save_path=None):
    """Generate a radar chart showing perspective drift for a specific model."""
    df = analyzer.model_results[model]
    dimensions = analyzer.dimensions
    
    if 'perspective' not in df.columns:
        print(f"No perspective column found for model {model}. Skipping perspective plot.")
        return
    
    # Get perspectives
    perspectives = df['perspective'].unique()
    
    # Get score columns
    score_cols = []
    for dim in dimensions:
        col = f"scores.{dim} (0-3)"
        if col in df.columns:
            score_cols.append(col)
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
    
    # Set up angles and labels
    labels = [col.replace('scores.', '').replace(' (0-3)', '') for col in score_cols]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    
    # Close the plot
    angles.append(angles[0])
    labels.append(labels[0])
    
    # Plot each perspective
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']  # Different color for each perspective
    for i, perspective in enumerate(perspectives):
        persp_df = df[df['perspective'] == perspective]
        scores = persp_df[score_cols].mean().values
        scores = np.append(scores, scores[0])  # Close the loop
        
        ax.plot(angles, scores, 'o-', linewidth=2, label=perspective, color=colors[i % len(colors)])
        ax.fill(angles, scores, alpha=0.1, color=colors[i % len(colors)])
    
    # Customize chart
    ax.set_thetagrids(np.degrees(angles[:-1]), labels[:-1])
    ax.set_ylim(0, 3)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['1', '2', '3'])
    ax.grid(True)
    
    plt.title(f'{model} - Perspective Analysis', fontsize=14)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def run_rlhf_demo(output_dir=None):
    """Run the RLHF demo to generate visualizations."""
    try:
        from src.demo_rlhf import run_demo
        
        # Create output directory if it doesn't exist
        if output_dir is None:
            output_dir = "results/rlhf_demo"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Run the demo with enhanced visualization output
        print("Running RLHF demo with sample prompts...")
        run_demo()
        
        # Check if the plots were generated
        dimension_plot = os.path.join(output_dir, "dimension_improvements.png")
        example_plot = os.path.join(output_dir, "example_improvement.png")
        
        if os.path.exists(dimension_plot):
            print(f"✓ Generated dimension improvements plot: {dimension_plot}")
        else:
            print("⨯ Failed to generate dimension improvements plot")
            
        if os.path.exists(example_plot):
            print(f"✓ Generated example improvement showcase: {example_plot}")
        else:
            print("⨯ Failed to generate example improvement showcase")
        
        # Copy training history to results for reference
        training_history = os.path.join(output_dir, "training_history.json")
        if os.path.exists(training_history):
            print(f"✓ Generated RLHF training history: {training_history}")
            
            # Create a human-readable summary from the training history
            try:
                with open(training_history, 'r') as f:
                    history = json.load(f)
                
                # Generate a summary markdown file
                summary_path = os.path.join(output_dir, "rlhf_summary.md")
                with open(summary_path, 'w') as f:
                    f.write("# RLHF Improvement Summary\n\n")
                    f.write(f"## Overview\n\n")
                    f.write(f"Total examples processed: {len(history)}\n\n")
                    
                    # Count improvements by category
                    categories = {}
                    dimensions_improved = {}
                    strategies_used = {}
                    
                    for entry in history:
                        # Track categories
                        category = entry.get('category', 'unknown')
                        if category not in categories:
                            categories[category] = {'count': 0, 'improved': 0}
                        categories[category]['count'] += 1
                        
                        # Check if improved
                        orig_scores = entry.get('original_scores', {})
                        impr_scores = entry.get('improved_scores', {})
                        
                        improved = False
                        for dim in orig_scores:
                            if dim in impr_scores:
                                change = impr_scores[dim] - orig_scores[dim]
                                
                                # Track dimension improvements
                                if dim not in dimensions_improved:
                                    dimensions_improved[dim] = []
                                dimensions_improved[dim].append(change)
                                
                                if change > 0:
                                    improved = True
                        
                        if improved:
                            categories[category]['improved'] += 1
                        
                        # Track strategies
                        for strategy in entry.get('improvements_applied', []):
                            if strategy not in strategies_used:
                                strategies_used[strategy] = 0
                            strategies_used[strategy] += 1
                    
                    # Write category summary
                    f.write("## Categories\n\n")
                    f.write("| Category | Processed | Improved | Rate |\n")
                    f.write("|----------|-----------|----------|------|\n")
                    for cat, stats in categories.items():
                        rate = (stats['improved'] / stats['count']) * 100 if stats['count'] > 0 else 0
                        f.write(f"| {cat} | {stats['count']} | {stats['improved']} | {rate:.1f}% |\n")
                    
                    f.write("\n## Dimension Improvements\n\n")
                    f.write("| Dimension | Avg Change | Max Improvement |\n")
                    f.write("|-----------|------------|----------------|\n")
                    for dim, changes in dimensions_improved.items():
                        avg = sum(changes) / len(changes)
                        max_impr = max(changes)
                        f.write(f"| {dim} | {avg:.3f} | {max_impr:.3f} |\n")
                    
                    f.write("\n## Strategies Used\n\n")
                    strategies_sorted = sorted(strategies_used.items(), key=lambda x: x[1], reverse=True)
                    for strategy, count in strategies_sorted:
                        readable = strategy.replace('_', ' ').title()
                        f.write(f"- {readable}: {count} times\n")
                
                print(f"✓ Generated human-readable RLHF summary: {summary_path}")
            except Exception as e:
                print(f"Error generating RLHF summary: {e}")
        
        return True
    except ImportError:
        print("Could not import RLHF demo. Make sure src/demo_rlhf.py exists.")
        return False
    except Exception as e:
        print(f"Error running RLHF demo: {e}")
        return False

def main():
    """Main entry point when script is run directly."""
    # Generate all plots
    success = generate_all_plots()
    
    if success:
        # Run RLHF demo
        run_rlhf_demo()
        
        print("\nAll plots and reports have been generated successfully!")
        print("You can now view the plots in the following directories:")
        print("- Comparison plots: results/plots/comparison/")
        print("- Model-specific plots: results/plots/model_specific/")
        print("- RLHF demo results: results/rlhf_demo/")
        print("\nYou can also run the dashboard to view all results:")
        print("streamlit run dashboard/streamlit_app.py")
    else:
        print("Failed to generate plots. Please check the error messages above.")

if __name__ == "__main__":
    main() 