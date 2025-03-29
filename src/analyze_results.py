import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import List, Dict, Optional
import os

class AlignmentAnalyzer:
    """
    Analyzes and visualizes model alignment evaluation results.
    
    Novel contribution: Multi-dimensional alignment visualization and
    perspective drift analysis.
    """
    
    def __init__(self, results_file: str):
        """Initialize with results file (CSV or JSON)."""
        self.results_file = results_file
        self.results = self._load_results(results_file)
        self.dimensions = ["helpfulness", "harmlessness", "ethical_judgment", "honesty"]
        
    def _load_results(self, file_path: str) -> pd.DataFrame:
        """Load results from file."""
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
            return pd.json_normalize(data)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    def plot_dimension_scores(self, save_path: Optional[str] = None):
        """Plot average scores across evaluation dimensions."""
        # Extract scores and convert to numeric
        score_cols = [f"scores.{dim}" for dim in self.dimensions]
        
        # Calculate average scores by category
        avg_scores = self.results.groupby('category')[score_cols].mean().reset_index()
        
        # Reshape for seaborn
        avg_scores_melted = pd.melt(
            avg_scores, 
            id_vars=['category'], 
            value_vars=score_cols,
            var_name='dimension', 
            value_name='score'
        )
        
        # Clean dimension names
        avg_scores_melted['dimension'] = avg_scores_melted['dimension'].apply(
            lambda x: x.replace('scores.', '')
        )
        
        # Create plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x='category', y='score', hue='dimension', data=avg_scores_melted)
        plt.title('Average Alignment Scores by Category')
        plt.xlabel('Prompt Category')
        plt.ylabel('Score (0-3)')
        plt.ylim(0, 3)
        plt.legend(title='Dimension')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        return plt
    
    def plot_perspective_drift(self, drift_data: Dict, save_path: Optional[str] = None):
        """
        Plot how model alignment drifts across different perspectives.
        
        A novel visualization approach.
        """
        # Convert drift data to DataFrame
        perspectives = list(drift_data['perspective_results'].keys())
        dimensions = self.dimensions
        
        # Prepare data structure
        data = []
        for perspective in perspectives:
            for dimension in dimensions:
                score = drift_data['perspective_results'][perspective]['scores'].get(dimension, 0)
                data.append({
                    'perspective': perspective,
                    'dimension': dimension,
                    'score': score
                })
        
        df = pd.DataFrame(data)
        
        # Create radar/spider chart
        plt.figure(figsize=(10, 8))
        
        # Count unique categories
        n_dims = len(dimensions)
        n_perspectives = len(perspectives)
        
        # Create angles for the radar plot
        angles = [n / float(n_dims) * 2 * 3.14159 for n in range(n_dims)]
        angles += angles[:1]  # Close the loop
        
        # Create plot
        ax = plt.subplot(111, polar=True)
        
        # Draw dimension labels
        plt.xticks(angles[:-1], dimensions, size=12)
        
        # Draw ylabels (scores)
        ax.set_rlabel_position(0)
        plt.yticks([1, 2, 3], ["1", "2", "3"], color="grey", size=10)
        plt.ylim(0, 3)
        
        # Plot data
        for i, perspective in enumerate(perspectives):
            perspective_data = df[df['perspective'] == perspective]
            values = perspective_data.set_index('dimension')['score'].reindex(dimensions).values.flatten().tolist()
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=perspective)
            ax.fill(angles, values, alpha=0.1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title("Alignment Across Perspectives", size=15)
        
        if save_path:
            plt.savefig(save_path)
            
        return plt
    
    def generate_report(self, output_dir: str = "results"):
        """Generate a comprehensive markdown report with visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate visualizations
        dimension_plot_path = os.path.join(output_dir, "dimension_scores.png")
        self.plot_dimension_scores(save_path=dimension_plot_path)
        
        # Create markdown report
        report = f"""# Alignment Evaluation Report

## Overview
- Total prompts evaluated: {len(self.results)}
- Categories: {', '.join(self.results['category'].unique())}
- Model(s): {', '.join(self.results['model_name'].unique() if 'model_name' in self.results.columns else ['unknown'])}

## Alignment Scores

![Dimension Scores]({os.path.basename(dimension_plot_path)})

## Key Findings

### By Category
"""
        
        # Add category insights
        for category in self.results['category'].unique():
            category_results = self.results[self.results['category'] == category]
            flags = [f for sublist in category_results['flags'].dropna() for f in sublist if isinstance(sublist, list)]
            
            report += f"""
### {category.title()}
- Average scores: {', '.join([f"{dim}: {category_results[f'scores.{dim}'].mean():.2f}" for dim in self.dimensions])}
- Flag frequency: {', '.join(f"{f}: {flags.count(f)}" for f in set(flags)) if flags else "No flags"}
"""
        
        # Write report to file
        report_path = os.path.join(output_dir, "analysis_report.md")
        with open(report_path, 'w') as f:
            f.write(report)
            
        return report_path 