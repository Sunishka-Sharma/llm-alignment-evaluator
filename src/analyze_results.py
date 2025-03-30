import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional

class AlignmentAnalyzer:
    """
    Analyzes and visualizes model alignment evaluation results.
    
    Novel contribution: Multi-dimensional alignment visualization and
    perspective drift analysis.
    """
    
    def __init__(self, results_file: str = None):
        """Initialize with optional results file (CSV or JSON)."""
        self.dimensions = ["helpfulness", "harmlessness", "ethical_judgment", "honesty"]
        self.model_results = {}
        if results_file:
            self.add_model_results(results_file)
        
    def add_model_results(self, results_file: str) -> None:
        """Add results for a model from file."""
        model_name = os.path.basename(os.path.dirname(results_file))
        self.model_results[model_name] = self._load_results(results_file)
        
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
        """Plot average scores across evaluation dimensions for all models."""
        plt.figure(figsize=(12, 8))
        
        models = list(self.model_results.keys())
        x = np.arange(len(self.dimensions))
        width = 0.8 / len(models)
        
        for i, model in enumerate(models):
            df = self.model_results[model]
            score_cols = [f"scores.{dim}" for dim in self.dimensions]
            avg_scores = df[score_cols].mean()
            
            plt.bar(x + i * width, avg_scores, width, label=model)
        
        plt.title('Average Alignment Scores by Model')
        plt.xlabel('Dimension')
        plt.ylabel('Score (0-3)')
        plt.xticks(x + width * (len(models) - 1) / 2, self.dimensions)
        plt.ylim(0, 3)
        plt.legend(title='Model')
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
    
    def generate_comparative_report(self, report_path: str) -> None:
        """Generate a comparative analysis report."""
        report = ["# Model Alignment Comparative Analysis\n"]
        
        # Overview section
        report.append("## Overview")
        models = list(self.model_results.keys())
        report.append(f"Total models evaluated: {len(models)}")
        report.append(f"Models: {', '.join(models)}\n")
        
        # Scores comparison
        report.append("## Alignment Scores Comparison")
        report.append("![Dimension Scores](plots/dimension_scores.png)\n")
        
        # Model-specific analysis
        report.append("## Model-Specific Analysis\n")
        for model in models:
            df = self.model_results[model]
            score_cols = [col for col in df.columns if col.startswith('scores.')]
            overall_score = df[score_cols].mean().mean()
            
            report.append(f"\n### {model}")
            report.append(f"- Overall alignment score: {overall_score:.2f}/3")
            report.append("- Dimension scores:")
            for col in score_cols:
                dim = col.replace('scores.', '')
                score = df[col].mean()
                report.append(f"  - {dim}: {score:.2f}/3")
            
            report.append("\nPerformance by category:")
            for cat in df['category'].unique():
                cat_score = df[df['category'] == cat][score_cols].mean().mean()
                report.append(f"- {cat}: {cat_score:.2f}/3")
        
        # Comparative insights if we have multiple models
        if len(models) > 1:
            report.append("\n## Comparative Insights")
            report.append("\nStrengths and differences:")
            
            # Compare models on each dimension
            for col in score_cols:
                dim = col.replace('scores.', '')
                report.append(f"\n### {dim.title()}")
                scores = {model: self.model_results[model][col].mean() for model in models}
                better_model = max(scores.items(), key=lambda x: x[1])[0]
                other_models = [m for m in models if m != better_model]
                
                for other_model in other_models:
                    score_diff = scores[better_model] - scores[other_model]
                    if score_diff > 0.3:  # Significant difference
                        report.append(f"- {better_model} shows notably better {dim} ({scores[better_model]:.2f} vs {scores[other_model]:.2f})")
                    else:
                        report.append(f"- Both models perform similarly in {dim} ({scores[better_model]:.2f} vs {scores[other_model]:.2f})")
        
        # Write report
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
            
        return report_path 