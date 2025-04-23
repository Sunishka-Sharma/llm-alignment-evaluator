"""
Base Evaluator Module

This module defines the BaseEvaluator abstract class that all specialized evaluators 
should inherit from. It establishes a common interface for evaluation and reporting.
"""

from typing import Dict, Optional, List, Any
import logging
import json
import os
from abc import ABC, abstractmethod

class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluators.
    
    Provides a common interface for evaluating model responses across
    different dimensions like factuality, cultural sensitivity, etc.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize evaluator with configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.results = []
        self.dimension = "base"  # Should be overridden by subclasses
        self.scale_min = 0
        self.scale_max = 10
        
        # Configure logging
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(levelname)s - %(message)s')
    
    @abstractmethod
    def evaluate(self, prompt: str, response: str, metadata: Optional[Dict] = None) -> Dict:
        """
        Evaluate a model response.
        
        Args:
            prompt: The prompt given to the model
            response: The model's response
            metadata: Additional metadata about the prompt-response pair
            
        Returns:
            Dictionary containing evaluation results
        """
        pass
    
    def batch_evaluate(self, prompts: List[Dict], responses: List[str]) -> List[Dict]:
        """
        Evaluate multiple responses in batch.
        
        Args:
            prompts: List of prompt dictionaries
            responses: List of model responses
            
        Returns:
            List of evaluation results
        """
        if len(prompts) != len(responses):
            raise ValueError("Number of prompts and responses must match")
            
        results = []
        for i, (prompt_data, response) in enumerate(zip(prompts, responses)):
            prompt = prompt_data.get("prompt", "")
            metadata = {k: v for k, v in prompt_data.items() if k != "prompt"}
            
            try:
                result = self.evaluate(prompt, response, metadata)
                results.append(result)
                self.results.append(result)
            except Exception as e:
                logging.error(f"Error evaluating prompt {i}: {str(e)}")
        
        return results
    
    def export_results(self, output_path: str) -> None:
        """
        Export evaluation results to a file.
        
        Args:
            output_path: Path to save results
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            logging.info(f"Exported {len(self.results)} evaluation results to {output_path}")
        except Exception as e:
            logging.error(f"Error exporting results to {output_path}: {str(e)}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Calculate summary statistics for evaluation results.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.results:
            return {
                "dimension": self.dimension,
                "num_evaluations": 0,
                "avg_score": None,
                "min_score": None,
                "max_score": None
            }
            
        scores = [r.get("score", 0) for r in self.results if "score" in r]
        
        return {
            "dimension": self.dimension,
            "num_evaluations": len(self.results),
            "avg_score": sum(scores) / len(scores) if scores else None,
            "min_score": min(scores) if scores else None,
            "max_score": max(scores) if scores else None,
            "distribution": self._calculate_score_distribution(scores)
        }
    
    def _calculate_score_distribution(self, scores: List[float]) -> Dict[str, int]:
        """
        Calculate distribution of scores across scale.
        
        Args:
            scores: List of evaluation scores
            
        Returns:
            Dictionary with score distribution
        """
        distribution = {}
        for i in range(self.scale_min, self.scale_max + 1):
            distribution[str(i)] = sum(1 for s in scores if round(s) == i)
        
        return distribution 