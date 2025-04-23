from .base_evaluator import BaseEvaluator
from .factuality_checker import FactualityChecker
from typing import Dict, Optional, List, Any

class FactualityEvaluator(BaseEvaluator):
    """
    Evaluator that checks the factual accuracy of model responses against a reference knowledge base.
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.dimension = "factuality"
        self.scale_min = 1
        self.scale_max = 10
        self.knowledge_base_path = config.get("knowledge_base_path", "data/wikipedia_sample.json")
        self.checker = FactualityChecker(knowledge_base_path=self.knowledge_base_path)
    
    def evaluate(self, prompt: str, response: str, metadata: Dict = None) -> Dict:
        """
        Evaluate the factual accuracy of a response.
        
        Args:
            prompt: The prompt given to the model
            response: The model's response
            metadata: Additional metadata about the prompt-response pair
            
        Returns:
            Dictionary containing evaluation results
        """
        factuality_results = self.checker.check_factuality_tfidf(prompt, response)
        
        # Map factuality score (0-1) to standard evaluation scale (1-10)
        scaled_score = min(10, max(1, round(factuality_results["factuality_score"] * 9 + 1)))
        
        evaluation = {
            "dimension": "factuality",
            "score": scaled_score,
            "explanation": self._generate_explanation(factuality_results),
            "metadata": {
                "facts_evaluated": factuality_results["facts_evaluated"],
                "reference_entries": factuality_results.get("reference_entries", []),
                "raw_factuality_score": factuality_results["factuality_score"]
            }
        }
        
        if "error" in factuality_results:
            evaluation["metadata"]["error"] = factuality_results["error"]
            
        return evaluation
    
    def _generate_explanation(self, factuality_results: Dict) -> str:
        """Generate a human-readable explanation of the factuality evaluation."""
        
        if "error" in factuality_results:
            return f"Error evaluating factuality: {factuality_results['error']}"
            
        facts_evaluated = factuality_results["facts_evaluated"]
        fact_score = factuality_results["factuality_score"]
        
        if facts_evaluated == 0:
            return "Could not evaluate factuality - no evaluable facts found in response."
            
        # Count correct facts
        correct_facts = 0
        for fact in factuality_results.get("fact_evaluations", []):
            if fact.get("correct", False):
                correct_facts += 1
                
        explanation = f"Evaluated {facts_evaluated} factual statements. "
        explanation += f"Found {correct_facts} accurate statements "
        explanation += f"({fact_score:.0%} accurate). "
        
        # Add detail about references
        ref_entries = factuality_results.get("reference_entries", [])
        if ref_entries:
            topics = [entry.get("topic", "unknown") for entry in ref_entries]
            explanation += f"Referenced topics: {', '.join(topics)}."
        
        return explanation 