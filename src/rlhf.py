import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import re
from textblob import TextBlob
import spacy
import language_tool_python

class RewardModel:
    """Enhanced reward model for alignment learning."""
    
    def __init__(self):
        self.feedback_data = []
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3
        )
        self.scaler = StandardScaler()
        self.nlp = spacy.load('en_core_web_sm')
        self.language_tool = language_tool_python.LanguageTool('en-US')
        self.feature_names = [
            'length', 'avg_word_length', 'sentence_count',
            'refusal_score', 'safety_score', 'complexity_score',
            'ethical_terms', 'perspective_terms', 'clarity_score',
            'sentiment_score', 'subjectivity_score', 'grammar_error_count',
            'entity_count', 'unique_entity_types', 'avg_sentence_complexity'
        ]
    
    def collect_feedback(self, response: str, human_rating: float, 
                        explanation: str = "", metadata: Optional[Dict] = None) -> None:
        """Collect human feedback with optional metadata."""
        self.feedback_data.append({
            "response": response,
            "rating": human_rating,
            "explanation": explanation,
            "metadata": metadata or {}
        })
        logging.info(f"Collected feedback: Rating {human_rating}")
    
    def extract_linguistic_features(self, response: str) -> Dict:
        """Extract sophisticated linguistic features."""
        # Basic NLP analysis
        doc = self.nlp(response)
        blob = TextBlob(response)
        
        # Grammar check
        grammar_errors = self.language_tool.check(response)
        
        # Entity analysis
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        unique_entity_types = len(set(ent.label_ for ent in doc.ents))
        
        # Sentence complexity (based on dependency tree depth)
        sentence_complexities = []
        for sent in doc.sents:
            depths = [len(list(token.ancestors)) for token in sent]
            sentence_complexities.append(max(depths) if depths else 0)
        
        return {
            'sentiment_score': blob.sentiment.polarity,
            'subjectivity_score': blob.sentiment.subjectivity,
            'grammar_error_count': len(grammar_errors),
            'entity_count': len(entities),
            'unique_entity_types': unique_entity_types,
            'avg_sentence_complexity': np.mean(sentence_complexities) if sentence_complexities else 0
        }
    
    def extract_features(self, response: str) -> np.ndarray:
        """Extract comprehensive features from response."""
        if not response:
            return np.zeros(len(self.feature_names))
        
        # Get basic features
        words = response.split()
        sentences = re.split(r'[.!?]+', response)
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        
        # Get linguistic features
        ling_features = self.extract_linguistic_features(response)
        
        # Pattern matching features
        refusal_terms = ["cannot", "won't", "not", "unable", "decline"]
        safety_terms = ["safe", "careful", "protect", "prevent", "avoid"]
        ethical_terms = ["ethical", "moral", "right", "wrong", "should", "consider"]
        perspective_terms = ["however", "although", "perspective", "view", "opinion"]
        
        features = [
            len(response),  # Length
            avg_word_len,  # Average word length
            len(sentences),  # Sentence count
            sum(1 for term in refusal_terms if term in response.lower()),  # Refusal score
            sum(1 for term in safety_terms if term in response.lower()),  # Safety score
            len(re.findall(r'\b\w{7,}\b', response)),  # Complexity score
            sum(1 for term in ethical_terms if term in response.lower()),  # Ethical terms
            sum(1 for term in perspective_terms if term in response.lower()),  # Perspective terms
            self._calculate_clarity_score(response),  # Clarity score
            ling_features['sentiment_score'],
            ling_features['subjectivity_score'],
            ling_features['grammar_error_count'],
            ling_features['entity_count'],
            ling_features['unique_entity_types'],
            ling_features['avg_sentence_complexity']
        ]
        
        return np.array(features)
    
    def _calculate_clarity_score(self, response: str) -> float:
        """Calculate clarity score based on readability metrics."""
        if not response:
            return 0.0
            
        words = response.split()
        sentences = re.split(r'[.!?]+', response)
        if not words or not sentences:
            return 0.0
            
        # Enhanced readability calculation with more factors
        avg_sentence_length = len(words) / max(len(sentences), 1)
        avg_word_length = np.mean([len(w) for w in words])
        long_words_ratio = len([w for w in words if len(w) > 6]) / max(len(words), 1)
        
        # Factors that reduce clarity:
        # 1. Long sentences (optimal is 15-20 words)
        sentence_length_penalty = max(0, (avg_sentence_length - 20) / 20)
        
        # 2. Long words
        word_length_penalty = max(0, (avg_word_length - 5) / 5)
        
        # 3. Ratio of long words
        complexity_penalty = long_words_ratio
        
        # Calculate base score (1.0 is perfect clarity)
        base_score = 1.0
        
        # Apply penalties
        final_score = base_score - (0.3 * sentence_length_penalty) - (0.3 * word_length_penalty) - (0.4 * complexity_penalty)
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, final_score))
    
    def train(self) -> None:
        """Train a gradient boosting reward model."""
        if len(self.feedback_data) < 3:  # Reduced minimum requirement
            logging.warning("Not enough feedback data to train")
            return
            
        X = np.array([self.extract_features(d["response"]) 
                     for d in self.feedback_data])
        y = np.array([d["rating"] for d in self.feedback_data])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model with adjusted parameters for small datasets
        self.model = GradientBoostingRegressor(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=2,
            min_samples_split=2
        )
        self.model.fit(X_scaled, y)
        
        # Log feature importances
        importances = dict(zip(self.feature_names, self.model.feature_importances_))
        logging.info(f"Feature importances: {importances}")
        logging.info(f"Trained reward model on {len(self.feedback_data)} samples")
    
    def predict_reward(self, response: str) -> float:
        """Predict reward for a response using trained model."""
        if not hasattr(self.model, 'feature_importances_'):
            return 0.0
            
        features = self.extract_features(response)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        return float(self.model.predict(features_scaled)[0])

class RLHFTrainer:
    """Enhanced RLHF training loop."""
    
    def __init__(self, reward_model: RewardModel):
        self.reward_model = reward_model
        self.training_history = []
        self.improvement_strategies = [
            self._add_ethical_consideration,
            self._add_safety_context,
            self._add_perspective_analysis,
            self._improve_clarity,
            self._add_examples,
            self._add_detailed_explanation
        ]
    
    def _add_ethical_consideration(self, response: str) -> str:
        """Add ethical considerations to response."""
        addition = "\n\nFrom an ethical perspective, it's important to consider the implications and potential impacts on all stakeholders involved. This includes examining the moral principles at play and potential consequences of different actions."
        return response + addition
    
    def _add_safety_context(self, response: str) -> str:
        """Add safety context to response."""
        addition = "\n\nIt's crucial to prioritize safety and well-being in this context. Consider implementing appropriate safeguards and preventive measures to minimize potential risks."
        return "I want to ensure this response prioritizes safety and well-being: " + response + addition
    
    def _add_perspective_analysis(self, response: str) -> str:
        """Add multiple perspective analysis."""
        addition = "\n\nThis situation can be viewed from multiple perspectives:\n1. Individual perspective: ...\n2. Societal perspective: ...\n3. Long-term perspective: ..."
        return response + addition
    
    def _improve_clarity(self, response: str) -> str:
        """Improve response clarity."""
        # For very long responses, try to make it more concise
        if len(response) > 1000:
            return "Let me provide a clear and concise summary:\n\n" + response
        return "Let me explain this clearly and systematically:\n\n" + response
    
    def _add_examples(self, response: str) -> str:
        """Add concrete examples."""
        addition = "\n\nHere are some concrete examples to illustrate:\n1. First scenario: ...\n2. Second scenario: ...\n3. Practical application: ..."
        return response + addition
    
    def _add_detailed_explanation(self, response: str) -> str:
        """Add detailed explanation with structured points."""
        addition = "\n\nTo elaborate further:\n1. Key principles:\n   - Point 1\n   - Point 2\n2. Important considerations:\n   - Factor 1\n   - Factor 2\n3. Practical implications:\n   - Impact 1\n   - Impact 2"
        return response + addition
    
    def improve_response(self, original_response: str, prompt: str = "", category: str = "",
                        max_attempts: int = 5,
                        min_improvement: float = 0.1) -> Tuple[str, Dict]:
        """
        Improve response using multiple strategies and reward prediction.
        
        Args:
            original_response: The original model response
            prompt: The original prompt
            category: The category of the prompt
            max_attempts: Maximum improvement attempts
            min_improvement: Minimum improvement threshold
            
        Returns:
            Tuple of (improved response, reward scores dict)
        """
        # Handle empty response
        if not original_response:
            best_response = "I understand you're looking for assistance. Let me help you with a clear and helpful response that addresses your needs effectively."
            if hasattr(self.reward_model, 'predict_rewards'):
                return best_response, self.reward_model.predict_rewards(best_response, prompt, category)
            else:
                return best_response, {"overall": self.reward_model.predict_reward(best_response)}
            
        # Get baseline scores
        if hasattr(self.reward_model, 'predict_rewards'):
            # For multidimensional model
            original_scores = self.reward_model.predict_rewards(original_response, prompt, category)
            best_scores = original_scores.copy()
            best_overall = sum(best_scores.values()) / len(best_scores) if best_scores else 0
        else:
            # For single-dimensional model
            original_scores = {"overall": self.reward_model.predict_reward(original_response)}
            best_scores = original_scores.copy()
            best_overall = best_scores["overall"]
            
        best_response = original_response
        improvements_tried = []
        
        # For very long responses, first try to improve clarity
        if len(original_response) > 1000:
            improved_response = self._improve_clarity(original_response)
            
            if hasattr(self.reward_model, 'predict_rewards'):
                improved_scores = self.reward_model.predict_rewards(improved_response, prompt, category)
                improved_overall = sum(improved_scores.values()) / len(improved_scores)
            else:
                improved_scores = {"overall": self.reward_model.predict_reward(improved_response)}
                improved_overall = improved_scores["overall"]
                
            if improved_overall > best_overall:
                best_response = improved_response
                best_scores = improved_scores
                best_overall = improved_overall
        
        for _ in range(max_attempts):
            # Try each improvement strategy
            candidates = [
                strategy(best_response)
                for strategy in self.improvement_strategies
                if strategy.__name__ not in improvements_tried
            ]
            
            # Skip if no new strategies to try
            if not candidates:
                break
                
            # Evaluate candidates
            if hasattr(self.reward_model, 'predict_rewards'):
                # For multidimensional model
                candidate_scores = [self.reward_model.predict_rewards(c, prompt, category) for c in candidates]
                candidate_overalls = [sum(s.values()) / len(s) for s in candidate_scores]
            else:
                # For single-dimensional model  
                candidate_overalls = [self.reward_model.predict_reward(c) for c in candidates]
                candidate_scores = [{"overall": score} for score in candidate_overalls]
            
            # Find best improvement
            if candidate_overalls:
                max_reward_idx = np.argmax(candidate_overalls)
                if candidate_overalls[max_reward_idx] > best_overall + min_improvement:
                    best_response = candidates[max_reward_idx]
                    best_scores = candidate_scores[max_reward_idx]
                    best_overall = candidate_overalls[max_reward_idx]
                    improvements_tried.append(
                        self.improvement_strategies[max_reward_idx].__name__
                    )
                else:
                    # If no improvement found, force at least one addition for long responses
                    if len(best_response) >= len(original_response) and not improvements_tried:
                        best_response = best_response + "\n\nAdditional insights and considerations: ..."
                        if hasattr(self.reward_model, 'predict_rewards'):
                            best_scores = self.reward_model.predict_rewards(best_response, prompt, category)
                        else:
                            best_scores = {"overall": self.reward_model.predict_reward(best_response)}
                    break
            else:
                break
        
        # Record improvement history
        self.training_history.append({
            "prompt": prompt,
            "category": category,
            "original_response": original_response,
            "improved_response": best_response,
            "original_scores": original_scores,
            "improved_scores": best_scores,
            "improvements_applied": improvements_tried
        })
        
        return best_response, best_scores
        
    def analyze_improvements(self) -> Dict:
        """
        Analyze the improvements made by the trainer.
        
        Returns:
            Dictionary containing improvement statistics
        """
        if not self.training_history:
            return {"error": "No training history available"}
            
        total_responses = len(self.training_history)
        
        # Calculate overall stats
        improved_responses = 0
        total_improvement = 0
        
        # For multidimensional analysis
        dimension_improvements = {}
        
        # Analyze strategy effectiveness
        strategy_effectiveness = {}
        
        for entry in self.training_history:
            original_scores = entry["original_scores"]
            improved_scores = entry["improved_scores"]
            
            # Calculate improvement for each dimension
            if "overall" in original_scores:
                # Single dimension case
                improvement = improved_scores["overall"] - original_scores["overall"]
                if improvement > 0:
                    improved_responses += 1
                    total_improvement += improvement
            else:
                # Multi-dimension case
                dimensions = set(original_scores.keys()) & set(improved_scores.keys())
                
                any_improvement = False
                for dim in dimensions:
                    dim_improvement = improved_scores[dim] - original_scores[dim]
                    
                    if dim not in dimension_improvements:
                        dimension_improvements[dim] = []
                    
                    dimension_improvements[dim].append(dim_improvement)
                    
                    if dim_improvement > 0:
                        any_improvement = True
                
                if any_improvement:
                    improved_responses += 1
                    
                avg_improvement = sum(improved_scores[d] - original_scores[d] 
                                     for d in dimensions) / len(dimensions)
                total_improvement += avg_improvement
            
            # Analyze strategy effectiveness
            for strategy in entry.get("improvements_applied", []):
                if strategy not in strategy_effectiveness:
                    strategy_effectiveness[strategy] = []
                
                if "overall" in improved_scores:
                    strategy_effectiveness[strategy].append(
                        improved_scores["overall"] - original_scores["overall"]
                    )
                else:
                    dimensions = set(original_scores.keys()) & set(improved_scores.keys())
                    if dimensions:
                        avg_improvement = sum(improved_scores[d] - original_scores[d] 
                                            for d in dimensions) / len(dimensions)
                        strategy_effectiveness[strategy].append(avg_improvement)
        
        # Calculate averages
        improvement_rate = improved_responses / total_responses if total_responses > 0 else 0
        avg_improvement = total_improvement / total_responses if total_responses > 0 else 0
        
        # Compute average dimension improvements
        avg_dimension_improvements = {}
        for dim, improvements in dimension_improvements.items():
            avg_dimension_improvements[dim] = sum(improvements) / len(improvements) if improvements else 0
        
        # Compute strategy effectiveness
        avg_strategy_effectiveness = []
        for strategy, improvements in strategy_effectiveness.items():
            if improvements:
                avg_gain = sum(improvements) / len(improvements)
                avg_strategy_effectiveness.append((strategy, avg_gain))
        
        # Sort by effectiveness
        avg_strategy_effectiveness.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "total_responses": total_responses,
            "improved_responses": improved_responses,
            "improvement_rate": improvement_rate,
            "avg_improvement": avg_improvement,
            "dimensions_improved": avg_dimension_improvements,
            "strategy_effectiveness": avg_strategy_effectiveness
        }
    
    def export_training_history(self, output_path: str) -> None:
        """
        Export training history to JSON for analysis.
        
        Args:
            output_path: Path to save the training history
        """
        import json
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Prepare data for export (with samples limited for space)
        export_data = {
            "examples": self.training_history[:10],  # First 10 examples
            "statistics": self.analyze_improvements()
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

class MultiDimensionalRewardModel(RewardModel):
    """
    Enhanced reward model that supports multidimensional scoring across alignment dimensions.
    This extends the base RewardModel to provide dimension-specific training and prediction.
    
    Dimensions include:
    - helpfulness (0-3)
    - harmlessness (0-3)
    - ethical_judgment (0-3) 
    - honesty (0-3)
    """
    
    def __init__(self):
        super().__init__()
        # Create separate models for each dimension
        self.dimensions = ["helpfulness", "harmlessness", "ethical_judgment", "honesty"]
        self.dimension_models = {
            dim: GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3
            ) for dim in self.dimensions
        }
        self.dimension_scalers = {dim: StandardScaler() for dim in self.dimensions}
        self.dimension_feedback = {dim: [] for dim in self.dimensions}
        
        # Cross-model evaluation data
        self.cross_eval_data = []
        
        # Dimension mapping from cross-evaluation to alignment dimensions
        self.cross_eval_mapping = {
            "helpfulness": "helpfulness",
            "safety": "harmlessness", 
            "ethical_judgment": "ethical_judgment",
            "clarity": "honesty"
        }
    
    def collect_feedback(self, response: str, ratings: Dict, prompt: str = "", category: str = ""):
        """
        Collect feedback with ratings for multiple dimensions.
        
        Args:
            response: The model response
            ratings: Dictionary of dimension-specific ratings
            prompt: The original prompt (optional)
            category: The category of the prompt (optional)
        """
        features = self.extract_features(response)
        
        for dim in self.dimensions:
            if dim in ratings:
                self.dimension_feedback[dim].append({
                    "response": response,
                    "rating": ratings[dim],
                    "features": features,
                    "prompt": prompt,
                    "category": category,
                    "metadata": {"dimension": dim}
                })
        
        # Also collect for the overall reward model
        overall_rating = sum(ratings.values()) / len(ratings) if ratings else 0
        self.feedback_data.append({
            "response": response,
            "rating": overall_rating,
            "features": features,
            "explanation": "",
            "metadata": {
                "prompt": prompt, 
                "category": category,
                "dimensions": ratings
            }
        })
    
    def add_cross_model_evaluation(self, cross_eval_data: List[Dict]) -> None:
        """
        Add cross-model evaluation data to enhance the reward model.
        
        Args:
            cross_eval_data: List of cross-evaluation results from other models
                Each item should be a dict with prompt, response, category, 
                and dimension-specific evaluations
        
        This allows the reward model to leverage evaluations from different models,
        combining self-evaluation with external perspectives for more robust scoring.
        """
        if not cross_eval_data:
            logging.warning("No cross-model evaluation data provided")
            return
            
        logging.info(f"Adding {len(cross_eval_data)} cross-model evaluations to reward model")
        
        for entry in cross_eval_data:
            if not isinstance(entry, dict):
                continue
                
            # Extract necessary data
            response = entry.get("response", "")
            prompt = entry.get("prompt", "")
            category = entry.get("category", "")
            evaluator = entry.get("evaluator", "unknown_model")
            
            # Skip if missing essential data
            if not response or not prompt:
                continue
                
            # Extract ratings from cross-evaluation
            ratings = {}
            for cross_dim, align_dim in self.cross_eval_mapping.items():
                if cross_dim in entry and "score" in entry[cross_dim]:
                    ratings[align_dim] = entry[cross_dim]["score"]
            
            # Skip if no valid ratings
            if not ratings:
                continue
                
            # Extract features
            features = self.extract_features(response)
            
            # Store the cross-evaluation data
            self.cross_eval_data.append({
                "response": response,
                "ratings": ratings,
                "features": features,
                "prompt": prompt,
                "category": category,
                "evaluator": evaluator
            })
            
            # Add to dimension-specific feedback with weight factor
            weight = 0.8  # Slightly lower weight than direct feedback
            for dim, rating in ratings.items():
                if dim in self.dimensions:
                    self.dimension_feedback[dim].append({
                        "response": response,
                        "rating": rating,
                        "features": features,
                        "prompt": prompt,
                        "category": category,
                        "metadata": {
                            "dimension": dim,
                            "source": "cross_eval",
                            "evaluator": evaluator,
                            "weight": weight
                        }
                    })
            
            # Also add to overall model with averaged score
            overall_rating = sum(ratings.values()) / len(ratings) if ratings else 0
            self.feedback_data.append({
                "response": response,
                "rating": overall_rating,
                "features": features,
                "explanation": "",
                "metadata": {
                    "prompt": prompt,
                    "category": category,
                    "dimensions": ratings,
                    "source": "cross_eval",
                    "evaluator": evaluator,
                    "weight": weight
                }
            })
            
        logging.info(f"Added {len(self.cross_eval_data)} cross-model evaluations to reward model")
        return len(self.cross_eval_data)
    
    def train(self):
        """Train dimension-specific models and overall model."""
        results = {}
        
        # Train dimension-specific models
        for dim in self.dimensions:
            if len(self.dimension_feedback[dim]) >= 5:  # Need at least 5 samples
                features = np.array([f["features"] for f in self.dimension_feedback[dim]])
                ratings = np.array([f["rating"] for f in self.dimension_feedback[dim]])
                
                # Scale features
                self.dimension_scalers[dim].fit(features)
                scaled_features = self.dimension_scalers[dim].transform(features)
                
                # Train model
                self.dimension_models[dim].fit(scaled_features, ratings)
                results[dim] = {
                    "samples": len(ratings),
                    "feature_importance": dict(zip(
                        self.feature_names,
                        self.dimension_models[dim].feature_importances_
                    ))
                }
            else:
                results[dim] = {"samples": len(self.dimension_feedback[dim]), "error": "Insufficient samples"}
        
        # Also train the overall model
        super().train()
        
        return results
    
    def predict_rewards(self, response: str, prompt: str = "", category: str = "") -> Dict:
        """
        Predict rewards for each dimension.
        
        Args:
            response: The model response to evaluate
            prompt: The original prompt (optional)
            category: The category of the prompt (optional)
            
        Returns:
            Dictionary of dimension-specific reward predictions
        """
        features = self.extract_features(response)
        rewards = {}
        
        for dim in self.dimensions:
            if hasattr(self.dimension_models[dim], 'feature_importances_'):
                # Scale features
                scaled_features = self.dimension_scalers[dim].transform([features])
                
                # Predict reward
                rewards[dim] = float(self.dimension_models[dim].predict(scaled_features)[0])
            else:
                # If model not trained, use a rule-based fallback
                if dim == "helpfulness":
                    rewards[dim] = min(len(response) / 500, 3) if len(response) > 100 else 1.0
                elif dim == "harmlessness":
                    rewards[dim] = 3.0 if not any(term in response.lower() for term in ["kill", "hurt", "harm", "illegal"]) else 1.0
                elif dim == "ethical_judgment":
                    rewards[dim] = 2.0 if any(term in response.lower() for term in ["ethical", "consider", "moral"]) else 1.5
                else:
                    rewards[dim] = 2.0  # Default
        
        return rewards
    
    def detect_edge_cases(self, response: str, prompt: str = "", category: str = "") -> List[str]:
        """
        Detect potential edge cases or concerning patterns in responses.
        
        Args:
            response: The model response to evaluate
            prompt: The original prompt
            category: The category of the prompt
            
        Returns:
            List of identified issues
        """
        issues = []
        
        # Safety issues
        if category == "safety" and any(term in response.lower() for term in ["here's how", "you can", "first step"]):
            issues.append("potential_harmful_instruction")
            
        # Helpfulness issues
        if category == "helpfulness" and len(response) < 50:
            issues.append("insufficient_detail")
            
        # Ethics issues
        if category == "ethics" and "yes" in response.lower()[:20] and not any(term in response.lower() for term in ["however", "consider", "complex"]):
            issues.append("oversimplified_ethics")
            
        # Bias issues
        if category == "bias" and any(stereo in response.lower() for stereo in ["men are better", "women are better", "inherently"]):
            issues.append("stereotypical_bias")
            
        return issues

def example_usage():
    """Example usage of the enhanced RLHF system."""
    # Initialize
    reward_model = RewardModel()
    trainer = RLHFTrainer(reward_model)
    
    # Collect diverse feedback examples
    responses = [
        "I cannot help with harmful activities.",
        "Here's a detailed explanation that considers multiple perspectives...",
        "Let me assist you while ensuring safety and ethical considerations.",
        "This is a complex issue that requires careful analysis.",
        "Here are some concrete examples to illustrate the concept..."
    ]
    ratings = [2.5, 3.0, 2.8, 2.7, 2.9]
    explanations = [
        "Clear safety boundary",
        "Thorough and balanced",
        "Safety-conscious and helpful",
        "Shows good judgment",
        "Practical and clear"
    ]
    
    # Collect feedback with metadata
    for resp, rating, expl in zip(responses, ratings, explanations):
        reward_model.collect_feedback(
            resp, rating, expl,
            metadata={"length": len(resp), "has_examples": "example" in resp.lower()}
        )
    
    # Train the reward model
    reward_model.train()
    
    # Try to improve a response
    original = "I can help with that task."
    improved, reward = trainer.improve_response(original)
    print(f"Original: {original}")
    print(f"Improved: {improved}")
    print(f"Reward improvement: {reward - reward_model.predict_reward(original):.2f}") 