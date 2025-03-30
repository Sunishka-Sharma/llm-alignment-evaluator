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
    
    def improve_response(self, original_response: str, 
                        max_attempts: int = 5,
                        min_improvement: float = 0.1) -> Tuple[str, float]:
        """
        Improve response using multiple strategies and reward prediction.
        """
        # Handle empty response
        if not original_response:
            best_response = "I understand you're looking for assistance. Let me help you with a clear and helpful response that addresses your needs effectively."
            return best_response, self.reward_model.predict_reward(best_response)
            
        best_response = original_response
        best_reward = self.reward_model.predict_reward(original_response)
        
        improvements_tried = []
        
        # For very long responses, first try to improve clarity
        if len(original_response) > 1000:
            best_response = self._improve_clarity(original_response)
            best_reward = self.reward_model.predict_reward(best_response)
        
        for _ in range(max_attempts):
            # Try each improvement strategy
            candidates = [
                strategy(best_response)
                for strategy in self.improvement_strategies
                if strategy.__name__ not in improvements_tried
            ]
            
            # Evaluate candidates
            rewards = [self.reward_model.predict_reward(c) for c in candidates]
            
            # Find best improvement
            if rewards:
                max_reward_idx = np.argmax(rewards)
                if rewards[max_reward_idx] > best_reward + min_improvement:
                    best_response = candidates[max_reward_idx]
                    best_reward = rewards[max_reward_idx]
                    improvements_tried.append(
                        self.improvement_strategies[max_reward_idx].__name__
                    )
                else:
                    # If no improvement found, force at least one addition for long responses
                    if len(best_response) >= len(original_response) and len(improvements_tried) == 0:
                        best_response = best_response + "\n\nAdditional insights and considerations: ..."
                    break
            else:
                break
        
        # Record improvement history
        self.training_history.append({
            "original_response": original_response,
            "final_response": best_response,
            "reward_improvement": best_reward - self.reward_model.predict_reward(original_response),
            "improvements_applied": improvements_tried
        })
        
        return best_response, best_reward

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