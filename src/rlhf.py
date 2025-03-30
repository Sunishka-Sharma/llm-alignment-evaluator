import numpy as np
from typing import List, Dict, Tuple
import logging

class RewardModel:
    """Simple reward model for alignment learning."""
    
    def __init__(self):
        self.feedback_data = []
        self.weights = None
    
    def collect_feedback(self, response: str, human_rating: float, 
                        explanation: str = "") -> None:
        """Collect human feedback on model responses."""
        self.feedback_data.append({
            "response": response,
            "rating": human_rating,
            "explanation": explanation
        })
        logging.info(f"Collected feedback: Rating {human_rating}")
    
    def extract_features(self, response: str) -> np.ndarray:
        """Extract basic features from response."""
        features = [
            len(response),  # Length
            response.count("!"),  # Excitement
            any(word in response.lower() for word in ["cannot", "won't", "not"]),  # Refusal
            response.count("."),  # Complexity
        ]
        return np.array(features)
    
    def train(self) -> None:
        """Train a simple linear reward model."""
        if len(self.feedback_data) < 5:
            logging.warning("Not enough feedback data to train")
            return
            
        X = np.array([self.extract_features(d["response"]) 
                     for d in self.feedback_data])
        y = np.array([d["rating"] for d in self.feedback_data])
        
        # Simple linear regression
        self.weights = np.linalg.pinv(X.T @ X) @ X.T @ y
        logging.info("Trained reward model on {len(self.feedback_data)} samples")
    
    def predict_reward(self, response: str) -> float:
        """Predict reward for a response."""
        if self.weights is None:
            return 0.0
        features = self.extract_features(response)
        return float(features @ self.weights)

class RLHFTrainer:
    """Basic RLHF training loop."""
    
    def __init__(self, reward_model: RewardModel):
        self.reward_model = reward_model
        self.training_history = []
    
    def improve_response(self, original_response: str, 
                        max_attempts: int = 3) -> Tuple[str, float]:
        """
        Simple response improvement based on reward prediction.
        This is a mock implementation - in reality, you'd use proper policy optimization.
        """
        best_response = original_response
        best_reward = self.reward_model.predict_reward(original_response)
        
        # Mock improvements by adding clarifications
        improvements = [
            lambda r: r + "\n\nTo clarify, this advice prioritizes safety and ethics.",
            lambda r: r + "\n\nPlease note this is a general guideline.",
            lambda r: "I want to be helpful while ensuring safety: " + r
        ]
        
        for improve_fn in improvements[:max_attempts]:
            candidate = improve_fn(original_response)
            reward = self.reward_model.predict_reward(candidate)
            
            if reward > best_reward:
                best_response = candidate
                best_reward = reward
        
        return best_response, best_reward

def example_usage():
    # Initialize
    reward_model = RewardModel()
    trainer = RLHFTrainer(reward_model)
    
    # Collect some feedback
    responses = [
        "I cannot help with harmful activities.",
        "Here's a detailed explanation of the concept...",
        "Let me assist you with that task.",
    ]
    ratings = [3.0, 2.5, 2.0]
    
    for resp, rating in zip(responses, ratings):
        reward_model.collect_feedback(resp, rating)
    
    # Train the reward model
    reward_model.train()
    
    # Try to improve a response
    original = "I can help with that."
    improved, reward = trainer.improve_response(original)
    print(f"Original: {original}")
    print(f"Improved: {improved}")
    print(f"Predicted reward: {reward}") 