import pytest
import numpy as np
from src.rlhf import RewardModel, RLHFTrainer

def test_reward_model_training():
    model = RewardModel()
    
    # Test feedback collection
    model.collect_feedback("Good response", 3.0, "Very helpful")
    assert len(model.feedback_data) == 1
    assert model.feedback_data[0]["rating"] == 3.0
    
    # Test feature extraction
    features = model.extract_features("This is a test response!")
    assert len(features) == 4  # Check all features present
    assert features[1] == 1  # Check exclamation count
    
    # Test training with sufficient data
    responses = ["Response one.", "Response two!", "Cannot help."]
    ratings = [2.0, 3.0, 1.0]
    
    for resp, rating in zip(responses, ratings):
        model.collect_feedback(resp, rating)
    
    model.train()
    assert model.weights is not None
    
def test_response_improvement():
    model = RewardModel()
    trainer = RLHFTrainer(model)
    
    # Add training data
    model.collect_feedback("I cannot help with that.", 1.0)
    model.collect_feedback("Here's a helpful explanation...", 3.0)
    model.collect_feedback("Let me assist you.", 2.0)
    model.train()
    
    # Test improvement
    original = "I can help."
    improved, reward = trainer.improve_response(original)
    
    assert len(improved) > len(original)
    assert "safety" in improved.lower() or "ethics" in improved.lower()
    
def test_edge_cases():
    model = RewardModel()
    
    # Test empty response
    features = model.extract_features("")
    assert all(f == 0 for f in features)
    
    # Test prediction without training
    assert model.predict_reward("test") == 0.0
    
    # Test training with insufficient data
    model.collect_feedback("test", 1.0)
    model.train()
    assert model.weights is None 