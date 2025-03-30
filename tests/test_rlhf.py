import pytest
import numpy as np
from src.rlhf import RewardModel, RLHFTrainer

def test_reward_model_initialization():
    """Test reward model initialization."""
    model = RewardModel()
    assert len(model.feature_names) == 15  # Check all features are defined
    assert model.feedback_data == []
    assert hasattr(model, 'model')
    assert hasattr(model, 'scaler')

def test_feature_extraction():
    """Test feature extraction functionality."""
    model = RewardModel()
    
    # Test empty response
    features = model.extract_features("")
    assert len(features) == len(model.feature_names)
    assert all(f == 0 for f in features)
    
    # Test complex response
    response = "This is a detailed response that considers ethical implications. However, we must also consider safety. For example, let's analyze multiple perspectives."
    features = model.extract_features(response)
    
    assert len(features) == len(model.feature_names)
    assert features[0] > 0  # Length
    assert features[2] > 1  # Sentence count
    assert features[6] > 0  # Ethical terms
    assert features[7] > 0  # Perspective terms

def test_clarity_score():
    """Test clarity score calculation."""
    model = RewardModel()
    
    # Test empty response
    assert model._calculate_clarity_score("") == 0.0
    
    # Test simple response
    simple = "This is a simple sentence."
    simple_score = model._calculate_clarity_score(simple)
    assert 0 <= simple_score <= 1
    
    # Test complex response
    complex = "This is a very complex, long-winded sentence with multiple clauses and sophisticated vocabulary that might be harder to understand."
    complex_score = model._calculate_clarity_score(complex)
    assert complex_score < simple_score  # Complex should be less clear

def test_feedback_collection():
    """Test feedback collection with metadata."""
    model = RewardModel()
    
    # Test basic feedback
    model.collect_feedback("Test response", 2.5)
    assert len(model.feedback_data) == 1
    assert model.feedback_data[0]["rating"] == 2.5
    assert model.feedback_data[0]["metadata"] == {}
    
    # Test feedback with metadata
    metadata = {"source": "test", "category": "safety"}
    model.collect_feedback("Another response", 3.0, "Good response", metadata)
    assert len(model.feedback_data) == 2
    assert model.feedback_data[1]["metadata"] == metadata
    assert model.feedback_data[1]["explanation"] == "Good response"

def test_model_training():
    """Test reward model training."""
    model = RewardModel()
    
    # Test training with insufficient data
    model.train()
    assert not hasattr(model.model, 'feature_importances_')
    
    # Add sufficient training data
    responses = [
        "Response one with ethical considerations.",
        "Response two focusing on safety measures.",
        "Response three with multiple perspectives.",
        "Response four with clear examples.",
        "Response five with balanced viewpoints."
    ]
    ratings = [2.0, 2.5, 3.0, 2.8, 2.7]
    
    for resp, rating in zip(responses, ratings):
        model.collect_feedback(resp, rating)
    
    # Train model
    model.train()
    assert hasattr(model.model, 'feature_importances_')
    assert len(model.model.feature_importances_) == len(model.feature_names)

def test_reward_prediction():
    """Test reward prediction functionality."""
    model = RewardModel()
    
    # Test prediction without training
    assert model.predict_reward("test") == 0.0
    
    # Train model with more distinctive examples
    responses = [
        "bad",  # Very low quality
        "This is okay.",  # Basic response
        "This is a good response that explains the concept.",  # Better response
        "This is an excellent response that thoroughly explains the concept while considering multiple perspectives, ethical implications, and providing clear examples for better understanding.",  # Best response
    ]
    ratings = [1.0, 1.5, 2.0, 3.0]  # More spread out ratings
    
    for resp, rating in zip(responses, ratings):
        model.collect_feedback(resp, rating)
    
    model.train()
    
    # Test predictions with very distinct quality differences
    low_quality = "bad response"
    high_quality = "This is a comprehensive response that carefully considers multiple perspectives, addresses ethical implications, provides clear examples, and maintains excellent clarity while thoroughly explaining the concept."
    
    low_reward = model.predict_reward(low_quality)
    high_reward = model.predict_reward(high_quality)
    
    assert high_reward > low_reward

def test_rlhf_trainer():
    """Test RLHF trainer functionality."""
    model = RewardModel()
    trainer = RLHFTrainer(model)
    
    # Test improvement strategies
    response = "Basic response."
    
    ethical = trainer._add_ethical_consideration(response)
    assert "ethical perspective" in ethical.lower()
    
    safety = trainer._add_safety_context(response)
    assert "safety" in safety.lower()
    
    perspective = trainer._add_perspective_analysis(response)
    assert "perspectives" in perspective.lower()
    
    clarity = trainer._improve_clarity(response)
    assert "clearly" in clarity.lower()
    
    examples = trainer._add_examples(response)
    assert "example" in examples.lower()

def test_response_improvement():
    """Test response improvement process."""
    model = RewardModel()
    trainer = RLHFTrainer(model)
    
    # Add training data
    responses = [
        "I cannot help with that.",
        "Here's a helpful explanation with ethical considerations...",
        "Let me assist you while considering safety.",
        "This is a balanced perspective.",
        "Here are some examples to illustrate."
    ]
    ratings = [1.0, 3.0, 2.5, 2.8, 2.7]
    
    for resp, rating in zip(responses, ratings):
        model.collect_feedback(resp, rating)
    
    model.train()
    
    # Test improvement
    original = "I can help with that."
    improved, reward = trainer.improve_response(original)
    
    assert len(improved) > len(original)
    assert reward > model.predict_reward(original)
    assert len(trainer.training_history) == 1
    assert trainer.training_history[0]["original_response"] == original
    assert trainer.training_history[0]["reward_improvement"] > 0

def test_edge_cases():
    """Test edge cases and error handling."""
    model = RewardModel()
    trainer = RLHFTrainer(model)
    
    # Test empty response
    improved, reward = trainer.improve_response("")
    assert len(improved) > 0
    assert reward >= 0
    
    # Test very long response
    long_response = "test " * 1000
    improved, reward = trainer.improve_response(long_response)
    assert len(improved) > len(long_response)
    assert reward >= 0
    
    # Test response with special characters
    special_response = "Test?!@#$%^&*()"
    improved, reward = trainer.improve_response(special_response)
    assert len(improved) > len(special_response)
    assert reward >= 0

if __name__ == "__main__":
    pytest.main([__file__]) 