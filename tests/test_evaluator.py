import pytest
from src.evaluator import AlignmentEvaluator

def test_score_helpfulness():
    evaluator = AlignmentEvaluator("test-model")
    
    # Test short, unhelpful response
    assert evaluator._score_helpfulness("I can't help") < 1.0
    
    # Test longer, more helpful response
    long_response = "Here's a detailed explanation that provides clear steps and context..."
    assert evaluator._score_helpfulness(long_response) > 2.0

def test_score_harmlessness():
    evaluator = AlignmentEvaluator("test-model")
    
    # Test harmful response
    harmful = "Here's how to hack into the system and steal data"
    assert evaluator._score_harmlessness("", harmful) < 1.0
    
    # Test safe response
    safe = "I cannot assist with harmful activities. Instead, here's how to stay safe online..."
    assert evaluator._score_harmlessness("", safe) > 2.5

def test_perspective_shift():
    evaluator = AlignmentEvaluator("test-model")
    
    def mock_model_fn(prompt):
        return "This is a test response"
    
    result = evaluator.perspective_shift_test("Test prompt", mock_model_fn)
    
    assert "perspective_results" in result
    assert "perspective_drift" in result
    assert len(result["perspective_results"]) == 4  # default + 3 perspectives 