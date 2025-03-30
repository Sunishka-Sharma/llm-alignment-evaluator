# LLM Alignment Evaluator 🎯

A comprehensive framework for evaluating and enhancing LLM behavioral alignment, developed to showcase skills in AI Safety and Model Behavior assessment.

## 🌟 Demo Highlights

### Interactive Dashboard
![Dashboard Demo](https://github-production-user-asset-6210df.s3.amazonaws.com/ef8a5bc9-c07e-4e33-852a-826a40f08296)

### Key Results 
- **Alignment Scores** (0-3 scale):
  ```
  GPT-4:
  - Helpfulness: 2.15/3
  - Harmlessness: 2.45/3
  - Ethical Judgment: 1.95/3
  - Honesty: 2.25/3
  Overall: 2.20/3

  Claude-3-Opus:
  - Helpfulness: 2.25/3
  - Harmlessness: 2.50/3
  - Ethical Judgment: 2.05/3
  - Honesty: 2.30/3
  Overall: 2.28/3
  ```

- **Novel Measurement Approaches**:
  1. **Automated Feature Extraction**: 15 programmatically extracted features for reproducible scoring
  2. **Perspective Shift Testing**: Model responses evaluated from different viewpoints
  3. **Model-to-Model Evaluation**: Creating an automated "peer review" system

## 📋 Table of Contents
- [Overview & Motivation](#-overview--motivation)
- [Core Features](#%EF%B8%8F-core-features)
- [Technical Implementation](#-technical-implementation)
- [Evaluation Framework](#-evaluation-framework)
- [Sample Results](#-sample-results)
- [Project Structure](#-project-structure)
- [Quick Start Guide](#-quick-start-guide)
- [Future Development](#-future-development)
- [Troubleshooting](#-troubleshooting)

## 🔍 Overview & Motivation

This project demonstrates my expertise in Language Model behavior evaluation and alignment techniques, showcasing practical implementations of key concepts from leading AI safety research:

1. **Behavioral Evaluation**: Systematic assessment of model outputs across safety and ethical dimensions
2. **Constitutional Guidance**: Rule-based approach to steering model behavior
3. **Edge Case Detection**: Identifying subtle misalignment in model responses
4. **Perspective Analysis**: Testing model consistency across different contexts

### Research Foundation
Based on key papers in AI alignment:
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- [Red Teaming Language Models with Language Models](https://arxiv.org/abs/2209.07858)
- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)

## ⚙️ Core Features

### Multi-Dimensional Evaluation
- Helpfulness (0-3): Relevance, completeness, practical value
- Harmlessness (0-3): Safety, content appropriateness  
- Ethical Judgment (0-3): Moral reasoning, balanced perspective
- Honesty (0-3): Factual accuracy, uncertainty disclosure

### Constitutional AI Implementation
- **Rules Framework**: 
  - harm_prevention: Prevents harmful instructions
  - avoid_deception: Filters deceptive content
  - respect_autonomy: Maintains individual agency
  - perspective_broadening: Encourages multiple viewpoints

### Perspective Analysis
- Tests model responses across different viewpoints:
  - Default: Base evaluation
  - Child: Age-appropriate explanations
  - Expert: Thorough ethical analysis  
  - Vulnerable: Safety-focused guidance

### RLHF Implementation
- **Reward Model**:
  - Enhanced feature extraction with NLP analysis
  - Multiple improvement strategies
  - Adaptive response enhancement
  - Example-based elaboration

## 🔧 Technical Implementation

### Automated Evaluation Pipeline
```python
def evaluate_prompt(prompt: str, category: str, response: str):
    # Automated scoring across dimensions
    scores = {
        "helpfulness": score_helpfulness(response),
        "harmlessness": score_harmlessness(response),
        "ethical_judgment": score_ethical_judgment(response),
        "honesty": score_honesty(response)
    }
    return scores
```

### Feature Extraction
```python
def extract_features(response: str) -> np.ndarray:
    # Extracts 15 linguistic and semantic features
    features = [
        length, avg_word_length, sentence_count,
        refusal_score, safety_score, complexity_score,
        ethical_terms, perspective_terms, clarity_score,
        sentiment_score, subjectivity_score, grammar_error_count,
        entity_count, unique_entity_types, avg_sentence_complexity
    ]
    return np.array(features)
```

## 📊 Evaluation Framework

The evaluation process is entirely programmatic and automated:

1. **Multi-Dimensional Scoring**
   - Helpfulness (0-3): Relevance, completeness, practical value
   - Harmlessness (0-3): Safety, content appropriateness
   - Ethical Judgment (0-3): Moral reasoning, balanced perspective
   - Honesty (0-3): Factual accuracy, uncertainty acknowledgment

2. **Request Flow**
   - Base evaluation: 1 request per prompt
   - Constitutional rewriting (if triggered): 2-3 requests
   - Perspective testing (every 5th prompt): 3 requests
   - Cross-model evaluation: 1 request per evaluating model

## 📈 Sample Results

### Test Coverage
- **Test Set**: 10 diverse prompts across categories
  - Alignment scenarios (4)
  - Ethical edge cases (3)
  - Safety boundaries (3)

### Constitutional Rewrites
- Trigger rate: 30% of prompts
- Most common rules:
  1. harm_prevention (40%)
  2. perspective_broadening (35%)
  3. avoid_deception (25%)

### Perspective Analysis
- Average drift: 0.4-0.6 points
- Strongest consistency: Safety responses
- Most variance: Technical explanations

### Cross-Model Evaluation
- Agreement rate: 85%
- Major discrepancies: Ethical edge cases
- Average score difference: 0.3 points

## 📁 Project Structure

```
llm-alignment-evaluator/
├── src/
│   ├── evaluator.py          # Core evaluation logic
│   ├── constitutional_rewriter.py  # Rule-based alignment
│   ├── analyze_results.py    # Analysis utilities
│   ├── rlhf.py              # RLHF implementation
│   ├── demo_rlhf.py         # RLHF demonstration
│   └── main.py              # Entry point
├── prompts/                  # Test scenarios
│   └── eval_prompts.csv     # Evaluation prompts
├── dashboard/               # Visualization
│   └── streamlit_app.py    # Interactive UI
├── results/                 # Evaluation outputs
│   ├── model_evaluations/   # Individual model results
│   ├── plots/              # Generated visualizations
│   └── analysis/           # Reports and metrics
├── tests/                  # Unit tests
└── docs/                   # Documentation
```

### Results Structure
The evaluation generates the following outputs:

```
results/
├── model_evaluations/          # Raw evaluation data
│   ├── gpt4/
│   │   ├── evaluation_results.csv
│   │   ├── rewrite_history.json
│   │   ├── request_log.json
│   │   └── cross_evaluation_results.json
│   └── claude3/
│       └── ...
└── plots/                      # Generated visualizations
    ├── comparison/            # Cross-model analysis
    │   ├── dimension_scores_comparison.png    # Overall model comparison
    │   └── cross_model_evaluation.png         # Model-to-model evaluation
    └── model_specific/        # Individual model analysis
        ├── gpt4/
        │   ├── radar.png                      # Individual model dimensions
        │   ├── categories.png                 # Category performance
        │   └── perspective_drift.png          # Perspective analysis
        └── claude3/
            └── ...
```

## 🚀 Quick Start Guide

### Prerequisites
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up API keys
cp .env.example .env
# Edit .env with your OpenAI and Anthropic API keys
```

### Running Experiments

1. **Single Model Evaluation**
```bash
# Basic evaluation
python src/main.py --model gpt-4

# With constitutional rewriting
python src/main.py --model gpt-4 --rewrite

# With perspective shift testing
python src/main.py --model gpt-4 --perspective-shift
```

2. **Complete Experiment Suite**
```bash
# Run all experiments
python src/main.py --run-all

# View results
streamlit run dashboard/streamlit_app.py
```

## 🔮 Future Development

1. **Large-Scale Testing**
   - 100+ diverse prompts
   - Additional models
   - Extended analysis

2. **Enhanced Features**
   - Automated scoring
   - Multi-turn evaluation
   - Advanced constitutional rules
   - Batch processing

## ❓ Troubleshooting

### Common Issues
1. **Missing Results**
```
Warning: No results found
Solution: Run evaluation first using --run-all
```

2. **API Errors**
```
Error: OpenAI/Anthropic API key not found
Solution: Ensure .env file exists with valid API keys
```

3. **Dashboard Issues**
```
Error: Duplicate plotly chart elements
Solution: Install watchdog for better performance
```

4. **RLHF Dependencies**
```bash
# Install SpaCy model
python -m spacy download en_core_web_sm
```

## 📝 Citation

```bibtex
@software{llm_alignment_evaluator,
  title = {LLM Alignment Evaluator},
  author = {Sunishka Sharma},
  year = {2024},
  url = {https://github.com/Sunishka-Sharma/llm-alignment-evaluator}
}
```

## ⚖️ License

MIT License - see [LICENSE](LICENSE) for details.