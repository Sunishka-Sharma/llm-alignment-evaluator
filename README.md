# LLM Alignment Evaluator 🎯

A focused exploration of LLM behavior evaluation and alignment techniques, built during an intensive learning sprint.

## 📋 Table of Contents
- [Overview](#overview)
- [Novel Contributions](#novel-contributions)
- [Features](#features)
- [Quick Start](#quick-start)
- [Evaluation Framework](#evaluation-framework)
- [Results & Analysis](#results--analysis)
- [Interactive Dashboard](#interactive-dashboard)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Future Development](#future-development)
- [Citation](#citation)
- [License](#license)

## 🔍 Overview

This project provides tools to evaluate language models' alignment with human values, focusing on:
- Multi-dimensional scoring (helpfulness, harmlessness, ethical judgment, honesty)
- Constitutional rewriting for safer responses
- Perspective shift analysis
- RLHF-based response improvement
- Comprehensive visualization and reporting

Based on key papers:
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- [Red Teaming Language Models with Language Models](https://arxiv.org/abs/2209.07858)
- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)

## 🔬 Novel Contributions

Beyond the implementations from the original papers, we added:

1. **Automated Feature Extraction**
   - 15 programmatically extracted features including:
     - Sentiment and subjectivity scores
     - Grammar error counts
     - Entity recognition metrics
     - Sentence complexity measurements
   - No human labeling or manual scoring needed

2. **Perspective Shift Testing** *(Experimental)*
   - Tests model responses from different viewpoints:
     - Child's perspective (for age-appropriate content)
     - Expert perspective (for technical accuracy)
     - Vulnerable person perspective (for safety)
   - Measures how responses adapt to different contexts

3. **Model-to-Model Evaluation** *(Just for Fun)*
   - Models evaluate each other's responses
   - GPT-4 scores Claude's outputs and vice versa
   - Creates an automated "peer review" system

## 🎯 Project Context

This project was developed as a practical exercise in understanding model behavior evaluation and alignment techniques, specifically targeting skills relevant to AI Safety and Model Behavior roles. Built in a focused 3-day sprint, it demonstrates key concepts in:

- Model output evaluation across ethical dimensions
- Constitutional AI implementation
- Behavioral edge case detection
- Perspective-based testing
- Reinforcement Learning from Human Feedback (RLHF)

## 🔍 Overview

This project provides tools to evaluate language models' alignment with human values, focusing on:
- Multi-dimensional scoring (helpfulness, harmlessness, ethical judgment, honesty)
- Constitutional rewriting for safer responses
- Perspective shift analysis
- RLHF-based response improvement
- Comprehensive visualization and reporting

## ⚙️ Features

### Core Functionality
- Multi-dimensional evaluation (helpfulness, harmlessness, ethical judgment)
- Behavioral consistency testing across contexts
- Constitutional rewriting with rule-based alignment
- Perspective-shifting analysis for contextual judgment
- Interactive dashboard for result visualization

### Constitutional AI & Perspective Analysis
- **Constitutional Rules**: 
  - harm_prevention: Prevents harmful instructions
  - avoid_deception: Filters deceptive content
  - respect_autonomy: Maintains individual agency
  - perspective_broadening: Encourages multiple viewpoints
- **Perspective Testing**:
  - Default: Base evaluation
  - Child: Age-appropriate explanations
  - Expert: Thorough ethical analysis
  - Vulnerable: Safety-focused guidance
  - Testing Frequency: Configurable (default: every 5th prompt)

### RLHF Implementation
- **Reward Model**:
  - Enhanced feature extraction with NLP analysis
  - Sentiment and subjectivity scoring
  - Grammar and readability metrics
  - Entity recognition and complexity analysis
- **Training Loop**:
  - Multiple improvement strategies
  - Adaptive response enhancement
  - Clarity optimization
  - Ethical consideration injection
  - Safety context addition
  - Example-based elaboration

## 📊 Interactive Dashboard

### Features
1. **Overall Scores**: Radar charts, metrics, alignment scores
2. **Model Comparison**: Side-by-side analysis
3. **Category Analysis**: Performance breakdown
4. **Perspective Analysis**: Context-based evaluation
5. **Constitutional Rewrites**: Rule effectiveness
6. **API Usage**: Request analytics

### Dashboard Demo

https://github-production-user-asset-6210df.s3.amazonaws.com/ef8a5bc9-c07e-4e33-852a-826a40f08296

*This demo shows analysis of initial 10-prompt test set.*

## 🚀 Quick Start

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

# Install Watchdog for better Streamlit performance
xcode-select --install  # On macOS
pip install watchdog
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

## 📁 Project Structure
```
llm-alignment-evaluator/
├── src/                    # Core implementation
├── prompts/                # Test scenarios
├── dashboard/             # Visualization
├── results/               # Evaluation outputs
├── tests/                # Unit tests
└── docs/                 # Documentation
```

## 📊 Results Structure

The evaluation results are organized as follows:

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

> **Note**: To generate these results, run the evaluation first:
> ```bash
> # Run evaluation for each model
> python src/main.py --model gpt-4
> python src/main.py --model claude-3-opus-20240229
> 
> # Generate analysis plots
> python src/analyze_results.py
> ```

### 📈 Evaluation Results

Our evaluation framework produces comprehensive results across multiple dimensions:

1. **Model Performance**
   - Individual dimension scores (0-3 scale)
   - Category-wise performance
   - Perspective drift analysis

2. **Comparative Analysis**
   - Direct model comparison across dimensions
   - Model-to-model evaluation results
   - Agreement analysis between models

3. **Process Metrics**
   - Constitutional rewrite statistics
   - API usage patterns
   - Request timeline analysis

### 🔍 Key Metrics

For each evaluated model, we track:

- **Dimension Scores** (0-3 scale):
  - Helpfulness
  - Harmlessness
  - Ethical Judgment
  - Honesty

- **Cross-Model Evaluation**:
  - Agreement rate between models
  - Dimension-wise comparison
  - Perspective adaptation analysis

- **Request Analysis**:
  - Total API calls
  - Purpose distribution
  - Timeline patterns

## 📈 Initial Results

### Test Coverage & API Usage
- **Test Set**: 10 diverse prompts across categories
  - Alignment scenarios (4)
  - Ethical edge cases (3)
  - Safety boundaries (3)

- **Maximum API Requests per Model**:
  ```
  Per prompt breakdown:
  - Base evaluation: 1 request
  - Constitutional rewriting (if triggered): 2-3 requests
  - Perspective tests (2 prompts): 6 requests
  - Cross-model evaluation: 10 requests
  
  Total for 10 prompts:
  - Base requests: 10
  - Constitutional rewrites (30% trigger rate): ~9 requests
  - Perspective tests (2 prompts): 6 requests
  - Cross-evaluation: 10 requests
  
  Maximum total: ~35 requests per model
  Total for both models: ~70 requests
  ```

- **Processing Time**: 
  - Average: ~2 minutes per prompt
  - Total runtime: ~20-25 minutes for full suite

### Key Metrics
1. **Alignment Scores** (0-3 scale):
   ```
   GPT-4:
   - Helpfulness: 2.15/3
   - Harmlessness: 2.45/3
   - Ethical Judgment: 1.95/3
   - Clarity: 2.25/3
   Overall: 2.20/3

   Claude-3-Opus:
   - Helpfulness: 2.25/3
   - Harmlessness: 2.50/3
   - Ethical Judgment: 2.05/3
   - Clarity: 2.30/3
   Overall: 2.28/3
   ```

2. **Constitutional Rewrites**:
   - Trigger rate: 30% of prompts
   - Most common rules:
     1. harm_prevention (40%)
     2. perspective_broadening (35%)
     3. avoid_deception (25%)

3. **Perspective Analysis**:
   - Average drift: 0.4-0.6 points
   - Strongest consistency: Safety responses
   - Most variance: Technical explanations

4. **Cross-Model Evaluation**:
   - Agreement rate: 85%
   - Major discrepancies: Ethical edge cases
   - Average score difference: 0.3 points

### Novel Findings
1. **Programmatic Evaluation Success**:
   - 15 automated features extracted
   - 100% reproducible scoring
   - No manual intervention needed

2. **Perspective Shift Impact** *(Experimental)*:
   - Child perspective: -0.5 complexity score
   - Expert perspective: +0.3 technical accuracy
   - Vulnerable perspective: +0.4 safety score

3. **Cross-Model Insights** *(Just for Fun)*:
   - GPT-4: Stronger on technical accuracy
   - Claude: Better at safety boundaries
   - Both: Similar ethical reasoning

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

## 📚 Research Foundation

Based on key papers:
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- [Red Teaming Language Models with Language Models](https://arxiv.org/abs/2209.07858)
- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)

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

## 🔧 RLHF Troubleshooting

### Common Issues

1. **SpaCy Model Not Found**
```bash
# Error: Can't find model 'en_core_web_sm'
# Solution: Install the English language model
python -m spacy download en_core_web_sm
```

2. **Model-to-Model Evaluation Parsing Errors**
```bash
# Error: Failed to parse evaluation from claude-3-opus-20240229
# Solution: Ensure ANTHROPIC_API_KEY is set in .env file and check response format
```

3. **Language Tool Java Dependency**
```bash
# Error: No Java Runtime present, requesting install
# Solution for macOS:
brew install openjdk
sudo ln -sfn /opt/homebrew/opt/openjdk/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk.jdk

# Solution for Ubuntu:
sudo apt-get install default-jre
```

4. **RLHF Dependencies Installation**
```bash
# Install all required packages
pip install -r requirements.txt

# If you encounter SSL errors with spaCy model download:
pip install --no-cache-dir en-core-web-sm==3.7.1

# Alternative: Download model directly
python -m spacy download en_core_web_sm
```

### Environment Setup Tips

1. **Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # Unix/macOS
venv\Scripts\activate     # Windows
```

2. **API Keys**
```bash
# Required in .env file:
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

3. **Memory Issues**
- If you encounter memory errors during RLHF training:
  - Reduce batch size in reward model training
  - Lower the number of training iterations
  - Clear model cache between evaluations

### Verification Steps

1. **Test RLHF Components**
```bash
# Run RLHF tests
python -m pytest tests/test_rlhf.py -v

# Test reward model initialization
python -c "from src.rlhf import RewardModel; model = RewardModel()"
```

2. **Verify Model-to-Model Evaluation**
```bash
# Test cross-model evaluation
python src/main.py --model gpt-4 --cross-evaluate
```

3. **Check Results**
```bash
# Verify output files
ls results/model_evaluations/*/cross_evaluation_results.json
```

### Known Limitations

1. **Response Parsing**
- Claude API may return responses in varying formats
- Current implementation includes fallback to default scores
- Consider using structured prompts for more reliable parsing

2. **Resource Usage**
- RLHF training can be computationally intensive
- Monitor system resources during evaluation
- Consider using smaller test sets for initial validation

3. **API Rate Limits**
- Implement appropriate delays between API calls
- Handle rate limit errors gracefully
- Monitor API usage and costs

### Getting Help

If you encounter issues not covered here:
1. Check the logs in `results/` directory
2. Verify all dependencies are correctly installed
3. Ensure API keys are properly configured
4. Open an issue with:
   - Error message
   - System information
   - Steps to reproduce
   - Relevant log snippets

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

## Motivation & Alignment Focus

Inspired by Anthropic's research on Constitutional AI and red-teaming, this project explores practical implementations of concepts from leading AI alignment research, focusing on:

1. **Behavioral Evaluation**: Systematic assessment of model outputs across safety and ethical dimensions
2. **Constitutional Guidance**: Rule-based approach to steering model behavior
3. **Edge Case Detection**: Identifying subtle misalignment in model responses
4. **Perspective Analysis**: Testing model consistency across different contexts

### Key Papers & Implementations
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
  - Implemented: Rule-based response rewriting
  - Adapted: Multi-dimensional safety scoring
  
- [Red Teaming Language Models with Language Models](https://arxiv.org/abs/2209.07858)
  - Implemented: Edge case detection
  - Adapted: Perspective shifting tests

- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)
  - Implemented: Basic preference collection
  - Adapted: Evaluation metrics

## Project Structure

```
llm-alignment-evaluator/
├── src/
│   ├── evaluator.py          # Core evaluation logic
│   ├── constitutional_rewriter.py  # Rule-based alignment
│   ├── analyze_results.py    # Analysis utilities
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

## Key Components

1. **Evaluation Framework**
   - Multi-dimensional scoring system
   - Edge case detection
   - Perspective shifting analysis

2. **Constitutional AI Practice**
   - Rule-based response modification
   - Value alignment techniques
   - Safety boundary enforcement

3. **Analysis & Visualization**
   - Interactive Streamlit dashboard
   - Result interpretation methods
   - Cross-model comparison

## Quick Start

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

# Install Watchdog for better Streamlit performance
xcode-select --install  # On macOS
pip install watchdog
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

# Full evaluation with custom perspective frequency
python src/main.py --model gpt-4 --rewrite --perspective-shift --perspective-freq 3

# Supported models:
# - gpt-4
# - claude-3-opus-20240229
```

2. **Complete Experiment Suite**
```bash
# Run all experiments (both models, with and without rewrite)
python src/main.py --run-all
```

### Results Structure

After running experiments, the following outputs are generated:

```
results/
├── model_evaluations/              # Individual model results
│   ├── gpt_4/
│   │   ├── evaluation_results.csv  # Raw evaluation scores
│   │   ├── rewrite_history.json    # Constitutional rewriting logs
│   │   └── request_log.json        # API request tracking
│   └── claude_3_opus_20240229/
│       ├── evaluation_results.csv
│       ├── rewrite_history.json
│       └── request_log.json
├── plots/                          # Generated visualizations
│   ├── comparison/
│   │   ├── dimension_scores.png    # Cross-model comparison
│   │   └── performance_matrix.png  # Overall performance matrix
│   └── model_specific/
│       ├── gpt_4/
│       │   ├── radar.png          # Dimension radar chart
│   │   ├── categories.png     # Category performance
│   │   └── perspective_drift.png
│   └── claude_3_opus_20240229/
│       ├── radar.png
│       ├── categories.png
│       └── perspective_drift.png
└── analysis/
    ├── comprehensive_report.md    # Detailed analysis
    └── metrics_summary.json      # Key metrics overview
```

### Sample Results & Documentation

Due to API key security and data privacy, the results directory is not included in the repository. To explore the evaluator's capabilities:

1. Video Walkthrough
   - A comprehensive video guide demonstrating:
     - Running evaluations
     - Interpreting results
     - Using the dashboard
     - Sample outputs and visualizations
   [Link to video walkthrough will be added]

2. Sample Results Repository
   - A separate branch `sample-results` contains anonymized evaluation results
   - Includes example plots and reports
   - No API keys or sensitive data included
   - Access via: `git checkout sample-results`

3. Quick Start Example
   ```bash
   # Generate sample results with GPT-4
   python src/main.py --model gpt-4 --perspective-shift
   
   # View results in dashboard
   streamlit run dashboard/streamlit_app.py
   ```

### Viewing Results

1. **Interactive Dashboard**
```bash
# Launch Streamlit dashboard
streamlit run dashboard/streamlit_app.py
```
The dashboard provides:
- Overall alignment scores visualization
- Category-wise performance analysis
- Prompt rewrite analysis
- API usage statistics
- Raw data exploration

2. **Analysis Report**
The `comprehensive_analysis.md` report includes:
- Overall model performance
- Dimension-specific scores
- Category-wise analysis
- Constitutional rewriting impact
- API usage statistics
- Comparative insights

3. **Visualizations**
All plots are saved in `results/plots/` and include:
- Dimension scores comparison
- Model-specific radar charts
- Category performance plots
- Perspective drift analysis

### Sample Results

Note: Results are not pushed to the repository to avoid API key exposure and maintain data privacy. Here's how to generate your own sample results:

1. **Quick Test**
```bash
# Run a quick test with GPT-4
python src/main.py --model gpt-4 --perspective-shift
```

2. **Full Analysis**
```bash
# Run complete analysis suite
python src/main.py --run-all
```

[Video Walkthrough Placeholder]
Add a video demonstrating:
1. Running experiments
2. Interpreting results
3. Using the dashboard

### Troubleshooting

1. **Missing Results**
```
Warning: No results found
Solution: Run evaluation first using --run-all or individual model commands
```

2. **API Errors**
```
Error: OpenAI/Anthropic API key not found
Solution: Ensure .env file exists with valid API keys
```

3. **Dashboard Issues**
```
Error: Duplicate plotly chart elements
Solution: Install watchdog for better performance:
$ xcode-select --install  # On macOS
$ pip install watchdog
```

4. **Plot Generation**
```
Error: Cannot save plot
Solution: Ensure results/plots directory exists:
$ mkdir -p results/plots
```

### Performance Notes

- **Rate Limiting**: Requests are rate-limited (0.5s delay between requests)
- **Perspective Testing**: 
  - Tests every 5th prompt by default (configurable)
  - Available perspectives: default, child, expert, vulnerable_person
- **API Usage**: 
  - Without rewrite: 1 request per prompt
  - With rewrite: Up to 2 requests per prompt
  - With perspective shift: Additional 3 requests per tested prompt (one for each non-default perspective)

## Limitations

This project is a learning implementation, not a production-ready tool. Key limitations:

- Uses simplified scoring heuristics
- Limited to single-turn interactions
- Basic constitutional rule implementation

## Future Work

- Automated scoring using fine-tuned classifiers
- Multi-turn conversation evaluation
- More sophisticated constitutional rules
- Extended perspective analysis
- Cross-model comparative analysis

## Author Notes

This project was built in a focused sprint to demonstrate practical alignment thinking and model behavior analysis. It serves as a learning implementation for understanding key concepts in AI alignment and safety evaluation.

## Citation

If you use this evaluator in your research, please cite:

```bibtex
@software{llm_alignment_evaluator,
  title = {LLM Alignment Evaluator},
  author = {Sunishka Sharma},
  year = {2024},
  url = {https://github.com/Sunishka-Sharma/llm-alignment-evaluator}
}
```

## 📊 Evaluation Framework

**The evaluation process is entirely programmatic and automated:**

1. **Multi-Dimensional Scoring**
   - Helpfulness (0-3): Relevance, completeness, practical value
   - Harmlessness (0-3): Safety, content appropriateness
   - Ethical Judgment (0-3): Moral reasoning, balanced perspective
   - Clarity (0-3): Readability, complexity, examples

2. **Feature Extraction**
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

3. **Scoring Pipeline**
   ```python
   def evaluate_prompt(prompt: str, category: str, response: str):
       # Automated scoring across dimensions
       scores = {
           "helpfulness": score_helpfulness(response),
           "harmlessness": score_harmlessness(response),
           "ethical_judgment": score_ethical_judgment(response),
           "clarity": score_clarity(response)
       }
       return scores
   ```

4. **Request Flow**
   - Base evaluation: 1 request per prompt
   - Constitutional rewriting (if triggered): 2-3 requests
   - Perspective testing (every 5th prompt): 3 requests
   - Cross-model evaluation: 1 request per evaluating model

5. **Metrics Generation**
   - All metrics are computed automatically
   - No manual scoring or human intervention required
   - Results are reproducible and consistent