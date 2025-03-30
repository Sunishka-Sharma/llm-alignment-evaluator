# LLM Alignment Evaluator

A focused exploration of LLM behavior evaluation and alignment techniques, built during an intensive learning sprint.

## Project Context

This project was developed as a practical exercise in understanding model behavior evaluation and alignment techniques, specifically targeting skills relevant to AI Safety and Model Behavior roles. Built in a focused 3-day sprint, it demonstrates key concepts in:

- Model output evaluation across ethical dimensions
- Constitutional AI implementation
- Behavioral edge case detection
- Perspective-based testing

## Overview

This project provides tools to evaluate language models' alignment with human values, focusing on:
- Multi-dimensional scoring (helpfulness, harmlessness, ethical judgment, honesty)
- Constitutional rewriting for safer responses
- Perspective shift analysis
- Comprehensive visualization and reporting

## Features

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

## Interactive Dashboard

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
```

2. **Complete Experiment Suite**
```bash
# Run all experiments
python src/main.py --run-all

# View results
streamlit run dashboard/streamlit_app.py
```

## Project Structure
```
llm-alignment-evaluator/
├── src/                    # Core implementation
├── prompts/                # Test scenarios
├── dashboard/             # Visualization
├── results/               # Evaluation outputs
├── tests/                # Unit tests
└── docs/                 # Documentation
```

## Initial Results

### Test Coverage
- 10 prompts tested across GPT-4 and Claude-3-Opus
- Basic alignment (4), ethical edge cases (3), safety boundaries (3)
- Average completion: ~2 minutes per prompt
- API usage: ~20-25 requests per model

### Key Findings
- Average alignment scores:
  - GPT-4: 2.03/3
  - Claude-3-Opus: 2.11/3
- Constitutional rewrites: ~30% of prompts
- Perspective drift: 0.4-0.6 points

## Future Development

1. **Large-Scale Testing**
   - 100+ diverse prompts
   - Additional models
   - Extended analysis

2. **Enhanced Features**
   - Automated scoring
   - Multi-turn evaluation
   - Advanced constitutional rules
   - Batch processing

## Research Foundation

Based on key papers:
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- [Red Teaming Language Models with Language Models](https://arxiv.org/abs/2209.07858)
- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)

## Troubleshooting

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

## Citation

```bibtex
@software{llm_alignment_evaluator,
  title = {LLM Alignment Evaluator},
  author = {Sunishka Sharma},
  year = {2024},
  url = {https://github.com/Sunishka-Sharma/llm-alignment-evaluator}
}
```

## License

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
│       │   ├── categories.png     # Category performance
│       │   └── perspective_drift.png
│       └── claude_3_opus_20240229/
│           ├── radar.png
│           ├── categories.png
│           └── perspective_drift.png
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