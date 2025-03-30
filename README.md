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
  4. **Cross-Model Evaluation Integration**: Feedback from multiple models incorporated into RLHF training

## 📋 Table of Contents
- [Overview & Motivation](#-overview--motivation)
- [Core Features](#%EF%B8%8F-core-features)
- [Technical Implementation](#-technical-implementation)
- [Evaluation Framework](#-evaluation-framework)
- [Sample Results](#-sample-results)
- [Project Structure](#-project-structure)
- [Quick Start Guide](#-quick-start-guide)
- [Execution Flow](#-execution-flow)
- [Future Development](#-future-development)
- [Troubleshooting](#-troubleshooting)
- [Visualizations](#-visualizations)

## 🔍 Overview & Motivation

This project demonstrates my expertise in Language Model behavior evaluation and alignment techniques, showcasing practical implementations of key concepts from leading AI safety research:

1. **Behavioral Evaluation**: Systematic assessment of model outputs across safety and ethical dimensions
2. **Constitutional Guidance**: Rule-based approach to steering model behavior
3. **Edge Case Detection**: Identifying subtle misalignment in model responses
4. **Perspective Analysis**: Testing model consistency across different contexts
5. **Cross-Model Evaluation**: Leveraging multiple models' perspectives for more robust evaluation

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

- **Rewriting Behavior**:
  - Conditionally triggered: Only ~30% of prompts typically match rule criteria
  - Rule-based analysis: Each prompt is checked against all constitutional rules
  - Multi-step process: Prompts can undergo multiple rewrite iterations if needed
  - API usage: Each rewrite attempt requires an additional API call
  - Tracking: All rewrites are logged in `rewrite_history.json` for analysis

### Perspective Analysis
- Tests model responses across different viewpoints:
  - Default: Base evaluation
  - Child: Age-appropriate explanations
  - Expert: Thorough ethical analysis  
  - Vulnerable: Safety-focused guidance
- **Perspective Drift Metrics**:
  - Measures how model's alignment shifts across different contexts
  - Calculates drift for each dimension (helpfulness, harmlessness, etc.)
  - Higher drift values indicate inconsistent judgment depending on context
- **Customizable Testing**:
  - Enable/disable with `--perspective-shift` flag
  - Adjust testing frequency with `--perspective-freq` parameter
  - Results visualized as radar plots showing variance across perspectives

### RLHF Implementation
- **Reward Model**:
  - Enhanced feature extraction with NLP analysis
  - Multiple improvement strategies
  - Adaptive response enhancement
  - Example-based elaboration
  - Multi-dimensional scoring (0-3) across alignment dimensions
  - Cross-model evaluation integration for more robust reward signals

### Cross-Model Evaluation
- Models evaluate each other's responses for more objective assessment
- Agreement analysis across different dimensions
- Identification of major discrepancies by category
- Integration with RLHF for more robust improvement
- Visualization of self vs external evaluation comparisons

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
   - Constitutional rewriting (if triggered): 1-2 additional requests per prompt that triggers rules
   - Perspective testing (every 5th prompt): 3 additional requests (one for each non-default perspective)
   - Cross-model evaluation: 1 request per prompt per evaluating model
   
   When using `--run-all`:
   - For each model (gpt-4 and claude-3-opus-20240229), runs are performed both with and without rewriting
   - Perspective shift testing is enabled by default for all runs
   - All prompts are tested, with perspective shifts occurring every 5th prompt
   - Cross-model evaluation is performed after each run
   
   **API Request Breakdown for `--run-all` with 10 prompts:**
   - Base response requests: 40 (10 prompts × 2 models × 2 runs)
   - Perspective shift requests: ~24 (2 prompts × 3 perspectives × 2 models × 2 runs)
   - Rewrite requests: ~6-12 (assuming ~30% of prompts trigger rewrites)
   - Cross-evaluation requests: ~40 (10 prompts × 1 evaluating model × 2 models × 2 runs)
   - **Total expected requests: ~110-120**

   > **Note**: API usage is tracked in `request_log.json` files and can be viewed in the dashboard under the "API Usage" tab.

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
│   ├── gpt_4/
│   │   ├── evaluation_results.csv
│   │   ├── rewrite_history.json
│   │   ├── request_log.json
│   │   └── cross_evaluation_results.json
│   └── claude_3_opus_20240229/
│       └── ...
├── plots/                      # Generated visualizations
│   ├── comparison/            # Cross-model analysis
│   │   ├── dimension_scores_spider.png    # Spider chart comparison
│   │   ├── dimension_scores_bar.png       # Bar chart comparison
│   │   ├── category_scores.png            # Category performance comparison
│   │   ├── rewrite_effectiveness.png      # Constitutional rewriting effectiveness
│   │   ├── cross_model_evaluation.png     # Model-to-model evaluation
│   │   └── self_vs_cross_evaluation.png   # Self vs cross-evaluation comparison
│   └── model_specific/        # Individual model analysis
│       ├── gpt_4/
│       │   ├── radar.png                  # Individual model dimensions
│       │   ├── dimension_scores_bar.png   # Dimension scores bar chart
│       │   ├── categories.png             # Category performance
│       │   └── perspective_drift.png      # Perspective analysis
│       └── claude_3_opus_20240229/
│           └── ...
├── analysis/                   # Generated reports
│   ├── comprehensive_report.md # Overall analysis report
│   ├── cross_model_report.md   # Detailed cross-model evaluation analysis
│   └── metrics_summary.json    # Summary metrics in JSON format
└── rlhf_demo/                  # RLHF training demo results
    ├── dimension_improvements.png     # Improvement by dimension visualization
    └── training_history.json          # RLHF training history and examples
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

1. **Complete Experiment Suite (Recommended Flow)**
```bash
# Run all model evaluations, generate reports, plots and RLHF demo in one command
python src/main.py --run-all

# View the interactive dashboard with all results
streamlit run dashboard/streamlit_app.py
```

2. **Individual Model Evaluation (For testing specific features)**
```bash
# Basic evaluation
python src/main.py --model gpt-4

# With constitutional rewriting
python src/main.py --model gpt-4 --rewrite

# With perspective shift testing
python src/main.py --model gpt-4 --perspective-shift

# Customize perspective shift frequency (default is every 5 prompts)
python src/main.py --model gpt-4 --perspective-shift --perspective-freq 3

# Generate visualizations separately after running individual evaluations
python generate_plots.py
```

## 🔄 Execution Flow

The framework follows a systematic process for evaluating and analyzing model alignment:

1. **Input Processing**: 
   - Load prompts from CSV/JSON files
   - Configure model(s) and evaluation settings

2. **Model Evaluation**:
   - **Constitutional Rewriting** (if enabled): Filter prompts through constitutional rules
   - **Response Generation**: Get model response for each prompt
   - **Multi-dimensional Scoring**: Evaluate each response across alignment dimensions
   - **Perspective Shift Testing** (if enabled): Test how model responses change when prompts are reframed from different perspectives (child, expert, vulnerable person)
   - **Cross-Model Evaluation**: Have other models evaluate each model's responses

3. **Analysis & Visualization**:
   - Generate individual model reports
   - Compare models across dimensions
   - Analyze perspective drift
   - Evaluate constitutional rewrite impact
   - Produce cross-model evaluation reports

4. **Results Presentation**:
   - Export data to CSV/JSON
   - Generate visualization plots
   - Create comprehensive reports
   - Present interactive dashboard

5. **RLHF Integration** (optional):
   - Train reward models on evaluation data
   - Test response improvements
   - Analyze improvement strategies
   - Compare self vs. cross-model evaluation

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

### Common Issues and Solutions

1. **Missing Results**
```
Error: No results found
Solution: Run evaluation using the command: 
  python src/main.py --run-all
```

2. **API Rate Limits**
```
Error: OpenAI/Anthropic API rate limit exceeded
Solution: 
  - The code has built-in rate limiting, but you may need to adjust REQUEST_DELAY in src/main.py
  - For larger evaluations, implement exponential backoff or run in smaller batches
```

3. **Missing Rewrite Data**
```
Error: No rewrite data available / Rewrite not functioning
Solution:
  - Ensure the --rewrite flag is set when running individual models
  - When using --run-all, the system automatically runs both with and without rewrite
  - Check request_log.json files to confirm rewrite requests were made
  - Verify that prompts actually triggered the rewrite rules (only ~30% of prompts typically trigger rewrites)
```

4. **Visualization Issues**
```
Error: Missing or incomplete visualizations
Solution:
  - Run the generate_plots.py script directly: python generate_plots.py
  - Ensure you have evaluation results in results/model_evaluations/ directory
  - Check for any error messages during plot generation
  - Verify that matplotlib and seaborn are properly installed
```

5. **Dashboard Issues**
```
Error: Duplicate plotly chart elements
Solution: 
  - Install watchdog for better performance: pip install watchdog
  - Clear browser cache or use incognito mode
  - If specific charts fail to load, check that all analysis files exist
```

6. **Missing Parameters Error**
```
Error: Function missing required positional arguments
Solution: 
  - If you modified the code, ensure function calls match parameter signatures
  - When adding custom functions, check parameter consistency
```

7. **RLHF Dependencies**
```bash
# Install required packages
pip install torch>=2.0.0 transformers>=4.30.0 nltk>=3.8.1 textblob>=0.17.1

# Install SpaCy model
python -m spacy download en_core_web_sm

# If encountering CUDA errors with PyTorch
pip install torch==2.0.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

8. **Memory Issues with Large Datasets**
```
Error: Memory error during analysis
Solution:
  - Process data in smaller batches by modifying prompts file
  - Reduce plots_dpi parameter in visualization functions
  - For local testing, use fewer prompts by editing prompts/eval_prompts.csv
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

## Visualizations

You can generate comparison plots and model-specific plots using the provided script:

```bash
python generate_plots.py
```

This script will:
1. Create all comparison plots in `results/plots/comparison/` including:
   - Dimension scores (bar and spider charts)
   - Category comparison
   - Rewrite effectiveness
   - Cross-model evaluation
   - Self vs Cross evaluation comparison
   
2. Generate model-specific plots in `results/plots/model_specific/<model_name>/` including:
   - Radar plots showing dimension scores
   - Bar charts for dimension scores
   - Category score comparisons
   - Perspective analysis (if perspective shift testing was enabled)
   
3. Run the RLHF demo to generate RLHF-related visualizations

The script uses existing evaluation results and does not require modifying the main.py file. It will automatically load all available model evaluations and generate the appropriate visualizations.

### Key Visualizations Explained

#### Rewrite Effectiveness
The rewrite effectiveness visualization provides insights into how well the constitutional rewriting system improves prompts:
- Shows the percentage of prompts that were successfully improved
- Displays the count of prompts that were improved vs. not improved for each model
- Helps analyze the effectiveness of different rewriting rules

#### Self vs Cross Evaluation
This visualization compares how models evaluate themselves versus how they are evaluated by other models:
- Side-by-side comparison of self and cross-evaluation scores across dimensions
- Helps identify potential biases in self-evaluation
- Reveals discrepancies in how models judge the same responses

### Benefits of Using `generate_plots.py`

- **No Code Modification**: Generates all plots without modifying the main codebase
- **Comprehensive Visualization**: Creates both comparison and model-specific plots in a single run
- **Custom Visualizations**: Adds specialized plots not available in the default pipeline:
  - Self vs Cross-evaluation comparison
  - Enhanced dimension score bar charts with value labels
  - Perspective drift analysis using radar charts
- **Improved Performance**: Optimized to generate all plots in a single run with proper memory management
- **Automatic Report Generation**: Creates cross-model evaluation reports in markdown format
- **RLHF Integration**: Automatically runs the RLHF demo and generates related visualizations

You can view all generated plots through the dashboard:
```bash
streamlit run dashboard/streamlit_app.py
```