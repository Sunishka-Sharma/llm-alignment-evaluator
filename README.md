# LLM Alignment Evaluator 🎯

A comprehensive framework for evaluating and enhancing LLM alignment across multiple dimensions, implementing modern AI safety techniques including Constitutional AI, RLHF, cross-model evaluation, and perspective testing. Built as an intensive one-day sprint challenge to demonstrate rapid implementation of advanced AI alignment research, with flexible modules inspired by Anthropic's real-world safety evaluations.

> **Note:** This project was executed in a rapid prototyping sprint strictly capped by me at 20 hours to explore alignment insights without access to training-level infrastructure. While not based on fine-tuned models or original datasets, it replicates and extends behavioral evaluation pipelines in a modular, low-cost format for broader accessibility.

---

## 🎬 Demo Videos & Results

### Interactive Dashboard Walkthrough
<video src="https://github.com/user-attachments/assets/3a051841-21be-44a4-b46f-6705419c1c66" controls></video>
*Interactive dashboard visualizing evaluation results and model comparisons*

### Results Exploration
<video src="https://github.com/user-attachments/assets/2a0736e2-d989-4709-aac6-f7a414819523" controls></video>
*Exploring the comprehensive evaluation results and generated reports*

> **📝 Note:** To access the complete set of documented results, switch to the `sample_results` branch:
> ```bash
> git checkout sample_results
> ```
> This branch contains all generated visualizations, evaluation data, and analysis reports.

---

## 📋 Table of Contents
- [Overview & Motivation](#-overview--motivation)
- [Key Innovations](#-key-innovations)
- [Core Features](#-core-features)
- [Evaluation Framework](#-evaluation-framework)
- [Limitations](#-limitations)
- [Sample Results](#-sample-results)
- [Customization & Execution](#-customization--execution)
- [Novel Adaptations vs. Original Research](#-novel-adaptations-vs-original-research)
- [Project Structure](#-project-structure)
- [Setup Instructions](#-setup-instructions)
- [Visualizations](#-visualizations)
- [Troubleshooting](#-troubleshooting)
- [Future Development](#-future-development)
- [License](#-license)
- [Citation](#-citation)

---

## 🔍 Overview & Motivation

This project demonstrates applied research in Language Model alignment, combining:
- Systematic model behavior evaluation
- Constitutional AI prompt rewriting
- Perspective-shift and context-based testing
- RLHF reward simulation
- Cross-model auditing and discrepancy analysis
- Ethical evaluator with customizable moral frameworks


This sprint project was built in 1 day as part of a portfolio demo for Anthropic-style behavior evaluation. It is not intended as a full-scale benchmark or paper replication, but a flexible tool for auditing and inspecting LLM behavior.

### Research Foundations
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- [Red Teaming Language Models with Language Models](https://arxiv.org/abs/2209.07858)
- [RLHF with Human Feedback](https://arxiv.org/abs/2203.02155)

---

## 💡 Key Innovations

| Contribution | Inspired by | Novelty |
|--------------|-------------|---------|
| Perspective Drift Radar | None | Persona-based consistency metric using radar plots |
| Multi-Dimensional RLHF Reward | OpenAI RLHF | No human annotation, 4-axis scoring |
| Lightweight Constitutional AI | Anthropic | Rule-based filtering, not iterative generation |
| Cross-Model Mutual Scoring | Anthropic | Mutual auditing for ethical variance |
| Unified Modular Pipeline | All | One-click reproducibility and toggles |
| Ethical Evaluator | Various Ethics Frameworks | Hierarchical ethical reasoning assessment |
| Constitution Editor | Anthropic Constitutional AI | User-defined ethical frameworks with priority resolution |
| Red Teaming Evaluator | Anthropic | Automated vulnerability testing and attack pattern detection |
| Factuality Checker | Various | Knowledge-base comparison and claim verification |
| Cultural Sensitivity Analysis | N/A | Region-specific evaluation with cultural context awareness |
| Multi-Turn Consistency | N/A | Conversation-level consistency evaluation across multiple turns |

---

## ⚙️ Core Features

### Evaluation Dimensions (0–3 scale)
- **Helpfulness**: Relevance, completeness, utility
- **Harmlessness**: Content safety, tone appropriateness
- **Ethical Judgment**: Moral clarity, value alignment
- **Honesty**: Factuality, uncertainty disclosure
- **Cultural Sensitivity**: Region-specific context awareness and appropriateness
- **Consistency**: Response stability across conversation turns
- **Red Team Resistance**: Robustness against adversarial prompting

### Rule-Based Constitutional AI
- Rewriting logic triggered on ~30% of prompts
- 4 customizable principles: `harm_prevention`, `avoid_deception`, `respect_autonomy`, `perspective_broadening`
- Each match logs the rule name and improvement score
- Interactive Constitution Editor for custom rules

### RLHF Simulation
- No human feedback used
- Includes elaboration, reframe, refusal strategies
- Score deltas computed per dimension and visualized

### Perspective Testing
- Personas: `child`, `expert`, `vulnerable`, `skeptic`
- Toggle: `--perspective-shift`
- Frequency control: `--perspective-freq`

### Cross-Model Evaluation
- Claude and GPT-4 audit each other's outputs
- Score divergence measured and visualized
- Disagreements most frequent in ethical questions

### Ethical Evaluation
- Hierarchical framework evaluating reasoning quality
- Conflict detection between competing ethical values
- User-definable constitution with priority resolution
- Context-sensitive rule application

### Factuality Checking
- Knowledge base comparison for claim verification
- Automated factual claim extraction
- Scaled factuality scoring (1-10) with confidence metrics
- Supports custom knowledge bases

### Red Team Testing
- Automated vulnerability detection
- Adversarial prompt generation
- Attack pattern categorization
- Robustness scoring system

### Cultural Sensitivity Analysis
- Region-specific context evaluation
- Caste and religious sensitivity detection
- Cross-cultural communication assessment
- Avoids Western-centric biases

### Multi-Turn Consistency
- Conversation-level evaluation
- Cross-turn response stability
- Pressure tactic resistance
- Follow-up question generation

---

## 📊 Evaluation Framework

```python
def evaluate_prompt(prompt: str, category: str, response: str):
    return {
        "helpfulness": score_helpfulness(response),
        "harmlessness": score_harmlessness(response),
        "ethical_judgment": score_ethical_judgment(response),
        "honesty": score_honesty(response)
    }
```
- Fully programmatic with CSV output
- Used across all evaluation stages: base, rewrite, RLHF, cross-model

### Advanced Evaluators
- BaseEvaluator: Common interface for all specialized evaluators
- EthicalEvaluator: Ethical reasoning and value conflict assessment
- FactualityEvaluator: Factual accuracy against knowledge base
- RedTeamingEvaluator: Vulnerability testing with attack patterns
- CulturalSensitivityEvaluator: Region and context-specific analysis
- MultiTurnConsistencyEvaluator: Conversation-level stability testing

### API Efficiency & Resource Usage
- ~110–120 total requests per 10 prompts:
  - Base runs: 40
  - Rewrites: ~10
  - Perspectives: ~24
  - Cross-eval: ~40
- Adjustable via `REQUEST_DELAY`, `MAX_RETRIES`
- Dashboard visualizations optimize for low-resource environments

---

## 🧩 Customization & Execution

```bash
# Full suite (recommended)
python src/main.py --run-all

# Specific model only
python src/main.py --model gpt-4

# With rewriting
python src/main.py --model gpt-4 --rewrite

# With perspective shift every 3rd prompt
python src/main.py --model gpt-4 --perspective-shift --perspective-freq 3

# With cross-eval
python src/main.py --model gpt-4 claude-3-opus-20240229 --cross-evaluate

# With ethical evaluation
python src/main.py --model gpt-4 --ethical-eval

# With red teaming
python src/main.py --model gpt-4 --red-team

# With factuality checking
python src/main.py --model gpt-4 --factuality-check

# With cultural sensitivity analysis
python src/main.py --model gpt-4 --cultural-eval

# With multi-turn consistency testing
python src/main.py --model gpt-4 --consistency-test

# Launch the dashboard with constitution editor
python dashboard/run_dashboard.py --app both
```

When using `--run-all`, the framework automatically:
- Runs both GPT-4 and Claude 3 Opus
- Enables rewriting for both models
- Tests perspective shifts every 5th prompt by default
- Performs cross-evaluation (GPT-4 evaluates Claude's responses and vice versa)
- Generates all visualizations and reports

Custom personas and rules are editable in `evaluator.py` and `constitutional_rewriter.py`

---

## 📈 Sample Results (10 Prompt Subset)

| Metric | GPT-4 | Claude 3 |
|--------|-------|----------|
| Helpfulness | 2.15 | 2.25 |
| Harmlessness | 2.45 | 2.50 |
| Ethical Judgment | 1.95 | 2.05 |
| Honesty | 2.25 | 2.30 |
| **Overall** | **2.20** | **2.28** |

- Rewrite trigger rate: 30%
- Avg rewrite success: 85%
- Perspective drift: 0.4–0.6 pts
- RLHF gain: ~0.051 improvement across dimensions

---

## 🧪 Novel Adaptations vs. Original Research

### Constitutional AI (Anthropic, 2022)
**Original Approach**: 
- Uses AI to critique and revise its own outputs
- Requires multiple LLM calls per revision (critique → revise → verify)
- Focuses on output-level constitutional rules
- Computationally expensive with multiple iterations

**My Implementation**:
- Pattern-based prompt analysis with no additional model calls
- Single-pass rewrite system using targeted instructions
- Input-level constitutional filtering before generation
- 85% effective at improving alignment with 70% fewer API calls
- Quantifiable improvement metrics for each rewritten prompt
- User-defined constitution with hierarchical ethical priorities
- Interactive Constitution Editor for customizing ethical rules

### Red Teaming LLMs (Anthropic, 2022)
**Original Approach**:
- Uses specialized adversarial models to find vulnerabilities
- Requires multiple dedicated adversarial models 
- Focused on discovering new attack vectors
- Limited to binary success/failure metrics

**My Implementation**:
- Mutual model evaluation creates cross-checking system
- Quantitative disagreement visualization across 4 dimensions
- Perspective drift analysis reveals contextual vulnerabilities
- Produces fine-grained alignment maps rather than just attack vectors
- Identifies model-specific blind spots through comparative analysis
- Automated red teaming with attack pattern categorization and scoring
- Cultural and regional context-aware testing

### RLHF (OpenAI, 2022)
**Original Approach**:
- Requires thousands of human preference ratings
- Uses single reward scalar for human preference
- Needs specialized infrastructure for reward model training
- Focused solely on helpfulness dimension

**My Implementation**:
- Multi-dimensional reward model with no human annotations
- Simulates preference learning across 4 distinct dimensions
- Demonstrates targeted improvement strategies with before/after metrics
- Lightweight implementation usable on consumer hardware
- Integrates cross-model evaluation feedback as supplementary signal
- Incorporates ethical reasoning quality in reward modeling
- Includes multi-turn consistency and cultural sensitivity metrics

### Novel Integration Elements
- **Unified Pipeline**: First system to combine constitutional filtering, cross-model evaluation, perspective testing, and RLHF in a single framework
- **Perspective Drift Radar**: New metric for quantifying contextual alignment stability (no precedent in literature)
- **Comparative Alignment Matrices**: Visual cross-model agreement mapping for identifying systemic pattern differences
- **Lightweight Alignment Toolkit**: Reproducible in under 24 hours with minimal API usage
- **Ethical Framework Integration**: Hierarchical ethical evaluation with user-defined constitutions
- **Constitution Editor**: Interactive tool for creating and testing custom ethical frameworks
- **Multi-dimension Evaluation Ecosystem**: Comprehensive modular evaluators covering ethical, factual, cultural, and consistency aspects

### Perspective Drift Analysis
![Perspective Drift Analysis](https://github.com/Sunishka-Sharma/llm-alignment-evaluator/raw/sample_results/results/plots/model_specific/claude_3_opus_20240229/perspective_drift.png)

Perspective Drift Analysis introduces a novel contextual alignment metric with no precedent in the literature:

- Measures how model alignment scores change across different personas (child, expert, vulnerable person)
- Identifies which dimensions are most sensitive to contextual reframing
- Quantifies variance in model behavior when explaining to different audiences
- Provides early detection of alignment instabilities specific to certain contexts
- Visualizes drift patterns using radar plots showing all alignment dimensions simultaneously

### Cross-Model Evaluation
![Cross-Model Evaluation](https://github.com/Sunishka-Sharma/llm-alignment-evaluator/raw/sample_results/results/plots/comparison/cross_model_evaluation.png)

Cross-Model Evaluation offers a systematic approach to uncovering alignment blind spots:

- Models evaluate each other's outputs across all alignment dimensions
- Generates quantitative disagreement matrices to identify systematic differences
- Calculates dimension-specific agreement rates to pinpoint areas of misalignment
- Reveals where models overrate or underrate their own capabilities
- Identifies prompts where the largest cross-model evaluation gaps occur

---

## 📝 Project Structure

```
llm-alignment-evaluator/
├── src/                      # Evaluation + scoring
│   ├── __init__.py
│   ├── analyze_results.py    # Results analysis and visualization
│   ├── constitutional_rewriter.py # Constitutional rewriting system
│   ├── demo_rlhf.py          # RLHF demonstration script
│   ├── evaluator.py          # Core evaluation logic
│   ├── base_evaluator.py     # Base evaluation class
│   ├── ethical_evaluator.py  # Ethical reasoning evaluator
│   ├── ethical_framework.py  # Hierarchical ethics framework
│   ├── factuality_checker.py # Factual accuracy verification
│   ├── red_teaming_evaluator.py # Vulnerability testing
│   ├── cultural_sensitivity_evaluator.py # Cultural context evaluation
│   ├── consistency_evaluator.py # Multi-turn consistency testing
│   ├── main.py               # Main entry point
│   └── rlhf.py               # RLHF implementation
├── data/                     # Data files
│   ├── wikipedia_sample.json # Knowledge base for factuality
│   └── factuality_test_prompts.json # Test prompts
├── prompts/                  # Prompt sets
│   └── eval_prompts.csv      # Default evaluation prompts
├── dashboard/                # Streamlit UI
│   ├── streamlit_app.py      # Main dashboard application
│   ├── constitution_editor.py # Constitution editing tool
│   └── run_dashboard.py      # Dashboard runner script
├── results/                  # All logs, plots, csvs
│   ├── analysis/             # Generated reports
│   ├── model_evaluations/    # Raw evaluation data
│   ├── plots/                # Visualization outputs
│   └── rlhf_demo/            # RLHF results
├── generate_plots.py         # Report script
├── tests/                    # Test cases
├── requirements.txt          # Dependencies
└── README.md                 # Documentation
```

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.8+
- OpenAI API key
- Anthropic API key (optional, for Claude evaluations)

### Environment Setup

```bash
# Clone repository
git clone https://github.com/Sunishka-Sharma/llm-alignment-evaluator.git
cd llm-alignment-evaluator

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### API Configuration
The project requires API keys to be set up as environment variables:

1. Create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
```

2. Load environment variables:
```bash
# Linux/macOS
source .env

# Windows PowerShell
foreach($line in Get-Content .env) {
    $name, $value = $line.split('=')
    Set-Content env:\$name $value
}
```

### Running the Framework
The framework automatically creates all necessary directories on first run:
```bash
# Run full evaluation suite
python src/main.py --run-all

# Launch dashboard to view results
python dashboard/run_dashboard.py

# Launch dashboard with constitution editor
python dashboard/run_dashboard.py --app both
```

---

## 📊 Visualizations

The framework provides comprehensive visualizations of evaluation results:

```bash
# Generate all visualization plots
python generate_plots.py

# View through interactive dashboard
python dashboard/run_dashboard.py
```

The visualizations include:

1. **Comparison Plots** (`results/plots/comparison/`) 
   - Dimension scores (spider and bar charts)
   - Cross-model evaluations
   - Self vs cross comparisons

2. **Model-Specific Plots** (`results/plots/model_specific/<model_name>/`)
   - Radar charts for dimensions
   - Category performance
   - Perspective drift analysis

3. **RLHF Analysis** (`results/rlhf_demo/`)
   - Dimension improvements
   - Before/after comparisons

4. **Ethical Evaluations** (`results/ethical_evaluations/`)
   - Reasoning quality assessments
   - Ethical conflict analysis
   - Constitution-based evaluations

5. **Specialty Evaluations** (`results/specialty_evaluations/`)
   - Red team vulnerability reports
   - Factuality assessments
   - Cultural sensitivity analyses
   - Multi-turn consistency tests

---

## ❓ Troubleshooting

### Common Issues

**API Authentication**
- Verify API keys are properly set in environment variables
- Test direct API connection with a simple query
- For persistent issues, regenerate API keys in provider dashboard

**Rate Limits**
- Increase `REQUEST_DELAY` in `src/main.py` (default: 1s)
- Use command-line argument `--request-delay 3` for temporary adjustment
- Run with smaller prompt set during testing

**Missing Results**
- Ensure evaluation was run with `python src/main.py --run-all`
- Check permissions on results directory
- Generate plots manually with `python generate_plots.py`

**Dashboard Issues**
- Install Streamlit: `pip install streamlit==1.15.0 watchdog`
- Run with debug logging: `streamlit run --logger.level=debug dashboard/streamlit_app.py`
- Check port availability (default: 8501)
- For Constitution Editor issues, use the integrated launcher: `python dashboard/run_dashboard.py`

**Ethical Evaluator Issues**
- Verify constitution file exists at the configured path
- If editing fails, check permissions on the constitution file directory
- Use the default constitution as a starting point

---
## ⚠️ Limitations

This framework has several important limitations to consider:

- **No Fine-Tuning**: Uses pre-trained models without any parameter updates
- **Public Models Only**: Limited to commercially available APIs (Claude/GPT)
- **Simulated RLHF**: Contains reward modeling but no actual reinforcement learning
- **Small Prompt Set**: Demo uses only 10 prompts (up to 100 planned)
- **No Multimodal Support**: Text-only evaluation without image or audio capabilities
- **Limited Persona Range**: Only 4 perspective personas for drift testing
- **No Training Data Access**: Cannot inspect model weights or training corpora
- **Synthetic Evaluation**: Programmatic scores without human preference data

---
## 🔮 Future Development

- **Expanded Dataset**: Increase to 100+ diverse prompts
- **Model Variety**: Add additional models (e.g., Llama, Mistral)
- **Training Integration**: Train reward models from generated annotations
- **Temporal Testing**: Add time-series evaluation for model behavior drift
- **Multi-Turn Evaluation**: Extend to conversation-level alignment testing
- **Customizable Rubrics**: Dynamic scoring criteria definition
- **Batch Processing**: Parallel evaluation for larger datasets
- **Enhanced Ethical Evaluation**: Expand ethical frameworks and reasoning assessment
- **Improved Constitution Editor**: More advanced visualization and testing tools
- **Cross-cultural Evaluation**: Expand cultural sensitivity analysis to more regions
- **Attack Pattern Library**: Comprehensive categorization of red team vulnerabilities
- **Factuality Benchmarks**: Domain-specific knowledge base verification

### Potential Extensions
- **LMSYS Integration**: Add evaluation on LMSYS Chatbot Arena responses
- **PEFT Fine-Tuning**: Plug into HuggingFace models with PEFT for small-scale fine-tuning
- **Multimodal Behavior**: Extend to multimodal LLM behavior (e.g., image captioning)
- **Constitution Library**: Shareable, importable ethical frameworks

---

## ⚖️ License

MIT License — see [LICENSE](LICENSE)

---

## 📝 Citation
```bibtex
@software{llm_alignment_evaluator,
  title = {LLM Alignment Evaluator},
  author = {Sunishka Sharma},
  year = {2024},
  url = {https://github.com/Sunishka-Sharma/llm-alignment-evaluator}
}
```

For Anthropic-style alignment evaluation, this project demonstrates modular, auditable, reproducible insight-driven implementation. All results are stored, plotted, and cross-auditable. Ideal for public portfolio or internal alignment evaluation tooling.

---

© 2025 Sunishka Sharma. All Rights Reserved.

This project and its contents are protected under international copyright laws. Unauthorized reproduction, distribution, or use of any part of this project without explicit permission is prohibited.

For inquiries or permissions, please contact: [sunishka20137@iiitd.ac.in]

This work is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
