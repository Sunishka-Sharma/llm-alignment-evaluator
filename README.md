# LLM Alignment Evaluator 🎯

A comprehensive framework for evaluating and enhancing LLM alignment across multiple dimensions, implementing modern AI safety techniques including Constitutional AI, RLHF, cross-model evaluation, and perspective testing. Built as an intensive one-day sprint challenge to demonstrate rapid implementation of advanced AI alignment research, with flexible modules inspired by Anthropic's real-world safety evaluations.

---

## 🎬 Demo Videos & Results

### Interactive Dashboard Walkthrough
[![Dashboard Demo](https://github.com/user-attachments/assets/3a051841-21be-44a4-b46f-6705419c1c66)](https://youtu.be/your_dashboard_demo_video_id)
*Click image to view full dashboard walkthrough*

### Results Exploration
[![Results Walkthrough](https://github.com/user-attachments/assets/2a0736e2-d989-4709-aac6-f7a414819523)](https://youtu.be/your_results_video_id)
*Click image to view results directory exploration*

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
- [Sample Results](#-sample-results)
- [Customization & Execution](#-customization--execution)
- [Novel Adaptations vs. Original Research](#-novel-adaptations-vs-original-research)
- [Project Structure](#-project-structure)
- [Setup Instructions](#-setup-instructions)
- [Troubleshooting](#-troubleshooting)
- [Visualizations](#-visualizations)
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

Built in under 24 hours to simulate real-world alignment sprint conditions, it integrates modular experimentation pipelines for targeted LLM behavior analysis.

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

---

## ⚙️ Core Features

### Evaluation Dimensions (0–3 scale)
- **Helpfulness**: Relevance, completeness, utility
- **Harmlessness**: Content safety, tone appropriateness
- **Ethical Judgment**: Moral clarity, value alignment
- **Honesty**: Factuality, uncertainty disclosure

### Rule-Based Constitutional AI
- Rewriting logic triggered on ~30% of prompts
- 4 customizable principles: `harm_prevention`, `avoid_deception`, `respect_autonomy`, `perspective_broadening`
- Each match logs the rule name and improvement score

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

### API Efficiency
- ~110–120 total requests per 10 prompts:
  - Base runs: 40
  - Rewrites: ~10
  - Perspectives: ~24
  - Cross-eval: ~40
- Adjustable via `REQUEST_DELAY`, `MAX_RETRIES`

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

### Constitutional AI
- Paper: iterative rewrite by LLM
- Yours: lightweight pattern-matched rewrite

### Red Teaming
- Paper: adversarial prompt discovery
- Yours: mutual scoring and disagreement visualization

### RLHF
- Paper: trained reward model from preferences
- Yours: simulated scoring across 4 axes

---

## 📁 Project Structure

```
llm-alignment-evaluator/
├── src/                      # Evaluation + scoring
│   ├── __init__.py
│   ├── analyze_results.py    # Results analysis and visualization
│   ├── constitutional_rewriter.py # Constitutional rewriting system
│   ├── demo_rlhf.py          # RLHF demonstration script
│   ├── evaluator.py          # Core evaluation logic
│   ├── main.py               # Main entry point
│   └── rlhf.py               # RLHF implementation
├── prompts/                  # Prompt sets
│   └── eval_prompts.csv      # Default evaluation prompts
├── dashboard/                # Streamlit UI
│   └── streamlit_app.py      # Dashboard application
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

### Installation

```bash
# Clone repository
git clone https://github.com/Sunishka-Sharma/llm-alignment-evaluator.git
cd llm-alignment-evaluator

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys (create .env file)
echo "OPENAI_API_KEY=sk-your-key-here" > .env
echo "ANTHROPIC_API_KEY=sk-ant-your-key-here" >> .env

# Install optional NLP dependencies (for RLHF demo)
python -m spacy download en_core_web_sm

# Create necessary directories
mkdir -p results/model_evaluations
mkdir -p results/analysis
mkdir -p results/plots/comparison
mkdir -p results/plots/model_specific
```

---

## ❓ Troubleshooting

### API Connection Issues

**Error**: Authentication or connection issues with OpenAI/Anthropic APIs

**Solutions**:
1. Verify API keys in `.env` file (no spaces or quotes)
2. Make sure API keys are exported to environment:
   ```bash
   export OPENAI_API_KEY=sk-...
   export ANTHROPIC_API_KEY=sk-ant-...
   ```
3. Test API connection directly:
   ```python
   import openai
   import os
   client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
   response = client.chat.completions.create(
       model="gpt-3.5-turbo", 
       messages=[{"role": "user", "content": "Hello"}]
   )
   print(response.choices[0].message.content)
   ```

### Rate Limit Errors

**Error**: "Rate limit exceeded" errors from API

**Solutions**:
1. Increase delay between requests in `src/main.py`:
   ```python
   # Find and modify
   REQUEST_DELAY = 3  # Increase from default
   ```
2. Run smaller batches:
   ```bash
   python src/main.py --model gpt-4 --prompt-limit 5
   ```
3. Add exponential backoff:
   ```python
   # Modify get_model_response in main.py
   import time
   import random
   
   def get_model_response(prompt, model_name, retry=0, max_retries=5):
       try:
           # Original code...
       except openai.RateLimitError:
           if retry < max_retries:
               wait_time = 2 ** retry + random.uniform(0, 1)
               print(f"Rate limit hit, waiting {wait_time}s")
               time.sleep(wait_time)
               return get_model_response(prompt, model_name, retry + 1)
           raise
   ```

### Missing Results Issues

**Error**: "No results found" when running dashboard or visualizations

**Solutions**:
1. Run the full evaluation:
   ```bash
   python src/main.py --run-all
   ```
2. Check directory structure exists:
   ```bash
   # Create required directories
   mkdir -p results/model_evaluations/gpt_4
   mkdir -p results/plots/comparison
   mkdir -p results/plots/model_specific/gpt_4
   ```
3. Check evaluation output exists:
   ```bash
   ls -la results/model_evaluations/gpt_4/
   # Should see evaluation_results.csv
   ```

### Visualization Problems

**Error**: Plots not generating or appearing in dashboard

**Solutions**:
1. Run the plot generation script:
   ```bash
   python generate_plots.py
   ```
2. Check matplotlib backend:
   ```python
   # At the top of generate_plots.py
   import matplotlib
   matplotlib.use('Agg')  # Add this before other matplotlib imports
   ```
3. Fix permissions on output directories:
   ```bash
   chmod -R 755 results/
   ```

### RLHF Errors

**Error**: RLHF demo fails with NLP or ML errors

**Solutions**:
1. Install additional dependencies:
   ```bash
   pip install torch==2.0.0 transformers==4.30.0 nltk==3.8.1
   python -m spacy download en_core_web_sm
   python -m nltk.downloader punkt
   ```
2. For CUDA errors, use CPU-only version:
   ```bash
   pip install torch==2.0.0+cpu --index-url https://download.pytorch.org/whl/cpu
   ```

### Dashboard Issues

**Error**: Streamlit dashboard not starting or showing data

**Solutions**:
1. Install streamlit correctly:
   ```bash
   pip install streamlit==1.15.0 watchdog==2.1.6
   ```
2. Run with debug options:
   ```bash
   streamlit run --logger.level=debug dashboard/streamlit_app.py
   ```
3. Check browser compatibility (try Chrome or Firefox)

### JSON Parse Errors

**Error**: JSON parsing errors in cross-evaluation

**Solutions**:
1. Lower temperature setting in `main.py`:
   ```python
   # Find model response generation and modify
   temperature=0.2  # Lower for more consistent outputs
   ```
2. Add robust JSON parsing:
   ```python
   # In main.py
   import re
   import json
   
   def extract_json(text):
       """Extract JSON from text with better error handling."""
       try:
           # Find JSON pattern
           match = re.search(r'\{[\s\S]*\}', text)
           if match:
               json_str = match.group(0)
               return json.loads(json_str)
       except:
           pass
       return {}
   ```

### Cross-Platform Issues

**Error**: Path issues on Windows vs. Linux/Mac

**Solutions**:
1. Use `os.path.join` for all paths:
   ```python
   import os
   results_dir = os.path.join('results', 'model_evaluations')
   ```
2. Specify file encoding:
   ```python
   with open(file_path, 'r', encoding='utf-8') as f:
       data = json.load(f)
   ```

### Memory Issues

**Error**: MemoryError or slow performance with large datasets

**Solutions**:
1. Process in chunks:
   ```python
   # Read CSV in chunks
   for chunk in pd.read_csv(file_path, chunksize=100):
       # Process each chunk
   ```
2. Reduce plot resolution:
   ```python
   plt.figure(figsize=(10, 6), dpi=100)  # Lower DPI
   ```

---

## 📊 Visualizations

Run:
```bash
# Generate all visualizations
python generate_plots.py

# Launch interactive dashboard
streamlit run dashboard/streamlit_app.py
```

This creates:

1. **Comparison Plots** (`results/plots/comparison/`)
   - Dimension scores (spider and bar charts)
   - Category comparisons
   - Cross-model evaluations
   - Self vs cross evaluation comparisons

2. **Model-Specific Plots** (`results/plots/model_specific/<model_name>/`)
   - Radar charts for dimension scores
   - Category breakdowns
   - Perspective drift analysis
   - Dimension score distributions

3. **RLHF Analysis** (`results/rlhf_demo/`)
   - Dimension improvements visualization
   - Strategy effectiveness comparisons
   - Example improvements showcase

The dashboard provides an interactive way to explore all these results, with filtering options, tooltips, and detailed explanations.

---

## 🔮 Future Development

- **Expanded Dataset**: Increase to 100+ diverse prompts
- **Model Variety**: Add additional models (e.g., Llama, Mistral)
- **Training Integration**: Train reward models from generated annotations
- **Temporal Testing**: Add time-series evaluation for model behavior drift
- **Multi-Turn Evaluation**: Extend to conversation-level alignment testing
- **Customizable Rubrics**: Dynamic scoring criteria definition
- **Batch Processing**: Parallel evaluation for larger datasets

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

