# 🔍 LLM Alignment Evaluator

A unified framework for evaluating LLM behavior on subtle moral, safety, and consistency dimensions, with optional constitutional patching.

## 🎯 Core Features

- 🔁 **Multi-Dimensional Evaluation**: Tests model responses across helpfulness, harmlessness, ethical judgment, and honesty
- ⚖️ **Perspective Shifting**: Analyzes how responses change based on context and user vulnerability
- 📜 **Constitutional Rewriting**: Light implementation of value-aligned prompt modification
- 📊 **Visual Analysis**: Streamlit dashboard for exploring evaluation results
- 🌈 **Edge Case Detection**: Identifies inconsistencies in model behavior

## 🔬 Motivation & Novel Contributions

Built in a focused sprint to demonstrate practical alignment thinking and model behavior analysis. Key insights:

1. **Safety-Consistency Divergence**: Models often exhibit misalignment between refusing harmful actions and justifying subtle unethical behavior
2. **Context-Dependent Ethics**: Responses can vary significantly based on how questions are framed
3. **Lightweight Constitutional Patching**: Simple rules can help align model outputs at inference time

### 📚 Key Papers Referenced
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- [Red Teaming Language Models with Language Models](https://arxiv.org/abs/2209.07858)
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

## 🛠️ Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-alignment-evaluator.git
cd llm-alignment-evaluator

# Run setup script
./setup.sh

# Add your API keys to .env file
cp .env.example .env
# Edit .env with your keys

# Run evaluation
python src/main.py --prompts prompts/eval_prompts.csv --model gpt-3.5-turbo

# Run with constitutional rewriting
python src/main.py --prompts prompts/eval_prompts.csv --model gpt-3.5-turbo --rewrite

# Launch dashboard
cd dashboard && streamlit run streamlit_app.py
```

## 📊 Example Results

The evaluator produces several types of insights:

1. **Dimension Scores** (0-3 scale):
   - Helpfulness: How useful and clear is the response?
   - Harmlessness: Does it avoid promoting harmful actions?
   - Ethical Judgment: Quality of moral reasoning
   - Honesty: Transparency about uncertainty

2. **Perspective Analysis**:
   - How do responses change when framed for different audiences?
   - Measures drift in ethical judgment across contexts

3. **Constitutional Improvements**:
   - Before/after comparisons of rewritten prompts
   - Success rate of harm reduction

## 🏗️ Project Structure

```
llm-alignment-evaluator/
├── prompts/                  # Test prompts by category
├── src/                      # Core evaluation logic
│   ├── evaluator.py         # Response scoring
│   ├── constitutional_rewriter.py  # Value alignment
│   └── analyze_results.py   # Metrics & visualization
├── dashboard/               # Streamlit interface
├── results/                 # Output data & reports
├── tests/                   # Unit tests
└── README.md
```

## 🔄 Evaluation Process

1. **Prompt Loading**: Categorized test cases from CSV
2. **Model Querying**: Supports OpenAI and Anthropic APIs
3. **Multi-Dimensional Scoring**: Rule-based evaluation
4. **Perspective Testing**: Context-shifted prompts
5. **Constitutional Rewriting**: Value-aligned modifications
6. **Analysis & Visualization**: Trends and patterns

## 🎯 Future Work

- Automated scoring using fine-tuned classifiers
- Expanded rule set for constitutional rewriting
- Multi-turn conversation evaluation
- Cross-model comparative analysis
- Automated red-teaming

## 📝 Citation

If you use this evaluator in your research, please cite:

```bibtex
@software{llm_alignment_evaluator,
  title = {LLM Alignment Evaluator},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/llm-alignment-evaluator}
}
```

## 🤝 Contributing

Contributions welcome! Please check out our [contribution guidelines](CONTRIBUTING.md).

## 📄 License

MIT License - see [LICENSE](LICENSE) for details. 