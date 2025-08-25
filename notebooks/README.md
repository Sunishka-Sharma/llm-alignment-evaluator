# Red-Teaming Notebooks for gpt-oss-20b 🎯

This directory contains specialized notebooks for red-teaming gpt-oss-20b, designed to run in Google Colab.

## 📚 Notebooks Overview

### 1. Chain of Thought Manipulation (`chain_of_thought_testing.ipynb`)
- Tests for hidden reasoning and strategic omission
- Multi-turn conversation analysis
- Automated vulnerability detection
- Visualization of reasoning patterns

### 2. Data Exfiltration (`data_exfiltration_testing.ipynb`)
- Training data leakage detection
- Pattern analysis and fingerprinting
- Memorization detection
- Advanced visualization tools

### 3. Deceptive Alignment (Coming Soon)
- Goal conflict detection
- Strategic behavior analysis
- Evaluation awareness testing

## 🚀 Running in Google Colab

1. **Open in Colab**
   - Go to [Google Colab](https://colab.research.google.com)
   - Click `File` → `Upload Notebook`
   - Select the notebook you want to run

2. **Setup Environment**
   ```python
   # The notebooks will automatically:
   %pip install -q --upgrade torch
   %pip install -q transformers triton==3.4 kernels
   ```

3. **Mount Data Directory**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. **Load Model**
   - The notebooks use HuggingFace's Transformers library
   - gpt-oss-20b is loaded automatically
   - Uses CUDA if available (Colab GPU recommended)

5. **Run Tests**
   - Execute cells in order
   - Results are saved automatically
   - Visualizations are displayed inline

## 💡 Tips for Best Results

1. **Use a GPU Runtime**
   - In Colab: `Runtime` → `Change runtime type` → Select `GPU`
   - Significantly faster inference

2. **Save Results**
   ```python
   # Results are automatically saved to:
   /content/results/[test_type]/findings.json
   ```

3. **Memory Management**
   - Clear output cells regularly
   - Restart runtime if memory issues occur
   - Use `low_cpu_mem_usage=True` in model loading

4. **Batch Processing**
   - Tests run automatically in batches
   - Progress bars show completion
   - Results are aggregated automatically

## 📊 Analyzing Results

Each notebook includes:
- Real-time analysis
- Interactive visualizations
- Severity scoring
- Automated report generation

Results are saved in Kaggle submission format:
```json
{
  "title": "Vulnerability Type in gpt-oss-20b",
  "description": "...",
  "severity": 8.5,
  "breadth": 7,
  "novelty": 9,
  "reproducibility": 10,
  "methodological_insight": 9,
  "vulnerability_details": {
    // Detailed analysis
  }
}
```

## 🔄 Updating Tests

To add new test cases:
1. Edit `data/red_teaming_prompts.json`
2. Add new attack vectors
3. Rerun notebooks

## 🚨 Important Notes

1. **API Keys**
   - No API keys needed
   - Uses local model deployment

2. **Resource Usage**
   - Model is ~40GB
   - Requires significant VRAM
   - Consider using 8-bit quantization

3. **Time Estimates**
   - Full test suite: ~2-3 hours
   - Individual notebooks: 30-45 minutes
   - Analysis generation: 5-10 minutes