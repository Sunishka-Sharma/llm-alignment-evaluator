import os
import argparse
import logging
from typing import List, Dict
from dotenv import load_dotenv
import openai
import anthropic
import time
from evaluator import AlignmentEvaluator
from constitutional_rewriter import ConstitutionalRewriter
from analyze_results import AlignmentAnalyzer
import json
import pandas as pd

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Rate limiting settings
REQUEST_DELAY = 0.5  # seconds between requests
last_request_time = 0

def rate_limited_request():
    """Ensure minimum delay between requests."""
    global last_request_time
    current_time = time.time()
    time_since_last = current_time - last_request_time
    if time_since_last < REQUEST_DELAY:
        time.sleep(REQUEST_DELAY - time_since_last)
    last_request_time = time.time()

def get_model_response(prompt: str, model_name: str) -> str:
    """Get a response from the specified model."""
    try:
        rate_limited_request()  # Rate limit all requests
        
        if "gpt" in model_name.lower():
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content
        
        elif "claude" in model_name.lower():
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            response = client.messages.create(
                model=model_name,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        
        else:
            raise ValueError(f"Unsupported model: {model_name}")
            
    except Exception as e:
        logging.error(f"Error getting response from {model_name}: {e}")
        return f"[Error: {str(e)}]"

def evaluate_model(
    model_name: str,
    prompts: List[Dict],
    output_dir: str,
    rewrite: bool = False,
    perspective_shift: bool = False,
    perspective_test_frequency: int = 5  # New parameter for configurable frequency
) -> str:
    """
    Evaluate a single model and return results path.
    
    Args:
        model_name: Name of the model to evaluate
        prompts: List of prompts to evaluate
        output_dir: Directory to save results
        rewrite: Whether to use constitutional rewriting
        perspective_shift: Whether to perform perspective shift testing
        perspective_test_frequency: Test perspective shifts every N prompts (default=5)
    """
    
    # Create model-specific output directory
    model_dir = os.path.join(output_dir, model_name.replace("-", "_"))
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize evaluator and rewriter
    evaluator = AlignmentEvaluator(model_name=model_name)
    rewriter = ConstitutionalRewriter() if rewrite else None
    
    # Track requests and their purposes
    request_log = []
    
    # Process each prompt
    for i, prompt_data in enumerate(prompts):
        prompt = prompt_data["prompt"]
        category = prompt_data["category"]
        expected = prompt_data.get("expected_behavior", "")
        
        logging.info(f"[{model_name}] Processing prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
        
        # Optionally rewrite prompt
        if rewriter:
            rewrite_result = rewriter.rewrite_prompt(
                prompt,
                lambda p: get_model_response(p, model_name)  # Use same model for rewriting
            )
            request_log.append({
                "purpose": "prompt_rewrite",
                "original": prompt,
                "rewritten": rewrite_result["final_prompt"],
                "rules_triggered": rewrite_result["rules_triggered"]
            })
            
            if rewrite_result["improved"]:
                logging.info(f"[{model_name}] Prompt rewritten due to {', '.join(rewrite_result['rules_triggered'])}")
                prompt = rewrite_result["final_prompt"]
        
        # Get model response
        response = get_model_response(prompt, model_name)
        request_log.append({
            "purpose": "main_response",
            "prompt": prompt,
            "category": category
        })
        
        # Evaluate response
        result = evaluator.evaluate_prompt(prompt, category, response)
        
        # Compare with expected behavior
        if expected:
            logging.info(f"Expected behavior: {expected}")
        
        # Optionally do perspective shift testing
        if perspective_shift and i % perspective_test_frequency == 0:
            # Use all available perspectives from evaluator
            for perspective in evaluator.perspectives:
                if perspective == "default":
                    continue  # Skip default as it's already tested
                    
                modified_prompt = f"Respond as {perspective}: {prompt}"
                response = get_model_response(modified_prompt, model_name)
                request_log.append({
                    "purpose": f"perspective_shift_{perspective}",
                    "prompt": modified_prompt
                })
                evaluator.evaluate_prompt(modified_prompt, category, response, perspective)
            
            logging.info(f"[{model_name}] Completed perspective testing for prompt {i+1}")
    
    # Export results
    results_path = os.path.join(model_dir, "evaluation_results.csv")
    evaluator.export_results(results_path)
    
    # Export rewrite history if available
    if rewriter:
        rewrite_path = os.path.join(model_dir, "rewrite_history.json")
        rewriter.export_history(rewrite_path)
    
    # Export request log
    request_log_path = os.path.join(model_dir, "request_log.json")
    with open(request_log_path, 'w') as f:
        json.dump({
            "total_requests": len(request_log),
            "requests": request_log
        }, f, indent=2)
    
    logging.info(f"\nTotal API requests made: {len(request_log)}")
    
    return results_path

def run_all_experiments(prompts_file: str = "prompts/eval_prompts.csv", output_dir: str = "results"):
    """Run all experiments with and without rewrite for both models."""
    models = ["gpt-4", "claude-3-opus-20240229"]
    results = {}
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load prompts
    evaluator = AlignmentEvaluator()
    prompts = evaluator.load_prompts(prompts_file)
    
    for model in models:
        # Run without rewrite
        logging.info(f"\nRunning {model} without rewrite...")
        results[f"{model}_base"] = evaluate_model(
            model, prompts, output_dir, 
            rewrite=False, 
            perspective_shift=True  # Enable perspective shift by default
        )
        
        # Run with rewrite
        logging.info(f"\nRunning {model} with rewrite...")
        results[f"{model}_rewrite"] = evaluate_model(
            model, prompts, output_dir, 
            rewrite=True,
            perspective_shift=True  # Enable perspective shift by default
        )
    
    # Generate comprehensive report with plots
    report_file = os.path.join(output_dir, "comprehensive_analysis.md")
    generate_comprehensive_report(results, report_file)
    
    logging.info("\nAll experiments complete! Results available at:")
    logging.info(f"1. Comprehensive analysis: {report_file}")
    logging.info(f"2. Plots: {plots_dir}")
    logging.info("3. Interactive dashboard: streamlit run dashboard/streamlit_app.py")
    
    return results

def generate_comprehensive_report(results_paths: dict, output_file: str = "results/comprehensive_analysis.md"):
    """Generate a comprehensive analysis report comparing all experiments."""
    analyzer = AlignmentAnalyzer()
    
    # Load all results
    for exp_name, path in results_paths.items():
        analyzer.add_model_results(path)
    
    report = ["# Comprehensive LLM Alignment Analysis\n"]
    
    # Overview section
    report.append("## Overview")
    report.append("This report compares the performance of different models with and without constitutional rewriting.\n")
    
    # Experiments summary
    report.append("### Experiments Conducted")
    for exp_name in results_paths.keys():
        report.append(f"- {exp_name}")
    report.append("")
    
    # Overall scores comparison
    report.append("## Overall Alignment Scores")
    report.append("![Dimension Scores](plots/dimension_scores_comparison.png)\n")
    report.append("### Key Findings")
    
    # Add model-specific analysis
    report.append("\n## Model-Specific Analysis")
    for exp_name, path in results_paths.items():
        df = pd.read_csv(path)
        score_cols = [col for col in df.columns if col.startswith('scores.')]
        overall_score = df[score_cols].mean().mean()
        
        report.append(f"\n### {exp_name}")
        report.append(f"- Overall alignment score: {overall_score:.2f}/3")
        report.append("- Dimension scores:")
        for col in score_cols:
            dim = col.replace('scores.', '')
            score = df[col].mean()
            report.append(f"  - {dim}: {score:.2f}/3")
        
        # Category performance
        report.append("\nPerformance by category:")
        for cat in df['category'].unique():
            cat_score = df[df['category'] == cat][score_cols].mean().mean()
            report.append(f"- {cat}: {cat_score:.2f}/3")
    
    # Rewrite analysis
    report.append("\n## Constitutional Rewriting Impact")
    for model in ["gpt-4", "claude-3-opus-20240229"]:
        report.append(f"\n### {model}")
        base_path = os.path.join(output_dir, f"{model.replace('-', '_')}")
        rewrite_file = os.path.join(base_path, "rewrite_history.json")
        
        if os.path.exists(rewrite_file):
            with open(rewrite_file) as f:
                rewrite_data = json.load(f)
            
            total_rewrites = len([r for r in rewrite_data if r.get("improved", False)])
            report.append(f"- Total prompts rewritten: {total_rewrites}")
            report.append("- Rules triggered:")
            rules_triggered = {}
            for entry in rewrite_data:
                for rule in entry.get("rules_triggered", []):
                    rules_triggered[rule] = rules_triggered.get(rule, 0) + 1
            for rule, count in rules_triggered.items():
                report.append(f"  - {rule}: {count} times")
    
    # API usage analysis
    report.append("\n## API Usage Analysis")
    for exp_name, path in results_paths.items():
        base_dir = os.path.dirname(path)
        request_file = os.path.join(base_dir, "request_log.json")
        
        if os.path.exists(request_file):
            with open(request_file) as f:
                request_data = json.load(f)
            
            report.append(f"\n### {exp_name}")
            report.append(f"- Total API requests: {request_data.get('total_requests', 0)}")
            
            # Analyze request purposes
            purposes = {}
            for req in request_data.get('requests', []):
                purpose = req.get('purpose', 'unknown')
                purposes[purpose] = purposes.get(purpose, 0) + 1
            
            report.append("Request breakdown:")
            for purpose, count in purposes.items():
                report.append(f"- {purpose}: {count}")
    
    # Write report
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
    
    logging.info(f"Comprehensive analysis saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="LLM Alignment Evaluator")
    parser.add_argument("--model", type=str, help="Model to evaluate (e.g., gpt-4, claude-3-opus-20240229)")
    parser.add_argument("--prompts", type=str, default="prompts/eval_prompts.csv", help="Path to prompts file")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory for results")
    parser.add_argument("--rewrite", action="store_true", help="Enable constitutional rewriting")
    parser.add_argument("--perspective-shift", action="store_true", help="Enable perspective shift testing")
    parser.add_argument("--perspective-freq", type=int, default=5, help="Test perspectives every N prompts")
    parser.add_argument("--run-all", action="store_true", help="Run all experiments (both models, with/without rewrite)")
    
    args = parser.parse_args()
    
    if args.run_all:
        results = run_all_experiments(args.prompts, args.output_dir)
        generate_comprehensive_report(results)
    elif args.model:
        evaluate_model(
            args.model,
            AlignmentEvaluator().load_prompts(args.prompts),
            args.output_dir,
            rewrite=args.rewrite,
            perspective_shift=args.perspective_shift,
            perspective_test_frequency=args.perspective_freq
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 