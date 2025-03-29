import os
import argparse
import logging
from typing import List, Dict
from dotenv import load_dotenv
import openai
import anthropic
from evaluator import AlignmentEvaluator
from constitutional_rewriter import ConstitutionalRewriter
from analyze_results import AlignmentAnalyzer

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_model_response(prompt: str, model_name: str) -> str:
    """Get a response from the specified model."""
    try:
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
            client = anthropic.Client(os.getenv("ANTHROPIC_API_KEY"))
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
    perspective_shift: bool = False
) -> str:
    """Evaluate a single model and return results path."""
    
    # Create model-specific output directory
    model_dir = os.path.join(output_dir, model_name.replace("-", "_"))
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize evaluator and rewriter
    evaluator = AlignmentEvaluator(model_name=model_name)
    rewriter = ConstitutionalRewriter() if rewrite else None
    
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
                lambda p: get_model_response(p, model_name)
            )
            
            if rewrite_result["improved"]:
                logging.info(f"[{model_name}] Prompt rewritten due to {', '.join(rewrite_result['rules_triggered'])}")
                original_prompt = prompt
                prompt = rewrite_result["final_prompt"]
                
                # Evaluate both original and rewritten
                original_response = get_model_response(original_prompt, model_name)
                evaluator.evaluate_prompt(original_prompt, category, original_response)
        
        # Get model response
        response = get_model_response(prompt, model_name)
        
        # Evaluate response
        result = evaluator.evaluate_prompt(prompt, category, response)
        
        # Compare with expected behavior
        if expected:
            logging.info(f"Expected behavior: {expected}")
            # TODO: Add automated comparison metrics
        
        # Optionally do perspective shift testing
        if perspective_shift and i % 3 == 0:  # Test every third prompt
            perspective_result = evaluator.perspective_shift_test(
                prompt,
                lambda p: get_model_response(p, model_name)
            )
            logging.info(f"[{model_name}] Perspective drift: {perspective_result['perspective_drift']}")
    
    # Export results
    results_path = os.path.join(model_dir, "evaluation_results.csv")
    evaluator.export_results(results_path)
    
    if rewriter:
        rewrite_path = os.path.join(model_dir, "rewrite_history.json")
        rewriter.export_history(rewrite_path)
    
    return results_path

def main():
    parser = argparse.ArgumentParser(description="LLM Alignment Evaluator")
    parser.add_argument("--prompts", type=str, default="prompts/eval_prompts.csv",
                      help="Path to prompts file")
    parser.add_argument("--models", type=str, default="gpt-3.5-turbo",
                      help="Comma-separated list of models to evaluate")
    parser.add_argument("--output", type=str, default="results",
                      help="Output directory")
    parser.add_argument("--rewrite", action="store_true",
                      help="Enable constitutional rewriting")
    parser.add_argument("--perspective-shift", action="store_true",
                      help="Enable perspective shift testing")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load prompts
    evaluator = AlignmentEvaluator()
    prompts = evaluator.load_prompts(args.prompts)
    logging.info(f"Loaded {len(prompts)} prompts from {args.prompts}")
    
    # Process each model
    models = [m.strip() for m in args.models.split(",")]
    results_paths = []
    
    for model_name in models:
        logging.info(f"\nEvaluating model: {model_name}")
        results_path = evaluate_model(
            model_name,
            prompts,
            args.output,
            args.rewrite,
            args.perspective_shift
        )
        results_paths.append(results_path)
        
    # If multiple models, generate comparative analysis
    if len(models) > 1:
        logging.info("\nGenerating comparative analysis...")
        analyzer = AlignmentAnalyzer(results_paths[0])  # Initialize with first model
        
        # Load additional models
        for path in results_paths[1:]:
            analyzer.add_model_results(path)
            
        # Generate comparative report
        report_path = os.path.join(args.output, "comparative_analysis.md")
        analyzer.generate_comparative_report(report_path)
        logging.info(f"Comparative analysis saved to {report_path}")
    
    logging.info("\nEvaluation complete! To view results:")
    logging.info(f"1. Check individual results in {args.output}/<model_name>/")
    logging.info("2. Run the dashboard: streamlit run dashboard/streamlit_app.py")

if __name__ == "__main__":
    main() 