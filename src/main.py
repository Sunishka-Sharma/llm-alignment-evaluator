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
    perspective_shift: bool = False
) -> str:
    """Evaluate a single model and return results path."""
    
    # Create model-specific output directory
    model_dir = os.path.join(output_dir, model_name.replace("-", "_"))
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize evaluator and rewriter
    evaluator = AlignmentEvaluator(model_name=model_name)
    rewriter = ConstitutionalRewriter() if rewrite else None
    
    # Track total requests for logging
    total_requests = 0
    
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
            total_requests += 1  # Count rewrite request
            
            if rewrite_result["improved"]:
                logging.info(f"[{model_name}] Prompt rewritten due to {', '.join(rewrite_result['rules_triggered'])}")
                prompt = rewrite_result["final_prompt"]
        
        # Get model response
        response = get_model_response(prompt, model_name)
        total_requests += 1  # Count main response request
        
        # Evaluate response
        result = evaluator.evaluate_prompt(prompt, category, response)
        
        # Compare with expected behavior
        if expected:
            logging.info(f"Expected behavior: {expected}")
        
        # Optionally do perspective shift testing (reduced frequency)
        if perspective_shift and i % 5 == 0:  # Test every fifth prompt instead of third
            perspectives = ["child", "expert"]  # Reduced perspectives
            for perspective in perspectives:
                modified_prompt = f"Respond as {perspective}: {prompt}"
                response = get_model_response(modified_prompt, model_name)
                total_requests += 1
                evaluator.evaluate_prompt(modified_prompt, category, response, perspective)
            
            logging.info(f"[{model_name}] Completed perspective testing for prompt {i+1}")
    
    logging.info(f"\nTotal API requests made: {total_requests}")
    
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
    parser.add_argument("--model", type=str, default="claude-3-opus-20240229",
                      help="Model to evaluate (gpt-3.5-turbo or claude-3-opus-20240229)")
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
    
    # Process model
    logging.info(f"\nEvaluating model: {args.model}")
    results_path = evaluate_model(
        args.model,
        prompts,
        args.output,
        args.rewrite,
        args.perspective_shift
    )
        
    # Generate analysis report
    logging.info("\nGenerating analysis...")
    analyzer = AlignmentAnalyzer(results_path)
    
    # Generate report
    report_path = os.path.join(args.output, "analysis_report.md")
    analyzer.generate_comparative_report(report_path)
    logging.info(f"Analysis saved to {report_path}")
    
    logging.info("\nEvaluation complete! To view results:")
    logging.info(f"1. Check results in {args.output}/")
    logging.info("2. Run the dashboard: streamlit run dashboard/streamlit_app.py")

if __name__ == "__main__":
    main() 