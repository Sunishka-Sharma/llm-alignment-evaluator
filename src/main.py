import os
import argparse
import logging
from evaluator import AlignmentEvaluator
from constitutional_rewriter import ConstitutionalRewriter
from analyze_results import AlignmentAnalyzer

# Optional: If you want to use OpenAI's API
import openai

def get_model_response(prompt: str, model_name: str = "gpt-3.5-turbo") -> str:
    """Get a response from the specified model."""
    try:
        # This is just a placeholder - in a real implementation you'd use an actual API
        # Example using OpenAI:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error getting model response: {e}")
        return f"[Error generating response: {e}]"

def main():
    parser = argparse.ArgumentParser(description="Alignment Compass - LLM Evaluation Tool")
    parser.add_argument("--prompts", type=str, default="prompts/eval_prompts.csv",
                        help="Path to prompts file")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo",
                        help="Model to evaluate")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory")
    parser.add_argument("--rewrite", action="store_true",
                        help="Enable constitutional rewriting")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize evaluator
    evaluator = AlignmentEvaluator(model_name=args.model)
    
    # Load prompts
    prompts = evaluator.load_prompts(args.prompts)
    logging.info(f"Loaded {len(prompts)} prompts from {args.prompts}")
    
    # Initialize rewriter if needed
    rewriter = ConstitutionalRewriter() if args.rewrite else None
    
    # Process each prompt
    for i, prompt_data in enumerate(prompts):
        prompt = prompt_data["prompt"]
        category = prompt_data["category"]
        
        logging.info(f"Processing prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
        
        # Optionally rewrite prompt if it violates constitutional rules
        if rewriter:
            rewrite_result = rewriter.rewrite_prompt(
                prompt, 
                lambda p: get_model_response(p, args.model)
            )
            
            if rewrite_result["improved"]:
                logging.info(f"Prompt rewritten due to {', '.join(rewrite_result['rules_triggered'])}")
                original_prompt = prompt
                prompt = rewrite_result["final_prompt"]
                
                # Evaluate both original and rewritten
                original_response = get_model_response(original_prompt, args.model)
                evaluator.evaluate_prompt(original_prompt, category, original_response)
        
        # Get model response
        response = get_model_response(prompt, args.model)
        
        # Evaluate the response
        result = evaluator.evaluate_prompt(prompt, category, response)
        
        # For some prompts, do perspective shift testing
        if i % 3 == 0:  # Test every third prompt for perspective shifts
            perspective_result = evaluator.perspective_shift_test(
                prompt,
                lambda p: get_model_response(p, args.model)
            )
            logging.info(f"Perspective drift: {perspective_result['perspective_drift']}")
    
    # Export results
    results_path = os.path.join(args.output, "evaluation_results.csv")
    evaluator.export_results(results_path)
    
    if rewriter:
        rewrite_path = os.path.join(args.output, "rewrite_history.json")
        rewriter.export_history(rewrite_path)
    
    # Generate analysis
    analyzer = AlignmentAnalyzer(results_path)
    report_path = analyzer.generate_report(args.output)
    
    logging.info(f"Evaluation complete. Results in {args.output}")
    logging.info(f"Report generated at {report_path}")

if __name__ == "__main__":
    main() 