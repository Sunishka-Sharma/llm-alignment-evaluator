from typing import List, Dict, Tuple, Callable, Optional
import re
import logging
import json

class ConstitutionalRule:
    """A rule in the constitutional rewriting system."""
    
    def __init__(self, name: str, description: str, check_fn: Callable[[str], bool], 
                 rewrite_prompt: str):
        self.name = name
        self.description = description
        self.check = check_fn  # Function that returns True if rule is violated
        self.rewrite_prompt = rewrite_prompt
    
    def __repr__(self):
        return f"ConstitutionalRule({self.name})"


class ConstitutionalRewriter:
    """
    A system for rewriting prompts that might violate ethical principles
    using a constitution of rules.
    """
    
    def __init__(self):
        self.rules = self._initialize_rules()
        self.rewrite_history = []
        logging.info(f"Initialized ConstitutionalRewriter with {len(self.rules)} rules")
    
    def _initialize_rules(self) -> List[ConstitutionalRule]:
        """Initialize the constitutional rules."""
        rules = [
            ConstitutionalRule(
                name="harm_prevention",
                description="Rewrite prompts that could lead to harmful instructions",
                check_fn=lambda x: any(term in x.lower() for term in ["kill", "hurt", "harm", "weapon", "steal"]) 
                         and not any(safe in x.lower() for safe in ["prevent", "avoid", "don't", "stop"]),
                rewrite_prompt="Rewrite this prompt to prevent potential harm while preserving the user's information need: {prompt}"
            ),
            
            ConstitutionalRule(
                name="avoid_deception",
                description="Rewrite prompts that encourage deception",
                check_fn=lambda x: any(term in x.lower() for term in ["trick", "deceive", "manipulate", "fake"]) 
                         or ("lie" in x.lower() and not any(good in x.lower() for good in ["rely", "belief"])),
                rewrite_prompt="Rewrite this prompt to avoid encouraging deception while still addressing the user's underlying question: {prompt}"
            ),
            
            ConstitutionalRule(
                name="respect_autonomy",
                description="Rewrite prompts that may violate others' autonomy",
                check_fn=lambda x: any(term in x.lower() for term in ["force", "make them", "without consent", "against their will"])
                         and not any(good in x.lower() for good in ["enforce", "maker", "consent form"]),
                rewrite_prompt="Rewrite this prompt to respect individual autonomy while still addressing the user's question: {prompt}"
            ),
            
            ConstitutionalRule(
                name="perspective_broadening",
                description="Rewrite narrow ethical questions to consider multiple perspectives",
                check_fn=lambda x: re.search(r"(should|is it (okay|right|wrong|ethical))", x.lower()) 
                                and len(x.split()) < 15
                                and not any(term in x.lower() for term in ["why", "how", "what if", "explain"]),
                rewrite_prompt="This prompt asks for a narrow ethical judgment. Rewrite it to encourage considering multiple perspectives and stakeholders: {prompt}"
            )
        ]
        return rules
    
    def analyze_prompt(self, prompt: str) -> List[ConstitutionalRule]:
        """Identify which constitutional rules are triggered by this prompt."""
        triggered_rules = []
        for rule in self.rules:
            if rule.check(prompt):
                triggered_rules.append(rule)
        return triggered_rules
    
    def rewrite_prompt(self, 
                      prompt: str, 
                      model_rewrite_fn: Callable[[str], str], 
                      max_iterations: int = 2) -> Dict:
        """
        Rewrite a prompt using constitutional rules and the provided model function.
        """
        original_prompt = prompt
        iterations = []
        current_prompt = prompt
        
        for i in range(max_iterations):
            # Find triggered rules
            triggered_rules = self.analyze_prompt(current_prompt)
            if not triggered_rules:
                break
                
            # Choose highest priority rule
            rule = triggered_rules[0]
            
            # Generate rewrite instruction
            rewrite_instruction = rule.rewrite_prompt.format(prompt=current_prompt)
            
            try:
                # Get rewritten prompt using provided model function
                rewritten_prompt = model_rewrite_fn(rewrite_instruction)
            except Exception as e:
                logging.error(f"Error in rewrite_prompt: {str(e)}")
                rewritten_prompt = current_prompt
            
            # Store iteration
            iterations.append({
                "iteration": i+1,
                "rule_applied": rule.name,
                "original": current_prompt,
                "rewritten": rewritten_prompt
            })
            
            # Update current prompt for next iteration
            current_prompt = rewritten_prompt
        
        result = {
            "original_prompt": original_prompt,
            "final_prompt": current_prompt,
            "rules_triggered": [r.name for r in triggered_rules],
            "iterations": iterations,
            "improved": current_prompt != original_prompt
        }
        
        # Store in history
        self.rewrite_history.append(result)
        return result
    
    def export_history(self, file_path: str) -> None:
        """Export rewrite history to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.rewrite_history, f, indent=2)
        logging.info(f"Exported rewrite history ({len(self.rewrite_history)} entries) to {file_path}") 