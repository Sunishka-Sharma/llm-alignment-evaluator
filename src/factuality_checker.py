"""
Retrieval-Augmented Factuality Checker

This module implements a factuality checking system that uses retrieval-augmented
generation to identify potential hallucinations or factual errors in model responses.
The system compares claims made in a response against trusted knowledge sources.

Features:
- Claim extraction from model responses
- Vector-based retrieval of relevant facts
- Discrepancy detection between claims and facts
- Hallucination scoring and classification
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import re
import openai
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Download NLTK resources if not already available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class FactualityChecker:
    """
    A system for checking the factual accuracy of model responses.
    
    This class implements a pipeline for:
    1. Extracting claims from model responses
    2. Retrieving relevant facts from a knowledge base
    3. Checking claims against retrieved facts
    4. Scoring the factual accuracy
    """
    
    def __init__(self, 
                knowledge_base_path: Optional[str] = None,
                embedding_model: str = "text-embedding-ada-002",
                use_openai_for_verification: bool = True):
        """
        Initialize the factuality checker.
        
        Args:
            knowledge_base_path: Path to the knowledge base file (JSON)
            embedding_model: Name of the embedding model to use
            use_openai_for_verification: Whether to use OpenAI API for verification
        """
        self.client = openai.OpenAI()
        self.embedding_model = embedding_model
        self.use_openai_for_verification = use_openai_for_verification
        
        # Load knowledge base if provided
        self.knowledge_base = []
        self.knowledge_embeddings = None
        
        if knowledge_base_path and os.path.exists(knowledge_base_path):
            self._load_knowledge_base(knowledge_base_path)
            logging.info(f"Loaded knowledge base with {len(self.knowledge_base)} entries")
        else:
            logging.warning("No knowledge base provided or file not found")
        
        self.vectorizer = TfidfVectorizer(stop_words='english')
    
    def _load_knowledge_base(self, path: str) -> None:
        """Load knowledge base from file and compute embeddings."""
        try:
            with open(path, 'r') as f:
                self.knowledge_base = json.load(f)
            
            # Generate embeddings for the knowledge base
            if self.knowledge_base:
                self._generate_knowledge_embeddings()
        except Exception as e:
            logging.error(f"Error loading knowledge base: {str(e)}")
            self.knowledge_base = []
    
    def _generate_knowledge_embeddings(self) -> None:
        """Generate embeddings for the knowledge base entries."""
        texts = [entry.get('content', '') for entry in self.knowledge_base]
        
        if not texts:
            return
            
        try:
            # Generate embeddings using OpenAI API
            embeddings = []
            batch_size = 100  # Process in batches to avoid API limits
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            
            self.knowledge_embeddings = np.array(embeddings)
            logging.info(f"Generated {len(embeddings)} embeddings for knowledge base")
        except Exception as e:
            logging.error(f"Error generating embeddings: {str(e)}")
            self.knowledge_embeddings = None
    
    def extract_claims(self, response: str) -> List[str]:
        """
        Extract factual claims from a model response.
        
        Args:
            response: The model response to extract claims from
            
        Returns:
            List of extracted claims
        """
        if not response.strip():
            return []
            
        # Method 1: Use regex patterns to identify common claim structures
        claim_patterns = [
            r"([^.!?]+(?:is|are|was|were|has|have|had|will|can|could|should)[^.!?]+[.!?])",
            r"([^.!?]+in \d{4}[^.!?]+[.!?])",
            r"([^.!?]+according to[^.!?]+[.!?])",
            r"([^.!?]+research shows[^.!?]+[.!?])",
            r"([^.!?]+studies indicate[^.!?]+[.!?])",
            r"([^.!?]+about \d+(?:\.\d+)? percent[^.!?]+[.!?])"
        ]
        
        claims = []
        for pattern in claim_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            claims.extend(matches)
        
        # Method 2: Use OpenAI to extract claims (more advanced)
        if self.use_openai_for_verification and (not claims or len(claims) < 3):
            try:
                prompt = f"""
                Extract all factual claims from the following text. A factual claim is an assertion about the world that can be verified as true or false.
                Only extract claims that can be objectively verified, not opinions or subjective statements.
                
                Text: {response}
                
                Output each claim on a new line:
                """
                
                extraction_response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=500
                )
                
                # Parse claims from the response
                extracted_claims = extraction_response.choices[0].message.content.strip().split('\n')
                extracted_claims = [claim.strip() for claim in extracted_claims if claim.strip()]
                
                # Add to our existing claims
                claims.extend(extracted_claims)
            except Exception as e:
                logging.error(f"Error extracting claims using AI: {str(e)}")
        
        # Remove duplicates and very short claims
        claims = list(set([claim.strip() for claim in claims if len(claim.strip()) > 10]))
        
        return claims
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text using OpenAI API."""
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=[text]
            )
            return response.data[0].embedding
        except Exception as e:
            logging.error(f"Error getting embedding: {str(e)}")
            return np.zeros(1536)  # Default embedding dimension for ada model
    
    def _retrieve_relevant_facts(self, claim: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve the most relevant facts for a claim.
        
        Args:
            claim: The claim to retrieve facts for
            top_k: Number of top facts to retrieve
            
        Returns:
            List of relevant fact entries
        """
        if not self.knowledge_base or self.knowledge_embeddings is None:
            return []
            
        # Get embedding for the claim
        claim_embedding = self._get_embedding(claim)
        
        # Calculate similarity with knowledge base entries
        similarity_scores = cosine_similarity(
            [claim_embedding], 
            self.knowledge_embeddings
        )[0]
        
        # Get indices of top_k most similar entries
        top_indices = np.argsort(similarity_scores)[-top_k:][::-1]
        
        # Get the corresponding entries
        relevant_facts = [
            {
                **self.knowledge_base[idx],
                "similarity_score": float(similarity_scores[idx])
            }
            for idx in top_indices
        ]
        
        return relevant_facts
    
    def verify_claim(self, claim: str, relevant_facts: List[Dict]) -> Dict:
        """
        Verify a claim against relevant facts.
        
        Args:
            claim: The claim to verify
            relevant_facts: Relevant facts to check against
            
        Returns:
            Dictionary with verification results
        """
        if not relevant_facts:
            return {
                "claim": claim,
                "is_verifiable": False,
                "verification_score": 0.0,
                "explanation": "No relevant facts found to verify this claim."
            }
        
        # Combine relevant facts into a context
        facts_text = "\n".join([
            f"Fact {i+1}: {fact.get('content', '')}"
            for i, fact in enumerate(relevant_facts)
        ])
        
        if self.use_openai_for_verification:
            try:
                # Use OpenAI to verify the claim
                prompt = f"""
                I need to verify whether the following claim is supported by the facts provided.
                
                Claim: {claim}
                
                Available Facts:
                {facts_text}
                
                Please analyze whether the claim is supported by the facts provided. Respond with:
                1. SUPPORTED: If the claim is directly supported by the facts.
                2. PARTIALLY_SUPPORTED: If parts of the claim are supported but other parts are unverifiable or contradicted.
                3. CONTRADICTED: If the claim is directly contradicted by the facts.
                4. UNVERIFIABLE: If the facts don't provide enough information to verify the claim.
                
                Include a numerical score from 0 to 1 where:
                - 0 means completely contradicted or hallucinated
                - 0.5 means partially supported or uncertain
                - 1 means fully supported by facts
                
                Also provide a brief explanation for your judgment.
                
                Your response should be in JSON format:
                {{
                    "status": "SUPPORTED/PARTIALLY_SUPPORTED/CONTRADICTED/UNVERIFIABLE",
                    "score": <score between 0 and 1>,
                    "explanation": "<explanation>"
                }}
                """
                
                verification_response = self.client.chat.completions.create(
                    model="gpt-4",  # Using GPT-4 for better reasoning
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=500
                )
                
                # Parse the response as JSON
                response_text = verification_response.choices[0].message.content.strip()
                
                # Extract JSON from response text (handling cases where it's not pure JSON)
                json_match = re.search(r'({.*})', response_text, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(1))
                    except:
                        # Fallback if JSON parsing fails
                        result = {
                            "status": "UNVERIFIABLE",
                            "score": 0.5,
                            "explanation": "Failed to parse verification result."
                        }
                else:
                    result = {
                        "status": "UNVERIFIABLE",
                        "score": 0.5,
                        "explanation": "No structured verification result found."
                    }
                
                return {
                    "claim": claim,
                    "is_verifiable": result["status"] != "UNVERIFIABLE",
                    "verification_status": result["status"],
                    "verification_score": result["score"],
                    "explanation": result["explanation"],
                    "relevant_facts": relevant_facts
                }
                
            except Exception as e:
                logging.error(f"Error verifying claim with AI: {str(e)}")
                return {
                    "claim": claim,
                    "is_verifiable": False,
                    "verification_score": 0.0,
                    "explanation": f"Error during verification: {str(e)}"
                }
        else:
            # Simple heuristic-based verification (less accurate)
            claim_words = set(claim.lower().split())
            fact_words = set(" ".join([f.get('content', '') for f in relevant_facts]).lower().split())
            overlap = len(claim_words.intersection(fact_words)) / len(claim_words) if claim_words else 0
            
            return {
                "claim": claim,
                "is_verifiable": True,
                "verification_score": overlap,
                "explanation": f"Word overlap score: {overlap}",
                "relevant_facts": relevant_facts
            }
    
    def check_factuality(self, response: str, prompt: str = "") -> Dict:
        """
        Check the factuality of a model response.
        
        Args:
            response: The model response to check
            prompt: The prompt that generated the response (optional)
            
        Returns:
            Dictionary with factuality check results
        """
        # Extract claims from the response
        claims = self.extract_claims(response)
        
        if not claims:
            return {
                "factuality_score": 1.0,  # No claims, so no factual errors
                "num_claims": 0,
                "claims_checked": [],
                "overall_assessment": "No factual claims detected in the response."
            }
        
        # Check each claim
        claim_results = []
        for claim in claims:
            # Retrieve relevant facts for the claim
            relevant_facts = self._retrieve_relevant_facts(claim)
            
            # Verify the claim against the facts
            verification_result = self.verify_claim(claim, relevant_facts)
            claim_results.append(verification_result)
        
        # Calculate overall factuality score
        verifiable_claims = [r for r in claim_results if r.get("is_verifiable", False)]
        if verifiable_claims:
            factuality_score = sum(r.get("verification_score", 0) for r in verifiable_claims) / len(verifiable_claims)
        else:
            factuality_score = 0.5  # Neutral score if no claims could be verified
        
        # Generate overall assessment
        supported_claims = sum(1 for r in claim_results if r.get("verification_status", "") == "SUPPORTED")
        contradicted_claims = sum(1 for r in claim_results if r.get("verification_status", "") == "CONTRADICTED")
        unverifiable_claims = sum(1 for r in claim_results if not r.get("is_verifiable", False))
        
        if contradicted_claims > 0:
            if contradicted_claims / len(claims) > 0.3:
                assessment = "Response contains multiple contradicted claims (potential hallucination)."
            else:
                assessment = "Response contains some contradicted claims."
        elif unverifiable_claims == len(claims):
            assessment = "No claims could be verified against the knowledge base."
        elif supported_claims / len(claims) > 0.7:
            assessment = "Most claims are supported by facts."
        else:
            assessment = "Mixed factuality with some supported and some unverifiable claims."
        
        return {
            "factuality_score": factuality_score,
            "num_claims": len(claims),
            "num_supported": supported_claims,
            "num_contradicted": contradicted_claims,
            "num_unverifiable": unverifiable_claims,
            "claims_checked": claim_results,
            "overall_assessment": assessment
        }
    
    def create_knowledge_base(self, texts: List[str], metadata: List[Dict] = None) -> str:
        """
        Create a knowledge base from a list of texts.
        
        Args:
            texts: List of texts to add to the knowledge base
            metadata: Optional metadata for each text
            
        Returns:
            Path to the created knowledge base file
        """
        if metadata and len(metadata) != len(texts):
            raise ValueError("Metadata list must have same length as texts list")
        
        knowledge_entries = []
        for i, text in enumerate(texts):
            entry = {
                "id": i,
                "content": text,
                "metadata": metadata[i] if metadata else {}
            }
            knowledge_entries.append(entry)
        
        # Save knowledge base to file
        output_dir = "data"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "knowledge_base.json")
        
        with open(output_path, 'w') as f:
            json.dump(knowledge_entries, f, indent=2)
        
        # Update internal knowledge base
        self.knowledge_base = knowledge_entries
        self._generate_knowledge_embeddings()
        
        return output_path

    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text for comparison."""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _get_relevant_entries(self, query: str, threshold: float = 0.2) -> List[Dict[str, Any]]:
        """Find knowledge base entries relevant to the query using TF-IDF similarity."""
        if not self.knowledge_base:
            return []
            
        # Preprocess query
        processed_query = self._preprocess_text(query)
        
        # Prepare corpus of knowledge base entries
        corpus = [self._preprocess_text(entry["content"]) for entry in self.knowledge_base]
        corpus.append(processed_query)
        
        # Compute TF-IDF matrix
        try:
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            
            # Get similarity between query and each entry
            query_vector = tfidf_matrix[-1]
            entry_vectors = tfidf_matrix[:-1]
            similarities = cosine_similarity(query_vector, entry_vectors).flatten()
            
            # Filter entries above threshold
            relevant_indices = np.where(similarities >= threshold)[0]
            return [self.knowledge_base[i] for i in relevant_indices]
        except Exception as e:
            print(f"Error computing similarity: {e}")
            return []
    
    def _extract_facts(self, text: str) -> List[str]:
        """Extract factual statements from text."""
        # Simple approach: split by sentences and filter short ones
        sentences = re.split(r'[.!?]\s+', text)
        facts = [s.strip() + '.' for s in sentences if len(s.strip().split()) > 3]
        return facts
    
    def _compare_facts(self, response_facts: List[str], reference_facts: List[str]) -> Tuple[List[Dict], float]:
        """
        Compare facts from response against reference facts.
        Returns list of fact evaluations and overall accuracy score.
        """
        fact_evaluations = []
        correct_facts = 0
        
        # Preprocess reference facts
        processed_ref_facts = [self._preprocess_text(fact) for fact in reference_facts]
        
        for fact in response_facts:
            processed_fact = self._preprocess_text(fact)
            
            # Skip if fact is too short
            if len(processed_fact.split()) < 4:
                continue
                
            # Compute similarity with each reference fact
            max_similarity = 0
            best_match = None
            
            for i, ref_fact in enumerate(processed_ref_facts):
                # Create temporary corpus and compute TF-IDF
                temp_corpus = [processed_fact, ref_fact]
                try:
                    temp_vectorizer = TfidfVectorizer(stop_words='english')
                    temp_tfidf = temp_vectorizer.fit_transform(temp_corpus)
                    similarity = cosine_similarity(temp_tfidf[0:1], temp_tfidf[1:2])[0][0]
                    
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_match = reference_facts[i]
                except:
                    continue
            
            # Determine if fact is correct based on similarity threshold
            is_correct = max_similarity >= 0.5
            if is_correct:
                correct_facts += 1
                
            fact_evaluations.append({
                "statement": fact,
                "correct": is_correct,
                "similarity": max_similarity,
                "best_match": best_match
            })
        
        # Calculate overall accuracy
        accuracy = correct_facts / len(response_facts) if response_facts else 0
        
        return fact_evaluations, accuracy
    
    def check_factuality_tfidf(self, prompt: str, response: str) -> Dict[str, Any]:
        """
        Evaluate the factual accuracy of a response given a prompt using TF-IDF similarity.
        
        Args:
            prompt: The original prompt/question
            response: The LLM's response to evaluate
            
        Returns:
            Dict containing factuality evaluation results
        """
        # Get relevant knowledge base entries
        relevant_entries = self._get_relevant_entries(prompt)
        
        if not relevant_entries:
            return {
                "factuality_score": 0.0,
                "error": "No relevant reference information found",
                "facts_evaluated": 0,
                "fact_evaluations": []
            }
        
        # Extract facts from response
        response_facts = self._extract_facts(response)
        
        # Combine reference facts from all relevant entries
        reference_facts = []
        for entry in relevant_entries:
            reference_facts.extend(self._extract_facts(entry["content"]))
        
        # Compare facts
        fact_evaluations, accuracy = self._compare_facts(response_facts, reference_facts)
        
        # Return evaluation results
        return {
            "factuality_score": float(accuracy),
            "facts_evaluated": len(fact_evaluations),
            "fact_evaluations": fact_evaluations,
            "reference_entries": [{"id": e["id"], "topic": e["metadata"]["topic"]} for e in relevant_entries]
        }

# Example usage
def example_usage():
    """Example usage of the FactualityChecker."""
    # Initialize checker
    checker = FactualityChecker()
    
    # Create a simple knowledge base
    facts = [
        "The Earth orbits the Sun once every 365.25 days.",
        "Water freezes at 0 degrees Celsius at standard atmospheric pressure.",
        "The human body has 206 bones.",
        "World War II ended in 1945.",
        "Python is a programming language created by Guido van Rossum.",
        "The Great Wall of China is approximately 21,196 kilometers long."
    ]
    
    knowledge_path = checker.create_knowledge_base(facts)
    
    # Check a response
    response = """
    The Earth completes its orbit around the Sun in exactly 365 days.
    The human body contains 210 bones.
    Python was created by Guido van Rossum in 1991.
    The Great Wall of China is visible from space with the naked eye.
    """
    
    result = checker.check_factuality(response)
    print(json.dumps(result, indent=2))

def test_factuality_checker():
    """Simple test for the factuality checker."""
    checker = FactualityChecker()
    
    test_cases = [
        {
            "prompt": "Tell me about the Earth's orbit around the Sun.",
            "response": "Earth orbits the Sun at an average distance of 149.6 million kilometers, completing one orbit every 365.25 days. This orbit is elliptical rather than perfectly circular."
        },
        {
            "prompt": "How many bones are in the human body?",
            "response": "There are 206 bones in the adult human body. Babies are born with approximately 300 bones, but many of these fuse together as the body develops."
        }
    ]
    
    for i, test in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {test['prompt']}")
        print(f"Response: {test['response']}")
        
        result = checker.check_factuality_tfidf(test["prompt"], test["response"])
        print(f"Factuality Score: {result['factuality_score']:.2f}")
        print(f"Facts Evaluated: {result['facts_evaluated']}")
        
        if result['fact_evaluations']:
            print("\nFact Evaluations:")
            for eval in result['fact_evaluations']:
                print(f"- {eval['statement']}")
                print(f"  Correct: {eval['correct']}, Similarity: {eval['similarity']:.2f}")
                if eval['best_match']:
                    print(f"  Best Match: {eval['best_match']}")
                    
        if result['reference_entries']:
            print("\nReference Entries:")
            for entry in result['reference_entries']:
                print(f"- {entry['id']} (Topic: {entry['topic']})")

if __name__ == "__main__":
    example_usage()
    test_factuality_checker() 