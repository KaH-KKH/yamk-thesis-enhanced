# src/utils/metrics.py
"""
Metrics calculation utilities
"""

import torch
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class MetricsCalculator:
    """Calculate various metrics for evaluation"""
    
    def __init__(self):
        self.sentence_model = None
    
    def calculate_semantic_similarity(self, texts1: List[str], texts2: List[str]) -> float:
        """Calculate semantic similarity between two sets of texts"""
        if self.sentence_model is None:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Encode texts
        embeddings1 = self.sentence_model.encode(texts1)
        embeddings2 = self.sentence_model.encode(texts2)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(embeddings1, embeddings2)
        
        # Return average similarity
        return float(np.mean(similarities))
    
    def calculate_test_coverage(self, use_case: str, test_case: str) -> Dict[str, float]:
        """Calculate how well test case covers use case"""
        coverage = {
            "actor_coverage": 0.0,
            "action_coverage": 0.0,
            "flow_coverage": 0.0
        }
        
        # Extract elements from use case
        use_case_lower = use_case.lower()
        test_case_lower = test_case.lower()
        
        # Actor coverage
        actors = self._extract_actors(use_case)
        covered_actors = sum(1 for actor in actors if actor.lower() in test_case_lower)
        coverage["actor_coverage"] = covered_actors / len(actors) if actors else 0
        
        # Action coverage
        actions = self._extract_actions(use_case)
        covered_actions = sum(1 for action in actions if action.lower() in test_case_lower)
        coverage["action_coverage"] = covered_actions / len(actions) if actions else 0
        
        # Flow coverage (steps)
        flow_steps = self._extract_flow_steps(use_case)
        covered_steps = sum(1 for step in flow_steps if self._step_covered(step, test_case_lower))
        coverage["flow_coverage"] = covered_steps / len(flow_steps) if flow_steps else 0
        
        return coverage
    
    def _extract_actors(self, text: str) -> List[str]:
        """Extract actors from text"""
        import re
        actors = []
        
        # Look for ACTORS section
        actors_match = re.search(r'ACTORS:(.*?)(?:PRECONDITIONS|MAIN FLOW|$)', text, re.DOTALL | re.IGNORECASE)
        if actors_match:
            actors_text = actors_match.group(1)
            actors = re.findall(r'[-\*]\s*(.+)', actors_text)
        
        return [a.strip() for a in actors]
    
    def _extract_actions(self, text: str) -> List[str]:
        """Extract actions from text"""
        import re
        action_verbs = ['login', 'click', 'enter', 'submit', 'verify', 'display', 'navigate']
        actions = []
        
        for verb in action_verbs:
            if verb in text.lower():
                actions.append(verb)
        
        return actions
    
    def _extract_flow_steps(self, text: str) -> List[str]:
        """Extract flow steps from use case"""
        import re
        steps = []
        
        # Look for numbered steps
        step_pattern = r'\d+\.\s*(.+?)(?=\d+\.|$)'
        matches = re.findall(step_pattern, text, re.DOTALL)
        steps.extend([m.strip() for m in matches])
        
        return steps
    
    def _step_covered(self, step: str, test_case: str) -> bool:
        """Check if a step is covered in test case"""
        # Simple keyword matching for now
        keywords = step.lower().split()
        significant_keywords = [kw for kw in keywords if len(kw) > 3]
        
        if not significant_keywords:
            return False
        
        # Check if at least 50% of significant keywords are in test case
        matches = sum(1 for kw in significant_keywords if kw in test_case)
        return matches >= len(significant_keywords) * 0.5
    