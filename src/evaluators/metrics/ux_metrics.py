# src/evaluators/metrics/ux_metrics.py
"""
User Experience metrics for evaluating generated content quality
"""

import re
import textstat
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import Counter
from loguru import logger
import nltk

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass


class UserExperienceMetrics:
    """Calculate UX-focused metrics for generated content"""
    
    def __init__(self):
        # Action verbs that indicate clear instructions
        self.action_verbs = {
            'click', 'enter', 'type', 'select', 'navigate', 'verify', 'check',
            'submit', 'login', 'logout', 'upload', 'download', 'create',
            'update', 'delete', 'search', 'filter', 'sort', 'view'
        }
        
        # Clarity indicators
        self.clarity_markers = {
            'specific': ['must', 'shall', 'should', 'will', 'exactly', 'specifically'],
            'vague': ['maybe', 'perhaps', 'might', 'could', 'possibly', 'sometime'],
            'sequential': ['first', 'then', 'next', 'after', 'finally', 'before'],
            'conditional': ['if', 'when', 'unless', 'otherwise', 'else']
        }
    
    def calculate_all_metrics(self, texts: List[str], text_type: str = "use_case") -> Dict[str, Any]:
        """Calculate all UX metrics"""
        metrics = {
            'readability': self._calculate_readability(texts),
            'clarity': self._calculate_clarity(texts),
            'actionability': self._calculate_actionability(texts, text_type),
            'completeness': self._calculate_completeness(texts, text_type),
            'usability': self._calculate_usability(texts, text_type)
        }
        
        # Overall UX score
        metrics['overall_ux_score'] = self._calculate_overall_ux_score(metrics)
        
        return metrics
    
    def _calculate_readability(self, texts: List[str]) -> Dict[str, float]:
        """Calculate readability metrics"""
        scores = {
            'flesch_reading_ease': [],
            'flesch_kincaid_grade': [],
            'gunning_fog': [],
            'smog_index': [],
            'automated_readability_index': [],
            'coleman_liau_index': [],
            'linsear_write_formula': [],
            'dale_chall_readability_score': []
        }
        
        sentence_lengths = []
        word_lengths = []
        
        for text in texts:
            try:
                # Standard readability metrics
                scores['flesch_reading_ease'].append(textstat.flesch_reading_ease(text))
                scores['flesch_kincaid_grade'].append(textstat.flesch_kincaid_grade(text))
                scores['gunning_fog'].append(textstat.gunning_fog(text))
                scores['smog_index'].append(textstat.smog_index(text))
                scores['automated_readability_index'].append(textstat.automated_readability_index(text))
                scores['coleman_liau_index'].append(textstat.coleman_liau_index(text))
                scores['linsear_write_formula'].append(textstat.linsear_write_formula(text))
                scores['dale_chall_readability_score'].append(textstat.dale_chall_readability_score(text))
                
                # Sentence and word analysis
                sentences = nltk.sent_tokenize(text)
                for sentence in sentences:
                    words = nltk.word_tokenize(sentence)
                    sentence_lengths.append(len(words))
                    word_lengths.extend([len(w) for w in words if w.isalpha()])
                    
            except Exception as e:
                logger.warning(f"Error calculating readability: {e}")
        
        # Calculate averages
        result = {}
        for metric, values in scores.items():
            if values:
                result[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # Add sentence and word metrics
        if sentence_lengths:
            result['avg_sentence_length'] = np.mean(sentence_lengths)
            result['sentence_length_variance'] = np.var(sentence_lengths)
        
        if word_lengths:
            result['avg_word_length'] = np.mean(word_lengths)
            result['complex_words_ratio'] = len([w for w in word_lengths if w > 6]) / len(word_lengths)
        
        # Overall readability score (normalized 0-100)
        if 'flesch_reading_ease' in result:
            result['overall_readability'] = result['flesch_reading_ease']['mean']
        
        return result
    
    def _calculate_clarity(self, texts: List[str]) -> Dict[str, float]:
        """Calculate clarity metrics"""
        clarity_scores = []
        specificity_scores = []
        
        for text in texts:
            text_lower = text.lower()
            
            # Count clarity markers
            specific_count = sum(text_lower.count(marker) for marker in self.clarity_markers['specific'])
            vague_count = sum(text_lower.count(marker) for marker in self.clarity_markers['vague'])
            sequential_count = sum(text_lower.count(marker) for marker in self.clarity_markers['sequential'])
            conditional_count = sum(text_lower.count(marker) for marker in self.clarity_markers['conditional'])
            
            # Calculate clarity score
            total_markers = specific_count + vague_count + sequential_count + conditional_count
            if total_markers > 0:
                clarity_score = (specific_count + sequential_count - vague_count) / total_markers
                clarity_scores.append(max(0, min(1, clarity_score)))
            
            # Specificity score (ratio of specific terms)
            words = nltk.word_tokenize(text_lower)
            if words:
                specificity = specific_count / len(words)
                specificity_scores.append(specificity)
            
        return {
            'clarity_score': np.mean(clarity_scores) if clarity_scores else 0,
            'specificity_score': np.mean(specificity_scores) if specificity_scores else 0,
            'avg_specific_markers': np.mean([sum(t.lower().count(m) for m in self.clarity_markers['specific']) for t in texts]),
            'avg_vague_markers': np.mean([sum(t.lower().count(m) for m in self.clarity_markers['vague']) for t in texts]),
            'sequential_flow_score': np.mean([sum(t.lower().count(m) for m in self.clarity_markers['sequential']) for t in texts])
        }
    
    def _calculate_actionability(self, texts: List[str], text_type: str) -> Dict[str, float]:
        """Calculate how actionable the content is"""
        actionability_scores = []
        action_densities = []
        
        for text in texts:
            # Tokenize and POS tag
            words = nltk.word_tokenize(text.lower())
            pos_tags = nltk.pos_tag(words)
            
            # Count verbs (action words)
            verbs = [word for word, pos in pos_tags if pos.startswith('VB')]
            action_verbs_found = [v for v in verbs if v in self.action_verbs]
            
            # Calculate metrics
            if words:
                action_density = len(action_verbs_found) / len(words)
                action_densities.append(action_density)
            
            # For use cases, check for numbered steps
            if text_type == "use_case":
                numbered_steps = len(re.findall(r'\d+\.', text))
                has_clear_flow = numbered_steps > 3
                actionability_scores.append(1.0 if has_clear_flow else 0.5)
            
            # For test cases, check for keywords
            elif text_type == "test_case":
                rf_keywords = len(re.findall(r'(Click|Type Text|Go To|Wait For|Should)', text))
                actionability_scores.append(min(1.0, rf_keywords / 10))
        
        return {
            'actionability_score': np.mean(actionability_scores) if actionability_scores else 0,
            'action_density': np.mean(action_densities) if action_densities else 0,
            'avg_actions_per_text': np.mean([len([v for v in nltk.word_tokenize(t.lower()) if v in self.action_verbs]) for t in texts]),
            'texts_with_clear_actions': sum(1 for t in texts if any(av in t.lower() for av in self.action_verbs)) / len(texts) if texts else 0
        }
    
    def _calculate_completeness(self, texts: List[str], text_type: str) -> Dict[str, float]:
        """Calculate completeness of content"""
        if text_type == "use_case":
            return self._calculate_use_case_completeness(texts)
        elif text_type == "test_case":
            return self._calculate_test_case_completeness(texts)
        else:
            return {'completeness_score': 0.5}
    
    def _calculate_use_case_completeness(self, texts: List[str]) -> Dict[str, float]:
        """Calculate use case completeness"""
        required_sections = {
            'actors': ['actor', 'user', 'system'],
            'preconditions': ['precondition', 'prerequisite', 'require'],
            'main_flow': ['main flow', 'steps', 'flow'],
            'postconditions': ['postcondition', 'result', 'outcome'],
            'alternative_flow': ['alternative', 'exception', 'error']
        }
        
        completeness_scores = []
        section_coverage = {section: 0 for section in required_sections}
        
        for text in texts:
            text_lower = text.lower()
            score = 0
            
            for section, keywords in required_sections.items():
                if any(keyword in text_lower for keyword in keywords):
                    score += 1
                    section_coverage[section] += 1
            
            # Bonus for having numbered steps
            if re.findall(r'\d+\.', text):
                score += 0.5
            
            completeness_scores.append(score / (len(required_sections) + 0.5))
        
        # Calculate coverage percentages
        if texts:
            for section in section_coverage:
                section_coverage[section] = section_coverage[section] / len(texts)
        
        return {
            'completeness_score': np.mean(completeness_scores) if completeness_scores else 0,
            'section_coverage': section_coverage,
            'avg_sections_present': np.mean([sum(1 for k in required_sections if any(kw in t.lower() for kw in required_sections[k])) for t in texts])
        }
    
    def _calculate_test_case_completeness(self, texts: List[str]) -> Dict[str, float]:
        """Calculate test case completeness"""
        required_elements = {
            'setup': ['setup', 'new browser', 'new page'],
            'actions': ['click', 'type', 'go to', 'select'],
            'verification': ['should', 'wait for', 'get text', 'element state'],
            'teardown': ['teardown', 'close browser', 'close']
        }
        
        completeness_scores = []
        element_coverage = {element: 0 for element in required_elements}
        
        for text in texts:
            text_lower = text.lower()
            score = 0
            
            for element, keywords in required_elements.items():
                if any(keyword in text_lower for keyword in keywords):
                    score += 1
                    element_coverage[element] += 1
            
            completeness_scores.append(score / len(required_elements))
        
        # Calculate coverage percentages
        if texts:
            for element in element_coverage:
                element_coverage[element] = element_coverage[element] / len(texts)
        
        return {
            'completeness_score': np.mean(completeness_scores) if completeness_scores else 0,
            'element_coverage': element_coverage,
            'avg_elements_present': np.mean([sum(1 for e in required_elements if any(kw in t.lower() for kw in required_elements[e])) for t in texts])
        }
    
    def _calculate_usability(self, texts: List[str], text_type: str) -> Dict[str, float]:
        """Calculate overall usability metrics"""
        usability_factors = {
            'length_appropriate': [],
            'structure_clear': [],
            'language_consistent': [],
            'examples_present': []
        }
        
        for text in texts:
            # Length appropriateness
            word_count = len(nltk.word_tokenize(text))
            if text_type == "use_case":
                # Use cases should be 100-500 words
                length_score = 1.0 if 100 <= word_count <= 500 else 0.5
            else:
                # Test cases can be longer
                length_score = 1.0 if 50 <= word_count <= 1000 else 0.5
            usability_factors['length_appropriate'].append(length_score)
            
            # Structure clarity (presence of sections/formatting)
            has_sections = bool(re.findall(r'[A-Z]{2,}:|^\d+\.|^-|^\*', text, re.MULTILINE))
            usability_factors['structure_clear'].append(1.0 if has_sections else 0.5)
            
            # Language consistency (vocabulary repetition for key terms)
            words = nltk.word_tokenize(text.lower())
            word_freq = Counter(words)
            # Check if key terms are used consistently
            consistency_score = 1.0 if any(freq > 2 for word, freq in word_freq.items() if len(word) > 4) else 0.5
            usability_factors['language_consistent'].append(consistency_score)
            
            # Examples present
            has_examples = bool(re.findall(r'(example|e\.g\.|for instance|such as)', text, re.IGNORECASE))
            usability_factors['examples_present'].append(1.0 if has_examples else 0.0)
        
        # Calculate overall usability
        usability_score = np.mean([
            np.mean(scores) for scores in usability_factors.values() if scores
        ])
        
        return {
            'usability_score': usability_score,
            'length_appropriateness': np.mean(usability_factors['length_appropriate']) if usability_factors['length_appropriate'] else 0,
            'structure_clarity': np.mean(usability_factors['structure_clear']) if usability_factors['structure_clear'] else 0,
            'language_consistency': np.mean(usability_factors['language_consistent']) if usability_factors['language_consistent'] else 0,
            'examples_present': np.mean(usability_factors['examples_present']) if usability_factors['examples_present'] else 0
        }
    
    def _calculate_overall_ux_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall UX score from all metrics"""
        scores = []
        
        # Weight different aspects
        weights = {
            'readability': 0.25,
            'clarity': 0.25,
            'actionability': 0.20,
            'completeness': 0.20,
            'usability': 0.10
        }
        
        for aspect, weight in weights.items():
            if aspect in metrics:
                if aspect == 'readability' and 'overall_readability' in metrics[aspect]:
                    # Normalize readability (0-100 to 0-1)
                    score = min(1.0, metrics[aspect]['overall_readability'] / 100)
                elif f'{aspect}_score' in metrics[aspect]:
                    score = metrics[aspect][f'{aspect}_score']
                else:
                    continue
                
                scores.append(score * weight)
        
        return sum(scores) / sum(weights.values()) if scores else 0