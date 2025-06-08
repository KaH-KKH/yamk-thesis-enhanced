# src/evaluators/metrics/ux_metrics.py
"""
User Experience metrics for evaluating generated content
"""

import re
import textstat
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import Counter
import nltk
from loguru import logger

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass


class UserExperienceMetrics:
    """Calculate user experience related metrics"""
    
    def __init__(self):
        self.action_verbs = {
            'click', 'enter', 'select', 'submit', 'navigate', 'login', 'logout',
            'verify', 'check', 'validate', 'create', 'update', 'delete', 'upload',
            'download', 'search', 'filter', 'sort', 'view', 'display', 'open'
        }
        
        self.clarity_indicators = {
            'specific': ['must', 'shall', 'should', 'will'],
            'vague': ['might', 'could', 'possibly', 'maybe', 'perhaps'],
            'structured': ['first', 'then', 'next', 'finally', 'after'],
            'conditional': ['if', 'when', 'unless', 'until']
        }
    
    def calculate_all_metrics(self, texts: List[str], text_type: str = "use_case") -> Dict[str, Any]:
        """Calculate all UX metrics for given texts"""
        
        metrics = {
            "readability": self._calculate_readability(texts),
            "clarity": self._calculate_clarity(texts),
            "actionability": self._calculate_actionability(texts, text_type),
            "completeness": self._calculate_completeness(texts, text_type),
            "usability": self._calculate_usability(texts, text_type),
            "accessibility": self._calculate_accessibility(texts)
        }
        
        return metrics
    
    def _calculate_readability(self, texts: List[str]) -> Dict[str, Any]:
        """Calculate various readability metrics"""
        readability_scores = {
            "flesch_reading_ease": [],
            "flesch_kincaid_grade": [],
            "gunning_fog": [],
            "smog_index": [],
            "automated_readability_index": [],
            "coleman_liau_index": [],
            "linsear_write_formula": [],
            "dale_chall_readability_score": []
        }
        
        sentence_metrics = {
            "avg_sentence_length": [],
            "avg_word_length": [],
            "syllables_per_word": []
        }
        
        for text in texts:
            if not text.strip():
                continue
                
            try:
                # Readability scores
                readability_scores["flesch_reading_ease"].append(textstat.flesch_reading_ease(text))
                readability_scores["flesch_kincaid_grade"].append(textstat.flesch_kincaid_grade(text))
                readability_scores["gunning_fog"].append(textstat.gunning_fog(text))
                readability_scores["smog_index"].append(textstat.smog_index(text))
                readability_scores["automated_readability_index"].append(textstat.automated_readability_index(text))
                readability_scores["coleman_liau_index"].append(textstat.coleman_liau_index(text))
                readability_scores["linsear_write_formula"].append(textstat.linsear_write_formula(text))
                readability_scores["dale_chall_readability_score"].append(textstat.dale_chall_readability_score(text))
                
                # Sentence metrics
                sentences = nltk.sent_tokenize(text)
                words = nltk.word_tokenize(text)
                
                if sentences and words:
                    sentence_metrics["avg_sentence_length"].append(len(words) / len(sentences))
                    sentence_metrics["avg_word_length"].append(np.mean([len(w) for w in words]))
                    sentence_metrics["syllables_per_word"].append(textstat.syllable_count(text) / len(words))
                    
            except Exception as e:
                logger.warning(f"Error calculating readability: {e}")
        
        # Calculate averages and interpretations
        results = {}
        
        for metric_name, scores in readability_scores.items():
            if scores:
                avg_score = np.mean(scores)
                results[metric_name] = {
                    "score": avg_score,
                    "interpretation": self._interpret_readability_score(metric_name, avg_score)
                }
        
        for metric_name, values in sentence_metrics.items():
            if values:
                results[metric_name] = np.mean(values)
        
        # Overall readability assessment
        if readability_scores["flesch_reading_ease"]:
            avg_flesch = np.mean(readability_scores["flesch_reading_ease"])
            results["overall_readability"] = self._get_overall_readability(avg_flesch)
        
        return results
    
    def _interpret_readability_score(self, metric: str, score: float) -> str:
        """Interpret readability scores"""
        if metric == "flesch_reading_ease":
            if score >= 90:
                return "Very Easy"
            elif score >= 80:
                return "Easy"
            elif score >= 70:
                return "Fairly Easy"
            elif score >= 60:
                return "Standard"
            elif score >= 50:
                return "Fairly Difficult"
            elif score >= 30:
                return "Difficult"
            else:
                return "Very Difficult"
                
        elif metric == "flesch_kincaid_grade":
            return f"Grade level: {score:.1f}"
            
        elif metric == "gunning_fog":
            return f"Years of education needed: {score:.1f}"
            
        return f"Score: {score:.2f}"
    
    def _get_overall_readability(self, flesch_score: float) -> str:
        """Get overall readability assessment"""
        if flesch_score >= 70:
            return "Good - Easy to understand"
        elif flesch_score >= 50:
            return "Moderate - Requires some effort"
        else:
            return "Poor - Difficult to understand"
    
    def _calculate_clarity(self, texts: List[str]) -> Dict[str, Any]:
        """Calculate clarity metrics"""
        clarity_metrics = {
            "specificity_score": [],
            "vagueness_score": [],
            "structure_score": [],
            "conditional_clarity": [],
            "ambiguity_score": []
        }
        
        for text in texts:
            words = nltk.word_tokenize(text.lower())
            
            # Count clarity indicators
            specific_count = sum(1 for word in words if word in self.clarity_indicators['specific'])
            vague_count = sum(1 for word in words if word in self.clarity_indicators['vague'])
            structured_count = sum(1 for word in words if word in self.clarity_indicators['structured'])
            conditional_count = sum(1 for word in words if word in self.clarity_indicators['conditional'])
            
            total_words = len(words)
            
            if total_words > 0:
                clarity_metrics["specificity_score"].append(specific_count / total_words)
                clarity_metrics["vagueness_score"].append(vague_count / total_words)
                clarity_metrics["structure_score"].append(structured_count / total_words)
                clarity_metrics["conditional_clarity"].append(conditional_count / total_words)
                
                # Ambiguity score (based on passive voice and complex sentences)
                ambiguity = self._calculate_ambiguity(text)
                clarity_metrics["ambiguity_score"].append(ambiguity)
        
        # Calculate final scores
        results = {}
        for metric, scores in clarity_metrics.items():
            if scores:
                results[metric] = np.mean(scores)
        
        # Overall clarity score
        if all(metric in results for metric in ["specificity_score", "vagueness_score", "structure_score"]):
            results["overall_clarity"] = (
                results["specificity_score"] * 0.4 +
                results["structure_score"] * 0.4 -
                results["vagueness_score"] * 0.2
            )
        
        return results
    
    def _calculate_ambiguity(self, text: str) -> float:
        """Calculate ambiguity score based on various factors"""
        sentences = nltk.sent_tokenize(text)
        ambiguity_factors = []
        
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            pos_tags = nltk.pos_tag(words)
            
            # Check for passive voice (simplified)
            passive_indicators = ['been', 'was', 'were', 'being', 'is', 'are', 'be']
            has_passive = any(word in passive_indicators for word, _ in pos_tags)
            
            # Check for complex sentence structure
            complexity = len(words) / 20  # Normalized by typical sentence length
            
            # Check for pronouns without clear antecedents
            pronouns = sum(1 for _, tag in pos_tags if tag in ['PRP', 'PRP$', 'WP', 'WP$'])
            pronoun_ratio = pronouns / len(words) if words else 0
            
            ambiguity_factors.append(
                (0.3 * has_passive) + (0.4 * min(complexity, 1)) + (0.3 * pronoun_ratio)
            )
        
        return np.mean(ambiguity_factors) if ambiguity_factors else 0
    
    def _calculate_actionability(self, texts: List[str], text_type: str) -> Dict[str, Any]:
        """Calculate how actionable the content is"""
        actionability_metrics = {
            "action_verb_density": [],
            "step_clarity": [],
            "instruction_quality": [],
            "executable_steps": []
        }
        
        for text in texts:
            # Tokenize and tag
            words = nltk.word_tokenize(text.lower())
            pos_tags = nltk.pos_tag(words)
            
            # Count action verbs
            action_count = sum(1 for word in words if word in self.action_verbs)
            verb_count = sum(1 for _, tag in pos_tags if tag.startswith('VB'))
            
            if len(words) > 0:
                actionability_metrics["action_verb_density"].append(action_count / len(words))
            
            # Analyze steps
            if text_type == "use_case":
                step_quality = self._analyze_use_case_steps(text)
                actionability_metrics["step_clarity"].append(step_quality["clarity"])
                actionability_metrics["executable_steps"].append(step_quality["executable_ratio"])
            elif text_type == "test_case":
                instruction_quality = self._analyze_test_instructions(text)
                actionability_metrics["instruction_quality"].append(instruction_quality)
        
        # Calculate results
        results = {}
        for metric, scores in actionability_metrics.items():
            if scores:
                results[metric] = np.mean(scores)
        
        # Overall actionability score
        if results:
            results["overall_actionability"] = np.mean(list(results.values()))
        
        return results
    
    def _analyze_use_case_steps(self, text: str) -> Dict[str, float]:
        """Analyze quality of use case steps"""
        # Extract steps (numbered items)
        step_pattern = r'\d+\.\s*(.+?)(?=\d+\.|$)'
        steps = re.findall(step_pattern, text, re.DOTALL)
        
        if not steps:
            return {"clarity": 0, "executable_ratio": 0}
        
        clear_steps = 0
        executable_steps = 0
        
        for step in steps:
            # Check clarity (has actor and action)
            has_actor = any(actor in step.lower() for actor in ['user', 'system', 'admin'])
            has_action = any(verb in step.lower() for verb in self.action_verbs)
            
            if has_actor and has_action:
                clear_steps += 1
            
            # Check executability (specific and measurable)
            if has_action and len(step.split()) > 3:  # More than just "User clicks button"
                executable_steps += 1
        
        return {
            "clarity": clear_steps / len(steps),
            "executable_ratio": executable_steps / len(steps)
        }
    
    def _analyze_test_instructions(self, text: str) -> float:
        """Analyze quality of test instructions"""
        # For Robot Framework tests, check for proper keywords and arguments
        quality_indicators = [
            r'Click\s+.+',  # Has selector
            r'Type Text\s+.+\s+.+',  # Has selector and text
            r'Wait For.*State\s+.+',  # Has wait condition
            r'Should\s+.+',  # Has assertion
        ]
        
        matches = sum(1 for pattern in quality_indicators if re.search(pattern, text))
        return matches / len(quality_indicators)
    
    def _calculate_completeness(self, texts: List[str], text_type: str) -> Dict[str, Any]:
        """Calculate completeness of content"""
        completeness_scores = []
        
        for text in texts:
            if text_type == "use_case":
                score = self._check_use_case_completeness(text)
            elif text_type == "test_case":
                score = self._check_test_case_completeness(text)
            else:
                score = 0
            
            completeness_scores.append(score)
        
        return {
            "completeness_score": np.mean(completeness_scores) if completeness_scores else 0,
            "missing_elements": self._identify_missing_elements(texts, text_type)
        }
    
    def _check_use_case_completeness(self, text: str) -> float:
        """Check completeness of use case"""
        required_elements = {
            'actors': ['actor', 'user', 'system'],
            'preconditions': ['precondition', 'prerequisite'],
            'main_flow': ['main flow', 'steps', 'flow'],
            'postconditions': ['postcondition', 'result'],
            'id': ['uc-', 'id:', 'identifier']
        }
        
        found_elements = 0
        for element, keywords in required_elements.items():
            if any(keyword in text.lower() for keyword in keywords):
                found_elements += 1
        
        return found_elements / len(required_elements)
    
    def _check_test_case_completeness(self, text: str) -> float:
        """Check completeness of test case"""
        required_patterns = [
            r'\*\*\* Settings \*\*\*',
            r'\*\*\* Test Cases \*\*\*',
            r'Documentation',
            r'Setup|Teardown',
            r'Should|Verify|Check'  # Assertions
        ]
        
        found_patterns = sum(1 for pattern in required_patterns if re.search(pattern, text))
        return found_patterns / len(required_patterns)
    
    def _identify_missing_elements(self, texts: List[str], text_type: str) -> List[str]:
        """Identify commonly missing elements"""
        missing_elements = []
        
        if text_type == "use_case":
            # Check what's commonly missing across all texts
            elements = ['actors', 'preconditions', 'postconditions', 'alternative flows']
            for element in elements:
                if sum(1 for text in texts if element in text.lower()) < len(texts) * 0.5:
                    missing_elements.append(element)
                    
        elif text_type == "test_case":
            elements = ['documentation', 'tags', 'teardown', 'error handling']
            for element in elements:
                if sum(1 for text in texts if element in text.lower()) < len(texts) * 0.5:
                    missing_elements.append(element)
        
        return missing_elements
    
    def _calculate_usability(self, texts: List[str], text_type: str) -> Dict[str, Any]:
        """Calculate usability metrics"""
        return {
            "navigation_ease": self._calculate_navigation_ease(texts),
            "information_findability": self._calculate_findability(texts),
            "consistency_score": self._calculate_consistency(texts),
            "user_friendliness": self._calculate_user_friendliness(texts, text_type)
        }
    
    def _calculate_navigation_ease(self, texts: List[str]) -> float:
        """How easy is it to navigate through the content"""
        navigation_scores = []
        
        for text in texts:
            # Check for clear structure markers
            has_headers = bool(re.findall(r'^[A-Z\s]+:$', text, re.MULTILINE))
            has_numbering = bool(re.findall(r'^\d+\.', text, re.MULTILINE))
            has_sections = text.count('\n\n') > 2  # Multiple sections
            
            score = (has_headers + has_numbering + has_sections) / 3
            navigation_scores.append(score)
        
        return np.mean(navigation_scores) if navigation_scores else 0
    
    def _calculate_findability(self, texts: List[str]) -> float:
        """How easy is it to find specific information"""
        findability_scores = []
        
        for text in texts:
            # Check for searchable elements
            has_ids = bool(re.findall(r'ID:|UC-|TC-', text))
            has_keywords = len(set(nltk.word_tokenize(text.lower()))) / len(nltk.word_tokenize(text))
            has_clear_labels = bool(re.findall(r'[A-Z][a-z]+:', text))
            
            score = (has_ids + has_keywords + has_clear_labels) / 3
            findability_scores.append(score)
        
        return np.mean(findability_scores) if findability_scores else 0
    
    def _calculate_consistency(self, texts: List[str]) -> float:
        """Check consistency across texts"""
        if len(texts) < 2:
            return 1.0
        
        # Check structural consistency
        structures = []
        for text in texts:
            structure = []
            if 'actors' in text.lower():
                structure.append('actors')
            if 'preconditions' in text.lower():
                structure.append('preconditions')
            if 'main flow' in text.lower():
                structure.append('main_flow')
            structures.append(tuple(structure))
        
        # Calculate how many texts have the same structure
        most_common = Counter(structures).most_common(1)[0][1]
        return most_common / len(texts)
    
    def _calculate_user_friendliness(self, texts: List[str], text_type: str) -> float:
        """Overall user friendliness score"""
        scores = []
        
        for text in texts:
            # Factors that contribute to user friendliness
            factors = []
            
            # Clear language (low complexity)
            complexity = textstat.flesch_kincaid_grade(text)
            factors.append(1 - min(complexity / 12, 1))  # Normalize to 0-1
            
            # Good formatting
            has_formatting = bool(re.findall(r'[-•*]|\d+\.', text))
            factors.append(float(has_formatting))
            
            # Appropriate length
            word_count = len(nltk.word_tokenize(text))
            if text_type == "use_case":
                ideal_length = 200  # words
            else:
                ideal_length = 150
            
            length_score = 1 - abs(word_count - ideal_length) / ideal_length
            factors.append(max(0, length_score))
            
            scores.append(np.mean(factors))
        
        return np.mean(scores) if scores else 0
    
    def _calculate_accessibility(self, texts: List[str]) -> Dict[str, Any]:
        """Calculate accessibility metrics"""
        return {
            "plain_language_score": self._calculate_plain_language(texts),
            "technical_jargon_ratio": self._calculate_jargon_ratio(texts),
            "international_friendly": self._calculate_international_friendliness(texts)
        }
    
    def _calculate_plain_language(self, texts: List[str]) -> float:
        """Calculate how much plain language is used"""
        scores = []
        
        for text in texts:
            # Simple words (common 1000 words would be ideal)
            words = nltk.word_tokenize(text.lower())
            simple_words = sum(1 for w in words if len(w) <= 7)  # Simplified metric
            
            if words:
                scores.append(simple_words / len(words))
        
        return np.mean(scores) if scores else 0
    
    def _calculate_jargon_ratio(self, texts: List[str]) -> float:
        """Calculate ratio of technical jargon"""
        # Common technical terms in software
        jargon = {
            'api', 'backend', 'frontend', 'database', 'query', 'endpoint',
            'authentication', 'authorization', 'token', 'session', 'cache',
            'framework', 'library', 'dependency', 'deployment', 'integration'
        }
        
        ratios = []
        for text in texts:
            words = nltk.word_tokenize(text.lower())
            jargon_count = sum(1 for word in words if word in jargon)
            
            if words:
                ratios.append(jargon_count / len(words))
        
        return np.mean(ratios) if ratios else 0
    
    def _calculate_international_friendliness(self, texts: List[str]) -> float:
        """Check if content is friendly for international users"""
        scores = []
        
        for text in texts:
            factors = []
            
            # Avoid idioms and cultural references
            idiom_patterns = ['piece of cake', 'hit the ground running', 'ball is in']
            has_idioms = any(idiom in text.lower() for idiom in idiom_patterns)
            factors.append(1 - float(has_idioms))
            
            # Use of dates/times in standard format
            has_standard_dates = bool(re.findall(r'\d{4}-\d{2}-\d{2}', text))
            factors.append(float(has_standard_dates) if re.search(r'\d+/\d+/\d+', text) else 1)
            
            # Avoid complex contractions
            contractions = len(re.findall(r"\w+'d|\w+'ll|\w+'ve", text))
            factors.append(1 - min(contractions / 10, 1))
            
            scores.append(np.mean(factors))
        
        return np.mean(scores) if scores else 0