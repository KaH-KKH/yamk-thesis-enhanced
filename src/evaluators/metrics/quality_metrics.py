# src/evaluators/metrics/quality_metrics.py
"""
Quality metrics for generated content evaluation
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import Counter
import nltk
from nltk.util import ngrams
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger
import textstat
import warnings

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True) 
    nltk.download('punkt_tab', quiet=True)
except:
    warnings.warn("NLTK resources could not be downloaded")

class QualityMetrics:
    """Calculate quality metrics for generated texts"""
    
    def __init__(self):
        self.sentence_model = None
        self.perplexity_model = None
        self.perplexity_tokenizer = None
        
    def calculate_all_metrics(self, texts: List[str], references: List[str] = None) -> Dict[str, Any]:
        """Calculate all quality metrics"""
        metrics = {}
        
        # Basic metrics (existing ones are handled in main evaluation)
        
        # Perplexity
        logger.info("Calculating perplexity...")
        metrics["perplexity"] = self.calculate_perplexity(texts)
        
        # Diversity metrics
        logger.info("Calculating diversity metrics...")
        metrics["diversity"] = self.calculate_diversity_metrics(texts)
        
        # Coherence
        logger.info("Calculating coherence...")
        metrics["coherence"] = self.calculate_coherence(texts)
        
        # Semantic similarity (if references provided)
        if references:
            logger.info("Calculating semantic similarity...")
            metrics["semantic_similarity"] = self.calculate_semantic_similarity(texts, references)
        
        return metrics
    
    def calculate_perplexity(self, texts: List[str]) -> Dict[str, float]:
        """Calculate perplexity using GPT-2"""
        if self.perplexity_model is None:
            logger.info("Loading GPT-2 for perplexity calculation...")
            self.perplexity_model = GPT2LMHeadModel.from_pretrained('gpt2')
            self.perplexity_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
            self.perplexity_model.eval()
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.perplexity_model = self.perplexity_model.cuda()
        
        perplexities = []
        
        for text in texts:
            try:
                # Tokenize
                encodings = self.perplexity_tokenizer(
                    text, 
                    return_tensors='pt',
                    max_length=512,
                    truncation=True
                )
                
                # Move to same device as model
                if torch.cuda.is_available():
                    encodings = {k: v.cuda() for k, v in encodings.items()}
                
                # Calculate loss
                with torch.no_grad():
                    outputs = self.perplexity_model(**encodings, labels=encodings['input_ids'])
                    perplexity = torch.exp(outputs.loss).item()
                
                perplexities.append(perplexity)
                
            except Exception as e:
                logger.warning(f"Error calculating perplexity: {e}")
                perplexities.append(float('inf'))
        
        return {
            "mean_perplexity": np.mean([p for p in perplexities if p != float('inf')]),
            "min_perplexity": np.min([p for p in perplexities if p != float('inf')]),
            "max_perplexity": np.max([p for p in perplexities if p != float('inf')]),
            "std_perplexity": np.std([p for p in perplexities if p != float('inf')])
        }
    
    def calculate_diversity_metrics(self, texts: List[str]) -> Dict[str, float]:
        """Calculate diversity metrics including Self-BLEU and Distinct-n"""
        metrics = {}
        
        # Self-BLEU: measures diversity between generated texts
        logger.info("Calculating Self-BLEU...")
        self_bleu_scores = []
        
        for i, text in enumerate(texts):
            # Compare each text with all others
            other_texts = texts[:i] + texts[i+1:]
            if other_texts:
                # Calculate BLEU score against other texts
                from sacrebleu import sentence_bleu
                bleu_scores = []
                for other in other_texts:
                    score = sentence_bleu(text, [other]).score
                    bleu_scores.append(score)
                self_bleu_scores.append(np.mean(bleu_scores))
        
        metrics["self_bleu"] = np.mean(self_bleu_scores) if self_bleu_scores else 0.0
        
        # Distinct-n: ratio of unique n-grams
        logger.info("Calculating Distinct-n...")
        for n in [1, 2, 3]:
            all_ngrams = []
            for text in texts:
                tokens = nltk.word_tokenize(text.lower())
                all_ngrams.extend(list(ngrams(tokens, n)))
            
            if all_ngrams:
                unique_ngrams = len(set(all_ngrams))
                total_ngrams = len(all_ngrams)
                metrics[f"distinct_{n}"] = unique_ngrams / total_ngrams
            else:
                metrics[f"distinct_{n}"] = 0.0
        
        # Vocabulary diversity
        all_tokens = []
        for text in texts:
            tokens = nltk.word_tokenize(text.lower())
            all_tokens.extend(tokens)
        
        metrics["vocabulary_size"] = len(set(all_tokens))
        metrics["token_type_ratio"] = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0
        
        return metrics
    
    def calculate_coherence(self, texts: List[str]) -> Dict[str, float]:
        """Calculate coherence metrics"""
        if self.sentence_model is None:
            logger.info("Loading sentence transformer for coherence...")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        coherence_scores = []
        
        for text in texts:
            # Split into sentences
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) > 1:
                # Encode sentences
                embeddings = self.sentence_model.encode(sentences)
                
                # Calculate coherence as average similarity between consecutive sentences
                similarities = []
                for i in range(len(embeddings) - 1):
                    sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
                    similarities.append(sim)
                
                coherence_scores.append(np.mean(similarities))
            else:
                coherence_scores.append(1.0)  # Single sentence is perfectly coherent
        
        return {
            "mean_coherence": np.mean(coherence_scores),
            "min_coherence": np.min(coherence_scores),
            "max_coherence": np.max(coherence_scores),
            "std_coherence": np.std(coherence_scores)
        }
    
    def calculate_semantic_similarity(self, candidates: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate semantic similarity between candidates and references"""
        if self.sentence_model is None:
            logger.info("Loading sentence transformer...")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Encode all texts
        candidate_embeddings = self.sentence_model.encode(candidates)
        reference_embeddings = self.sentence_model.encode(references)
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(candidate_embeddings, reference_embeddings)
        
        # For each candidate, find most similar reference
        max_similarities = np.max(similarities, axis=1)
        
        return {
            "mean_similarity": np.mean(max_similarities),
            "min_similarity": np.min(max_similarities),
            "max_similarity": np.max(max_similarities),
            "std_similarity": np.std(max_similarities)
        }
    
    def calculate_readability_metrics(self, texts: List[str]) -> Dict[str, float]:
        """Calculate various readability metrics"""
        metrics = {}
        
        scores = {
            "flesch_reading_ease": [],
            "flesch_kincaid_grade": [],
            "gunning_fog": [],
            "coleman_liau_index": [],
            "automated_readability_index": [],
            "dale_chall_readability_score": []
        }
        
        for text in texts:
            try:
                scores["flesch_reading_ease"].append(textstat.flesch_reading_ease(text))
                scores["flesch_kincaid_grade"].append(textstat.flesch_kincaid_grade(text))
                scores["gunning_fog"].append(textstat.gunning_fog(text))
                scores["coleman_liau_index"].append(textstat.coleman_liau_index(text))
                scores["automated_readability_index"].append(textstat.automated_readability_index(text))
                scores["dale_chall_readability_score"].append(textstat.dale_chall_readability_score(text))
            except Exception as e:
                logger.warning(f"Error calculating readability: {e}")
        
        # Calculate averages
        for metric_name, values in scores.items():
            if values:
                metrics[f"avg_{metric_name}"] = np.mean(values)
        
        return metrics