
import numpy as np
from typing import Dict, List, Optional
from transformers import pipeline


class SentimentMetrics:
    """
    A class for computing sentiment-based metrics.
    
    Calculates:
    - Sentiment variance across sentences
    - Mean sentiment
    - Sentiment range
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize sentiment analyzer.
        
        Args:
            model_name: HuggingFace model for sentiment analysis
        """
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=-1  # Use CPU
        )
        self.sentences = []
        self.sentiment_scores = []
        
    def process_text(self, text: str, sentences: Optional[List[str]] = None):
        """
        Process text and split into sentences if not provided.
        
        Args:
            text: Input text
            sentences: Pre-split sentences (optional)
        """
        if sentences:
            self.sentences = sentences
        else:
            # Simple sentence splitting
            import re
            self.sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    
    def compute_sentiment_scores(self) -> np.ndarray:
        """
        Compute sentiment scores for all sentences.
        Maps POSITIVE/NEGATIVE to continuous scale [-1, 1].
        
        Returns:
            Array of sentiment scores
        """
        if not self.sentences:
            raise ValueError("No sentences to analyze. Call process_text() first.")
        
        scores = []
        for sentence in self.sentences:
            # Handle long sentences by truncation
            if len(sentence) > 512:
                sentence = sentence[:512]
            
            result = self.sentiment_analyzer(sentence)[0]
            label = result['label']
            confidence = result['score']
            
            # Map to [-1, 1] scale
            if label == 'POSITIVE':
                score = confidence
            else:  # NEGATIVE
                score = -confidence
            
            scores.append(score)
        
        self.sentiment_scores = np.array(scores)
        return self.sentiment_scores
    
    def variance_of_sentiment(self) -> Dict[str, float]:
        """
        Calculate variance of sentiment across sentences.
        Formula: σ²_s = (1/N) Σ (s_i - μ_s)²
        
        Returns:
            Dictionary with sentiment statistics
        """
        if len(self.sentiment_scores) == 0:
            self.compute_sentiment_scores()
        
        if len(self.sentiment_scores) == 0:
            return {
                'sentiment_variance': 0.0,
                'sentiment_mean': 0.0,
                'sentiment_std': 0.0,
                'sentiment_range': 0.0
            }
        
        mean_sentiment = np.mean(self.sentiment_scores)
        variance = np.var(self.sentiment_scores)
        std = np.std(self.sentiment_scores)
        sentiment_range = np.max(self.sentiment_scores) - np.min(self.sentiment_scores)
        
        return {
            'sentiment_variance': float(variance),
            'sentiment_mean': float(mean_sentiment),
            'sentiment_std': float(std),
            'sentiment_range': float(sentiment_range),
            'num_sentences': len(self.sentiment_scores)
        }
    
    def sentiment_distribution(self) -> Dict[str, float]:
        """
        Analyze sentiment distribution.
        
        Returns:
            Dictionary with distribution statistics
        """
        if len(self.sentiment_scores) == 0:
            self.compute_sentiment_scores()
        
        if len(self.sentiment_scores) == 0:
            return {
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'neutral_ratio': 0.0
            }
        
        positive = np.sum(self.sentiment_scores > 0.2) / len(self.sentiment_scores)
        negative = np.sum(self.sentiment_scores < -0.2) / len(self.sentiment_scores)
        neutral = 1 - positive - negative
        
        return {
            'positive_ratio': float(positive),
            'negative_ratio': float(negative),
            'neutral_ratio': float(neutral)
        }
    
    def compute_all_metrics(self, text: str, sentences: Optional[List[str]] = None) -> Dict:
        """
        Compute all sentiment-related metrics.
        
        Args:
            text: Input text
            sentences: Pre-split sentences (optional)
            
        Returns:
            Dictionary with all sentiment metrics
        """
        self.process_text(text, sentences)
        
        return {
            'variance': self.variance_of_sentiment(),
            'distribution': self.sentiment_distribution()
        }
