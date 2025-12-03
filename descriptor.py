from typing import Dict, Optional
import pickle
from metrics import TextMetrics
from sentiment import SentimentMetrics


class TextDescriptor:
    """
    Main class for computing comprehensive text metrics using Stanza.
    
    Provides a unified interface for calculating:
    - Lexical and syntactic complexity
    - Sentiment variance
    - Entity features
    - Embedding-based metrics
    """
    
    def __init__(self,
                 nlp_lang: str = "en",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize the TextDescriptor with Stanza.
        
        Args:
            nlp_lang: Language code for Stanza (e.g., "en", "zh", "es")
            embedding_model: SentenceTransformer model name
            sentiment_model: HuggingFace sentiment model name
        """
        self.text_metrics = TextMetrics(nlp_lang, embedding_model)
        self.sentiment_metrics = SentimentMetrics(sentiment_model)
        
    def compute_all_metrics(self, 
                           text: str,
                           include_sentiment: bool = True,
                           include_embeddings: bool = True,
                           include_entities: bool = True) -> Dict:
        """
        Compute all available metrics for the given text.
        
        Args:
            text: Input text to analyze
            include_sentiment: Whether to compute sentiment metrics
            include_embeddings: Whether to compute embedding metrics
            include_entities: Whether to compute entity metrics
            
        Returns:
            Dictionary containing all computed metrics organized by category
        """
        # Process text
        self.text_metrics.process_text(text)
        
        # Count tokens for Stanza (must iterate through sentences -> words)
        total_tokens = sum(
            len([w for w in sent.words if w.upos not in ['PUNCT', 'SYM']])
            for sent in self.text_metrics.doc.sentences
        )
        
        # Initialize results
        results = {
            'text_length': len(text),
            'basic_stats': {
                'num_sentences': len(self.text_metrics.sentences),
                'num_tokens': total_tokens,
            }
        }
        
        # Lexical and syntactic features (always computed)
        results['shannon_entropy'] = self.text_metrics.shannon_entropy()
        results['pos_features'] = self.text_metrics.pos_features()
        results['vocabulary_complexity'] = self.text_metrics.vocabulary_complexity()
        results['lexical_richness'] = self.text_metrics.lexical_richness()
        
        # Optional: Embedding metrics
        if include_embeddings:
            results['embedding_variance'] = self.text_metrics.variance_of_embeddings()
        
        # Optional: Entity metrics
        if include_entities:
            results['entity_specificity'] = self.text_metrics.entity_specificity()
        
        # Optional: Sentiment metrics
        if include_sentiment:
            sentiment_results = self.sentiment_metrics.compute_all_metrics(
                text, 
                sentences=self.text_metrics.sentences
            )
            results['sentiment_variance'] = sentiment_results['variance']
            results['sentiment_distribution'] = sentiment_results['distribution']
        
        return results
    
    def save_results(self, results: Dict, pkl_file_path: str):
        """
        Save computed metrics to a pickle file.
        
        Args:
            results: Dictionary of computed metrics
            pkl_file_path: File path where results will be saved
        """
        with open(pkl_file_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"✓ Results saved to {pkl_file_path}")
    
    def load_results(self, pkl_file_path: str) -> Dict:
        """
        Load previously computed metrics from a pickle file.
        
        Args:
            pkl_file_path: File path to load results from
            
        Returns:
            Dictionary of loaded metrics
        """
        with open(pkl_file_path, 'rb') as f:
            results = pickle.load(f)
        print(f"✓ Results loaded from {pkl_file_path}")
        return results
    
    def print_summary(self, results: Dict):
        """
        Print a formatted summary of the computed metrics.
        
        Args:
            results: Dictionary of computed metrics
        """
        print("\n" + "="*60)
        print("TEXT METRICS SUMMARY (Stanza)")
        print("="*60)
        
        # Basic stats
        print(f"\n Basic Statistics:")
        print(f"  Text Length: {results['text_length']} characters")
        print(f"  Sentences: {results['basic_stats']['num_sentences']}")
        print(f"  Tokens: {results['basic_stats']['num_tokens']}")
        
        # Shannon entropy
        print(f"\n Shannon Entropy:")
        entropy = results['shannon_entropy']
        print(f"  Entropy: {entropy['entropy']:.3f}")
        print(f"  Normalized: {entropy['normalized_entropy']:.3f}")
        print(f"  Unique/Total: {entropy['unique_tokens']}/{entropy['total_tokens']}")
        
        # POS features
        print(f"\n POS Features:")
        pos = results['pos_features']
        print(f"  Noun Ratio: {pos['noun_ratio']:.3f}")
        print(f"  Verb Ratio: {pos['verb_ratio']:.3f}")
        print(f"  Lexical Ratio: {pos['lexical_ratio']:.3f}")
        print(f"  POS Variability: {pos['pos_variability']:.3f}")
        
        # Lexical richness
        print(f"\n Lexical Richness:")
        lr = results['lexical_richness']
        print(f"  TTR: {lr['ttr']:.3f}")
        print(f"  RTTR: {lr['rttr']:.3f}")
        print(f"  Unique/Total: {lr['unique_tokens']}/{lr['total_tokens']}")
        
        # Vocabulary complexity
        print(f"\n Vocabulary Complexity:")
        vc = results['vocabulary_complexity']
        print(f"  Complexity: {vc['vocab_complexity']:.3f}")
        print(f"  Unique Lemmas: {vc['unique_lemmas']}")
        
        # Entity specificity (if available)
        if 'entity_specificity' in results:
            print(f"\n Entity Specificity:")
            es = results['entity_specificity']
            print(f"  Specificity: {es['entity_specificity']:.3f}")
            print(f"  Unique Entities: {es['unique_entities']}")
            print(f"  Persons: {es['person_count']}, Orgs: {es['org_count']}")
        
        # Embedding variance (if available)
        if 'embedding_variance' in results:
            print(f"\n Embedding Variance:")
            print(f"  Variance: {results['embedding_variance']:.3f}")
        
        # Sentiment (if available)
        if 'sentiment_variance' in results:
            print(f"\n Sentiment Metrics:")
            sv = results['sentiment_variance']
            print(f"  Mean: {sv['sentiment_mean']:.3f}")
            print(f"  Variance: {sv['sentiment_variance']:.3f}")
            print(f"  Range: {sv['sentiment_range']:.3f}")
        
        print("\n" + "="*60 + "\n")