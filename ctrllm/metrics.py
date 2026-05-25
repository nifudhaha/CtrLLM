"""
Main Metrics Module
Unified interface for computing all text metrics
"""

from typing import Dict, Optional, List
from .syntactic import SyntacticAnalyzer
from .semantic import SemanticAnalyzer
from .entity import EntityAnalyzer
from .sentiment import SentimentAnalyzer
from .argument import ArgumentAnalyzer
from .rule_based import RuleBasedAnalyzer


class TextMetrics:
    """
    Main class for computing comprehensive text metrics.
    
    Integrates all analysis modules:
    - Syntactic 
    - Semantic 
    - Entity 
    - Sentiment 
    - Argument 
    - Rule-based 
    """
    
    def __init__(self,
                 lang: str = "en",
                 embedding_model: str = "all-mpnet-base-v2",
                 sentiment_model: str = "siebert/sentiment-roberta-large-english",
                 enable_rule_based: bool = True):
        """
        Initialize text metrics calculator.
        
        Args:
            lang: Language code for Stanza (e.g., 'en', 'zh', 'es')
            embedding_model: SentenceTransformer model name
            sentiment_model: HuggingFace sentiment model name
            enable_rule_based: Whether to enable rule-based analyzer (requires OpenAI API)
        """
        self.lang = lang
        
        # Initialize analyzers
        self.syntactic = SyntacticAnalyzer(lang=lang)
        self.semantic = SemanticAnalyzer(embedding_model=embedding_model)
        self.entity = EntityAnalyzer(lang=lang)
        self.sentiment = SentimentAnalyzer(model_name=sentiment_model)
        self.argument = ArgumentAnalyzer(lang=lang, embedding_model=embedding_model)
        
        # Rule-based analyzer (optional, requires API)
        self.rule_based = None
        if enable_rule_based:
            try:
                self.rule_based = RuleBasedAnalyzer()
            except Exception as e:
                print(f"Warning: Could not initialize RuleBasedAnalyzer: {e}")
        
        self.text = None
        self.sentences = []
        
    def compute_all(self,
                    text: str,
                    include_syntactic: bool = True,
                    include_semantic: bool = True,
                    include_entity: bool = True,
                    include_sentiment: bool = True,
                    include_argument: bool = True,
                    include_rule_based: bool = True) -> Dict:
        """
        Compute all available metrics for the given text.
        
        Args:
            text: Input text to analyze
            include_syntactic: Whether to compute syntactic metrics
            include_semantic: Whether to compute semantic metrics (slower)
            include_entity: Whether to compute entity metrics
            include_sentiment: Whether to compute sentiment metrics (slower)
            include_argument: Whether to compute argument metrics (slower, requires API)
            include_rule_based: Whether to compute rule-based metrics (requires API)
            
        Returns:
            Dictionary containing all computed metrics
            
        Note:
            - balanced_pro_con: Requires LLM
            - narrative_roles: Requires LLM
            - harm_index: Requires LLM
            - devil_angel_shift: No LLM (based on narrative_roles)
        """
        self.text = text
        
        # Initialize results
        results = {
            'text_length': len(text),
            'basic_stats': {}
        }
        
        # Process with syntactic analyzer (always need this for sentences)
        if include_syntactic:
            self.syntactic.process_text(text)
            self.sentences = self.syntactic.sentences
            
            # Count tokens
            total_tokens = sum(
                len([w for w in sent.words if w.upos not in ['PUNCT', 'SYM']])
                for sent in self.syntactic.doc.sentences
            )
            
            results['basic_stats'] = {
                'num_sentences': len(self.sentences),
                'num_tokens': total_tokens
            }
            
            # Compute syntactic metrics
            results.update(self.syntactic.get_all_metrics())
        
        # Semantic metrics
        if include_semantic and self.sentences:
            self.semantic.set_sentences(self.sentences)
            semantic_metrics = self.semantic.get_all_metrics()
            results['embedding_variance'] = semantic_metrics['embedding_variance']
            results['pairwise_similarity'] = semantic_metrics['pairwise_similarity']
        
        # Entity metrics
        if include_entity:
            self.entity.process_text(text)
            results['entity_specificity'] = self.entity.get_all_metrics()
        
        # Sentiment metrics
        if include_sentiment and self.sentences:
            self.sentiment.set_sentences(self.sentences)
            sentiment_metrics = self.sentiment.get_all_metrics()
            results['sentiment_variance'] = sentiment_metrics['variance']
            results['sentiment_distribution'] = sentiment_metrics['distribution']
        
        # Argument metrics
        if include_argument:
            self.argument.process_text(text)
            arg_metrics = self.argument.get_all_metrics()
            results['argument_metrics'] = {
                'num_arguments': arg_metrics['num_arguments'],
                'argumentativeness': arg_metrics['argumentativeness'],
                'main_vs_fringe': arg_metrics['main_vs_fringe'],
                'argument_diversity': arg_metrics['argument_diversity'],
                'argument_distinctness': arg_metrics['argument_distinctness'],
                'deliberation_intensity': arg_metrics['deliberation_intensity']
            }
        
        # Rule-based metrics (can run independently)
        if include_rule_based and self.rule_based:
            self.rule_based.set_text(text)
            
            # Set arguments if available (for API compatibility)
            if include_argument:
                self.rule_based.set_arguments(arg_metrics['arguments'])
            
            # Get all rule-based metrics
            rule_metrics = self.rule_based.get_all_metrics()
            results['rule_based_metrics'] = rule_metrics
        
        return results
    
    def compute_syntactic(self, text: str) -> Dict:
        """
        Compute only syntactic metrics (fastest).
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with syntactic metrics
        """
        return self.compute_all(
            text,
            include_syntactic=True,
            include_semantic=True,
            include_entity=True,
            include_sentiment=True
        )
    
    def compute_semantic(self, text: str) -> Dict:
        """
        Compute only semantic metrics.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with semantic metrics
        """
        self.syntactic.process_text(text)
        self.sentences = self.syntactic.sentences
        
        self.semantic.set_sentences(self.sentences)
        return self.semantic.get_all_metrics()
    
    def compute_entity(self, text: str) -> Dict:
        """
        Compute only entity metrics.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with entity metrics
        """
        self.entity.process_text(text)
        return self.entity.get_all_metrics()
    
    def compute_sentiment(self, text: str) -> Dict:
        """
        Compute only sentiment metrics.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment metrics
        """
        self.syntactic.process_text(text)
        self.sentences = self.syntactic.sentences
        
        self.sentiment.set_sentences(self.sentences)
        return self.sentiment.get_all_metrics()
    
    def compute_rule_based(self, 
                          text: str, 
                          arguments: Optional[List[str]] = None) -> Dict:
        """
        Compute only rule-based metrics.
        
        Args:
            text: Input text
            arguments: Pre-detected arguments (optional, for API compatibility)
            
        Returns:
            Dictionary with rule-based metrics:
            {
                'balanced_pro_con': {...},      # LLM-based
                'narrative_roles': {...},       # LLM-based
                'harm_index': {...},           # LLM-based
                'devil_angel_shift': {...}     # Based on narrative_roles
            }
            
        Note:
            - Requires enable_rule_based=True in __init__
            - All metrics require OPENAI_API_KEY
            
        Example:
            >>> results = metrics.compute_rule_based(text)
        """
        if not self.rule_based:
            raise ValueError(
                "RuleBasedAnalyzer not initialized. "
                "Set enable_rule_based=True in TextMetrics.__init__"
            )
        
        self.rule_based.set_text(text)
        
        if arguments:
            self.rule_based.set_arguments(arguments)
        
        return self.rule_based.get_all_metrics()
