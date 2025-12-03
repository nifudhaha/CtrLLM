

import numpy as np
from typing import List, Dict, Optional, Tuple
import pickle
from collections import Counter
import stanza
from sentence_transformers import SentenceTransformer


class TextMetrics:
    """
    A class for computing various text complexity and diversity metrics using Stanza.
    
    This class provides methods to calculate:
    - Sentiment variance
    - Embedding variance  
    - Shannon's entropy
    - POS features
    - Entity specificity
    - Lexical richness metrics
    """
    
    def __init__(self, 
                 lang: str = "en",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the TextMetrics calculator with Stanza.
        
        Args:
            lang: Language code for Stanza (e.g., "en", "zh", "es")
            embedding_model: SentenceTransformer model name
        """
        # Initialize Stanza pipeline
        # processors: tokenize, pos, lemma, ner
        try:
            self.nlp = stanza.Pipeline(
                lang=lang,
                processors='tokenize,pos,lemma,ner',
                download_method=None  # Don't auto-download
            )
        except Exception as e:
            print(f"Downloading Stanza {lang} models...")
            stanza.download(lang)
            self.nlp = stanza.Pipeline(
                lang=lang,
                processors='tokenize,pos,lemma,ner'
            )
            
        self.embedding_model = SentenceTransformer(embedding_model)
        self.doc = None
        self.sentences = []
        self.embeddings = None
        
    def process_text(self, text: str):
        """
        Process text with Stanza and extract sentences.
        
        Args:
            text: Input text to analyze
        """
        self.doc = self.nlp(text)
        # Stanza sentences are accessed via doc.sentences
        self.sentences = [sent.text.strip() for sent in self.doc.sentences if sent.text.strip()]
        
    def compute_embeddings(self) -> np.ndarray:
        """
        Compute sentence embeddings.
        
        Returns:
            Array of sentence embeddings with shape (N, embedding_dim)
        """
        if not self.sentences:
            raise ValueError("No sentences found. Call process_text() first.")
            
        self.embeddings = self.embedding_model.encode(self.sentences)
        return self.embeddings
    
    def variance_of_embeddings(self) -> float:
        """
        Calculate variance of sentence embeddings in semantic space.
        Formula: σ²_emb = (1/N) Σ ||e_i - e_mean||²
        
        Returns:
            Variance of embeddings
        """
        if self.embeddings is None:
            self.compute_embeddings()
            
        mean_embedding = np.mean(self.embeddings, axis=0)
        variance = np.mean([np.linalg.norm(emb - mean_embedding)**2 
                           for emb in self.embeddings])
        return float(variance)
    
    def shannon_entropy(self, use_lemma: bool = True) -> Dict[str, float]:
        """
        Calculate Shannon's entropy of token/lemma distribution.
        Formula: H = -Σ p(w) log p(w)
        
        Args:
            use_lemma: If True, use lemmas; if False, use raw tokens
            
        Returns:
            Dictionary with 'entropy' and 'normalized_entropy'
        """
        if self.doc is None:
            raise ValueError("No document processed. Call process_text() first.")
            
        # Extract tokens/lemmas from Stanza
        tokens = []
        for sent in self.doc.sentences:
            for word in sent.words:
                # Skip punctuation
                if word.upos not in ['PUNCT', 'SYM', 'X']:
                    if use_lemma:
                        tokens.append(word.lemma.lower())
                    else:
                        tokens.append(word.text.lower())
        
        if not tokens:
            return {'entropy': 0.0, 'normalized_entropy': 0.0}
            
        # Calculate probability distribution
        total_tokens = len(tokens)
        token_counts = Counter(tokens)
        probs = np.array([count / total_tokens for count in token_counts.values()])
        
        # Shannon entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        # Normalized entropy
        unique_tokens = len(token_counts)
        normalized_entropy = entropy / np.log2(unique_tokens) if unique_tokens > 1 else 0.0
        
        return {
            'entropy': float(entropy),
            'normalized_entropy': float(normalized_entropy),
            'unique_tokens': unique_tokens,
            'total_tokens': total_tokens
        }
    
    def pos_features(self) -> Dict[str, float]:
        """
        Calculate POS (Part-of-Speech) distribution features using Stanza UPOS tags.
        
        Returns:
            Dictionary with ratios for different POS categories
        """
        if self.doc is None:
            raise ValueError("No document processed. Call process_text() first.")
            
        # Collect all words (excluding punctuation)
        words = []
        for sent in self.doc.sentences:
            for word in sent.words:
                if word.upos not in ['PUNCT', 'SYM']:
                    words.append(word)
        
        total_tokens = len(words)
        
        if total_tokens == 0:
            return {
                'noun_ratio': 0.0,
                'verb_ratio': 0.0,
                'adj_ratio': 0.0,
                'adv_ratio': 0.0,
                'lexical_ratio': 0.0,
                'pos_variability': 0.0
            }
        
        # Count UPOS tags (Universal POS tags)
        pos_counts = Counter([word.upos for word in words])
        
        # Stanza uses UPOS: NOUN, PROPN, VERB, ADJ, ADV, etc.
        noun_count = pos_counts.get('NOUN', 0) + pos_counts.get('PROPN', 0)
        verb_count = pos_counts.get('VERB', 0)
        adj_count = pos_counts.get('ADJ', 0)
        adv_count = pos_counts.get('ADV', 0)
        
        # Lexical words (content words)
        lexical_count = noun_count + verb_count + adj_count + adv_count
        
        # POS variability (number of distinct POS tags)
        pos_variability = len(pos_counts) / total_tokens
        
        return {
            'noun_ratio': noun_count / total_tokens,
            'verb_ratio': verb_count / total_tokens,
            'adj_ratio': adj_count / total_tokens,
            'adv_ratio': adv_count / total_tokens,
            'lexical_ratio': lexical_count / total_tokens,
            'pos_variability': pos_variability,
            'distinct_pos_tags': len(pos_counts)
        }
    
    def entity_specificity(self) -> Dict[str, float]:
        """
        Calculate entity specificity based on named entities from Stanza NER.
        Formula: Entity_specificity = #unique_entities / (T/100)
        
        Returns:
            Dictionary with entity counts and specificity scores
        """
        if self.doc is None:
            raise ValueError("No document processed. Call process_text() first.")
            
        # Count total tokens
        total_tokens = sum(len([w for w in sent.words if w.upos not in ['PUNCT', 'SYM']]) 
                          for sent in self.doc.sentences)
        
        if total_tokens == 0:
            return {
                'entity_specificity': 0.0,
                'unique_entities': 0,
                'total_entities': 0,
                'entity_types': {}
            }
        
        # Extract entities from Stanza
        # Stanza entities are accessed via doc.entities
        entities = []
        entity_types = Counter()
        
        for sent in self.doc.sentences:
            for ent in sent.ents:
                entities.append(ent.text)
                entity_types[ent.type] += 1
        
        unique_entities = len(set(entities))
        
        # Calculate specificity (normalized per 100 tokens)
        entity_specificity = unique_entities / (total_tokens / 100)
        
        return {
            'entity_specificity': float(entity_specificity),
            'unique_entities': unique_entities,
            'total_entities': len(entities),
            'entity_types': dict(entity_types),
            'person_count': entity_types.get('PERSON', 0),
            'org_count': entity_types.get('ORG', 0) + entity_types.get('ORGANIZATION', 0),
            'gpe_count': entity_types.get('GPE', 0) + entity_types.get('LOC', 0)
        }
    
    def vocabulary_complexity(self) -> Dict[str, float]:
        """
        Calculate vocabulary complexity based on POS tag diversity per lemma.
        Formula: Vocab_complexity = (1/#lemmas) Σ k_ℓ
        where k_ℓ is the number of distinct POS tags for lemma ℓ
        
        Returns:
            Dictionary with vocabulary complexity metrics
        """
        if self.doc is None:
            raise ValueError("No document processed. Call process_text() first.")
            
        # Build lemma -> POS tags mapping
        lemma_pos = {}
        for sent in self.doc.sentences:
            for word in sent.words:
                if word.upos not in ['PUNCT', 'SYM']:
                    lemma = word.lemma.lower()
                    if lemma not in lemma_pos:
                        lemma_pos[lemma] = set()
                    lemma_pos[lemma].add(word.upos)
        
        if not lemma_pos:
            return {
                'vocab_complexity': 0.0,
                'unique_lemmas': 0,
                'avg_pos_per_lemma': 0.0
            }
        
        # Calculate average POS tag count per lemma
        pos_counts = [len(pos_set) for pos_set in lemma_pos.values()]
        vocab_complexity = np.mean(pos_counts)
        
        return {
            'vocab_complexity': float(vocab_complexity),
            'unique_lemmas': len(lemma_pos),
            'avg_pos_per_lemma': float(vocab_complexity),
            'max_pos_per_lemma': max(pos_counts),
            'lemmas_with_multiple_pos': sum(1 for c in pos_counts if c > 1)
        }
    
    def lexical_richness(self) -> Dict[str, float]:
        """
        Calculate various lexical richness metrics.
        Includes TTR (Type-Token Ratio) and related measures.
        
        Returns:
            Dictionary with lexical richness metrics
        """
        if self.doc is None:
            raise ValueError("No document processed. Call process_text() first.")
            
        # Extract tokens (only alphabetic words)
        tokens = []
        for sent in self.doc.sentences:
            for word in sent.words:
                if word.upos not in ['PUNCT', 'SYM', 'X'] and word.text.isalpha():
                    tokens.append(word.text.lower())
        
        if not tokens:
            return {
                'ttr': 0.0,
                'rttr': 0.0,
                'cttr': 0.0,
                'unique_tokens': 0,
                'total_tokens': 0
            }
        
        unique_tokens = len(set(tokens))
        total_tokens = len(tokens)
        
        # Type-Token Ratio
        ttr = unique_tokens / total_tokens
        
        # Root TTR
        rttr = unique_tokens / np.sqrt(total_tokens)
        
        # Corrected TTR
        cttr = unique_tokens / np.sqrt(2 * total_tokens)
        
        return {
            'ttr': float(ttr),
            'rttr': float(rttr),
            'cttr': float(cttr),
            'unique_tokens': unique_tokens,
            'total_tokens': total_tokens
        }
    
    def compute_all_metrics(self, text: str) -> Dict:
        """
        Compute all available metrics for the given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing all computed metrics
        """
        self.process_text(text)
        
        # Count total tokens for basic stats
        total_tokens = sum(len([w for w in sent.words if w.upos not in ['PUNCT', 'SYM']]) 
                          for sent in self.doc.sentences)
        
        # Calculate average sentence length
        avg_sent_length = total_tokens / len(self.doc.sentences) if self.doc.sentences else 0
        
        metrics = {
            'basic_stats': {
                'num_sentences': len(self.sentences),
                'num_tokens': total_tokens,
                'avg_sentence_length': avg_sent_length
            },
            'embedding_variance': self.variance_of_embeddings(),
            'shannon_entropy': self.shannon_entropy(),
            'pos_features': self.pos_features(),
            'entity_specificity': self.entity_specificity(),
            'vocabulary_complexity': self.vocabulary_complexity(),
            'lexical_richness': self.lexical_richness()
        }
        
        return metrics
    
    def save_results(self, results: Dict, pkl_file_path: str):
        """
        Save computed metrics to a pickle file.
        
        Args:
            results: Dictionary of computed metrics
            pkl_file_path: File path where results will be saved
        """
        with open(pkl_file_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved to {pkl_file_path}")
    
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
        return results