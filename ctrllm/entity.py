"""
Entity Analysis Module
Contains metrics related to named entities and specificity
"""

from typing import Dict
from collections import Counter
import stanza


class EntityAnalyzer:
    """
    Analyzes entity-related properties of text including:
    - Entity specificity
    - Entity type distribution
    - Entity counts
    """
    
    def __init__(self, lang: str = "en"):
        """
        Initialize entity analyzer with Stanza NER.
        
        Args:
            lang: Language code (e.g., 'en', 'zh', 'es')
        """
        try:
            self.nlp = stanza.Pipeline(
                lang=lang,
                processors='tokenize,ner',
                download_method=None
            )
        except Exception:
            print(f"Downloading Stanza {lang} models...")
            stanza.download(lang)
            self.nlp = stanza.Pipeline(
                lang=lang,
                processors='tokenize,ner'
            )
        
        self.doc = None
        
    def process_text(self, text: str):
        """
        Process text with Stanza NER.
        
        Args:
            text: Input text to analyze
        """
        self.doc = self.nlp(text)
    
    def entity_specificity(self) -> Dict[str, float]:
        """
        Calculate entity specificity based on named entities.
        Formula: Entity_specificity = #unique_entities / (T/100)
        
        Returns:
            Dictionary with entity counts and specificity scores
        """
        if self.doc is None:
            raise ValueError("No document processed. Call process_text() first.")
        
        # Count total tokens
        total_tokens = sum(
            len([w for w in sent.words if w.upos not in ['PUNCT', 'SYM']])
            for sent in self.doc.sentences
        )
        
        if total_tokens == 0:
            return {
                'entity_specificity': 0.0,
                'unique_entities': 0,
                'total_entities': 0,
                'entity_types': {},
                'person_count': 0,
                'org_count': 0,
                'location_count': 0
            }
        
        # Extract entities
        entities = []
        entity_types = Counter()
        
        for sent in self.doc.sentences:
            for ent in sent.ents:
                entities.append(ent.text)
                entity_types[ent.type] += 1
        
        unique_entities = len(set(entities))
        entity_specificity = unique_entities / (total_tokens / 100)
        
        # Consolidate entity types
        person_count = entity_types.get('PERSON', 0)
        date_count = entity_types.get('DATE', 0) + entity_types.get('TIME', 0)
        org_count = entity_types.get('ORG', 0) + entity_types.get('ORGANIZATION', 0)
        location_count = (entity_types.get('GPE', 0) + 
                         entity_types.get('LOC', 0) + 
                         entity_types.get('LOCATION', 0))
        
        return {
            'entity_specificity': float(entity_specificity),
            'unique_entities': unique_entities,
            'total_entities': len(entities),
            'entity_types': dict(entity_types),
            'person_count': person_count,
            'org_count': org_count,
            'location_count': location_count
            'date_count': date_count
        }
    
    def get_all_metrics(self) -> Dict:
        """
        Compute all entity metrics.
        
        Returns:
            Dictionary with all entity metrics
        """
        return self.entity_specificity()
