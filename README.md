# Text Metrics Calculator

A Python toolkit for computing quantifiable text complexity and diversity metrics based on linguistic analysis.

## Features

This package provides comprehensive text analysis metrics organized into several categories:

### 📊 Basic Statistics
- Sentence count
- Token count  
- Average sentence length

### 🔢 Syntactic & Lexical Complexity
- **Shannon's Entropy**: Measures vocabulary distribution uniformity
- **POS Features**: Part-of-speech distribution (nouns, verbs, adjectives, etc.)
- **Vocabulary Complexity**: Average number of POS tags per lemma
- **Lexical Richness**: TTR, RTTR, CTTR metrics

### 🎯 Semantic Features
- **Embedding Variance**: Semantic diversity in sentence vector space
- **Entity Specificity**: Named entity density (persons, organizations, locations)

### 😊 Sentiment Analysis
- **Sentiment Variance**: Emotional fluctuation across sentences
- **Sentiment Distribution**: Positive/negative/neutral ratios

## Installation

```bash
# Install required packages
pip install spacy sentence-transformers transformers torch numpy scikit-learn

# Download stanza model
pip install stanza
python -c "import stanza; stanza.download('en')"
```

## Quick Start

```python
from text_metrics import TextDescriptor

# Initialize the descriptor
descriptor = TextDescriptor()

# Sample text
text = """
Climate change represents one of the most significant challenges facing humanity. 
Scientists worldwide agree that global temperatures are rising due to greenhouse gas emissions.
However, debates continue about the best policy approaches to address this crisis.
Some advocate for immediate, dramatic action, while others prefer gradual transitions.
"""

# Compute all metrics
results = descriptor.compute_all_metrics(text)

# Print summary
descriptor.print_summary(results)

# Save results
descriptor.save_results(results, "metrics_output.pkl")
```

## Detailed Usage

### Using Individual Metric Classes

```python
from text_metrics import TextMetrics, SentimentMetrics

# Text-based metrics only
text_metrics = TextMetrics()
text_metrics.process_text(text)

# Get specific metrics
entropy = text_metrics.shannon_entropy()
pos_features = text_metrics.pos_features()
entity_info = text_metrics.entity_specificity()

# Sentiment metrics separately
sentiment_metrics = SentimentMetrics()
sentiment_results = sentiment_metrics.compute_all_metrics(text)
```

### Selective Computation

```python
# Disable expensive computations
results = descriptor.compute_all_metrics(
    text,
    include_sentiment=False,  # Skip sentiment analysis
    include_embeddings=False,  # Skip embedding computation
    include_entities=True      # Include entity analysis
)
```

## Metrics Reference

### Shannon's Entropy
Measures the uniformity of word/token distribution:
- **Formula**: `H = -Σ p(w) log p(w)`
- **Higher values** = more diverse vocabulary
- **Normalized version** accounts for vocabulary size

### Variance of Embeddings  
Quantifies semantic diversity using sentence embeddings:
- **Formula**: `σ²_emb = (1/N) Σ ||e_i - e_mean||²`
- **Higher values** = more semantically diverse content

### POS Features
Distribution of grammatical categories:
- Noun/Verb/Adjective/Adverb ratios
- Lexical ratio (content words / total)
- POS variability (distinct tags / total)

### Vocabulary Complexity
Average number of different POS tags per lemma:
- **Formula**: `(1/#lemmas) Σ k_ℓ` where k_ℓ = distinct POS tags for lemma ℓ
- Indicates grammatical flexibility of vocabulary

### Entity Specificity
Named entity density normalized per 100 tokens:
- **Formula**: `#unique_entities / (T/100)`
- Tracks persons, organizations, and locations

### Lexical Richness
Multiple measures of vocabulary diversity:
- **TTR**: Type-Token Ratio = unique_tokens / total_tokens
- **RTTR**: Root TTR = unique / sqrt(total)
- **CTTR**: Corrected TTR = unique / sqrt(2 * total)

### Sentiment Variance
Emotional fluctuation across text:
- **Formula**: `σ²_s = (1/N) Σ (s_i - μ_s)²`
- Maps sentiment to [-1, 1] scale
- **Higher variance** = more emotional variability

## Output Format

Results are returned as a nested dictionary:

```python
{
    'text_length': 450,
    'basic_stats': {
        'num_sentences': 10,
        'num_tokens': 95
    },
    'shannon_entropy': {
        'entropy': 4.523,
        'normalized_entropy': 0.812,
        'unique_tokens': 68,
        'total_tokens': 95
    },
    'pos_features': {
        'noun_ratio': 0.284,
        'verb_ratio': 0.158,
        'lexical_ratio': 0.621,
        'pos_variability': 0.142
    },
    'vocabulary_complexity': {
        'vocab_complexity': 1.234,
        'unique_lemmas': 72,
        'max_pos_per_lemma': 3
    },
    'lexical_richness': {
        'ttr': 0.716,
        'rttr': 6.972,
        'cttr': 4.931
    },
    'entity_specificity': {
        'entity_specificity': 3.158,
        'unique_entities': 3,
        'person_count': 1,
        'org_count': 2
    },
    'embedding_variance': 0.453,
    'sentiment_variance': {
        'sentiment_variance': 0.023,
        'sentiment_mean': 0.125,
        'sentiment_range': 0.567
    }
}
```

## Use Cases

This toolkit is designed for:

1. **Comparing text sources** (e.g., Wikipedia vs. Britannica vs. LLM-generated content)
2. **Analyzing writing complexity and diversity**
3. **Content quality assessment**
4. **Educational text analysis**
5. **Research on information presentation**

## Implementation Notes

- **stanza** for NLP processing (POS tagging, NER, dependency parsing)
- **SentenceTransformers** for semantic embeddings
- **HuggingFace Transformers** for sentiment analysis
- All metrics are **directly computable** 

## Limitations

- Sentiment analysis uses a binary model (positive/negative)
- Embedding variance depends on model choice
- Entity recognition quality varies by text domain
- Computationally intensive for very long texts

## References
TBD

## License

MIT License
