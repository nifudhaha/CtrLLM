# Text Metrics Calculator - Project Summary

##  Project Overview

A Python toolkit for computing **quantifiable text complexity and diversity metrics** based on linguistic analysis. 

##  Key Features

### Implemented Metrics (All Directly Computable)

1. **Shannon's Entropy** - Vocabulary distribution uniformity
2. **Embedding Variance** - Semantic diversity in vector space  
3. **Sentiment Variance** - Emotional fluctuation across sentences
4. **POS Features** - Part-of-speech distribution analysis
5. **Vocabulary Complexity** - Grammatical flexibility of words
6. **Lexical Richness** - TTR, RTTR, CTTR metrics
7. **Entity Specificity** - Named entity density

##  Project Structure

```
text_metrics/
├── __init__.py           # Package initialization
├── metrics.py            # Core text metrics (lexical, syntactic, entities)
├── sentiment.py          # Sentiment-based metrics
├── descriptor.py         # Main unified interface
├── demo.py              # Usage examples
├── requirements.txt      # Dependencies
├── README.md            # User documentation
└── METRICS_EXPLAINED.md  # Detailed metric explanations
```

##  Quick Start

```python
from text_metrics import TextDescriptor

# Initialize
descriptor = TextDescriptor()

# Compute metrics
results = descriptor.compute_all_metrics(your_text)

# View summary
descriptor.print_summary(results)

# Save results
descriptor.save_results(results, "output.pkl")
```

##  What Each Module Does

### `metrics.py` - TextMetrics Class
**Purpose**: Core linguistic analysis

**Methods**:
- `variance_of_embeddings()` - Semantic diversity
- `shannon_entropy()` - Vocabulary distribution
- `pos_features()` - Grammatical analysis
- `entity_specificity()` - Named entity density
- `vocabulary_complexity()` - Word flexibility
- `lexical_richness()` - Vocabulary diversity

### `sentiment.py` - SentimentMetrics Class
**Purpose**: Emotional analysis

**Methods**:
- `variance_of_sentiment()` - Emotional fluctuation
- `sentiment_distribution()` - Positive/negative/neutral ratios
- `compute_sentiment_scores()` - Sentence-level sentiment

### `descriptor.py` - TextDescriptor Class
**Purpose**: Unified interface

**Methods**:
- `compute_all_metrics()` - One-stop computation
- `save_results()` - Persist to disk
- `load_results()` - Load from disk
- `print_summary()` - Formatted display

## 🔧 Dependencies

### Core Libraries
- **spaCy**: NLP processing (POS, NER, tokenization)
- **SentenceTransformers**: Semantic embeddings
- **HuggingFace Transformers**: Sentiment analysis
- **NumPy**: Numerical computations
- **scikit-learn**: Text vectorization

### Installation
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

##  Use Cases

### 1. Comparing Information Sources
**Question**: How does Wikipedia compare to LLM-generated content?

**Relevant metrics**:
- Shannon Entropy (vocabulary diversity)
- Embedding Variance (semantic diversity)
- Entity Specificity (concrete vs. abstract)

### 2. Bias Detection
**Question**: Does a text present multiple perspectives?

**Relevant metrics**:
- Sentiment Variance (emotional balance)
- Embedding Variance (topic diversity)
- Vocabulary Complexity (linguistic sophistication)

### 3. Writing Quality Assessment
**Question**: How complex and rich is this text?

**Relevant metrics**:
- Lexical Richness (vocabulary diversity)
- POS Features (grammatical structure)
- Shannon Entropy (word distribution)

### 4. Content Analysis at Scale
**Question**: Analyze 1000s of documents automatically

**Benefit**: All metrics are computable without human annotation

##  Typical Output

```python
{
    'basic_stats': {
        'num_sentences': 15,
        'num_tokens': 243
    },
    'shannon_entropy': {
        'entropy': 4.523,
        'normalized_entropy': 0.812
    },
    'pos_features': {
        'noun_ratio': 0.284,
        'verb_ratio': 0.158,
        'lexical_ratio': 0.621
    },
    'vocabulary_complexity': {
        'vocab_complexity': 1.234
    },
    'lexical_richness': {
        'ttr': 0.716,
        'rttr': 6.972
    },
    'entity_specificity': {
        'entity_specificity': 3.158,
        'unique_entities': 8
    },
    'embedding_variance': 0.453,
    'sentiment_variance': {
        'sentiment_variance': 0.023,
        'sentiment_mean': 0.125
    }
}
```

##  Advantages

1. **No Manual Rules**: All metrics computed automatically
2. **Reproducible**: Same input always produces same output
3. **Scalable**: Can process large datasets
4. **Modular**: Use only the metrics you need
5. **Well-documented**: Clear formulas and interpretations

##  Limitations

1. **Model-dependent**: Results depend on underlying models
2. **English-only**: Current implementation for English text
3. **Computational cost**: Embedding/sentiment analysis can be slow
4. **Binary sentiment**: Simple positive/negative classification

##  Future Enhancements

Potential additions (not yet implemented):

1. **Argument Detection**: Identify pro/con arguments
2. **Polarity Metrics**: Stance distribution analysis
3. **Causal Mechanisms**: Detect cause-effect relationships
4. **Narrative Roles**: Hero/villain/victim detection
5. **Uncertainty Markers**: Hedge words and polysemy

These would require:
- Trained argument detection models
- Stance classification models
- LLM-based annotations

##  Documentation

- **README.md**: User guide and quick start
- **METRICS_EXPLAINED.md**: Detailed metric descriptions with formulas
- **demo.py**: Working examples
- **This file**: Project overview

## 🎓 Academic Context

Based on research comparing Wikipedia, Britannica, and LLM-generated content on controversial topics. Metrics designed to capture:

- **Perspective Multiplexity**: Multiple viewpoints
- **Syntactic Complexity**: Linguistic sophistication  
- **Semantic Diversity**: Topic coverage breadth


##  Contributing

To extend this toolkit:

1. Add new metrics to `metrics.py` or `sentiment.py`
2. Update `descriptor.py` to integrate new metrics
3. Add tests and documentation
4. Keep metrics **directly computable** (no manual rules)



## License

MIT License - Free for academic and commercial use
