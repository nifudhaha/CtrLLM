# Metrics Explanation

## Overview

This document explains each metric in detail, including its formula, interpretation, and use cases.

---

## 1. Shannon's Entropy

### What it measures
The diversity and uniformity of vocabulary distribution in the text.

### Formula
```
H(token) = -Σ p(w) log₂ p(w)

where:
- p(w) = frequency of word w / total words
- Summation over all unique words
```

### Normalized version
```
H_norm = H(token) / log₂(#unique_tokens)
```

### Interpretation
- **Higher entropy** (>4.0): Very diverse vocabulary, words used relatively evenly
- **Lower entropy** (<3.0): Repetitive vocabulary, some words dominate
- **Normalized range**: 0-1, where 1 means perfectly uniform distribution

### Use case
Comparing vocabulary diversity across different sources (e.g., Wikipedia vs. LLM-generated text)

---

## 2. Embedding Variance

### What it measures
Semantic diversity across sentences in the text.

### Formula
```
σ²_emb = (1/N) Σ ||e_i - e_mean||²

where:
- e_i = embedding vector for sentence i
- e_mean = mean embedding across all sentences
- || || = L2 norm (Euclidean distance)
```

### Interpretation
- **Higher variance** (>0.4): Sentences cover diverse semantic topics
- **Lower variance** (<0.2): Sentences are semantically similar/cohesive
- **Zero variance**: All sentences are identical in meaning

### Use case
Detecting whether a text presents multiple perspectives or focuses narrowly on one theme

---

## 3. POS (Part-of-Speech) Features

### What it measures
Distribution of grammatical categories in the text.

### Metrics included

#### Noun Ratio
```
noun_ratio = (#NOUN + #PROPN) / total_tokens
```
Higher values indicate more entity/concept-focused text

#### Verb Ratio
```
verb_ratio = #VERB / total_tokens
```
Higher values indicate more action/process-focused text

#### Lexical Ratio
```
lexical_ratio = (#NOUN + #VERB + #ADJ + #ADV) / total_tokens
```
- **Higher ratio** (>0.6): Content-rich, informative text
- **Lower ratio** (<0.4): More function words, potentially simpler structure

#### POS Variability
```
pos_variability = #distinct_POS_tags / total_tokens
```
Measures grammatical diversity

### Use case
Analyzing writing style and information density

---

## 4. Vocabulary Complexity

### What it measures
The grammatical flexibility of words - how many different grammatical roles each word plays.

### Formula
```
vocab_complexity = (1 / #lemmas) Σ k_ℓ

where:
- k_ℓ = number of distinct POS tags for lemma ℓ
- Summation over all unique lemmas
```

### Example
The word "run" might appear as:
- NOUN: "a morning run"
- VERB: "I run daily"
This lemma would contribute k=2 to the complexity score.

### Interpretation
- **Higher complexity** (>1.3): Words used in multiple grammatical contexts
- **Lower complexity** (~1.0): Each word tends to appear in only one grammatical role

### Use case
Assessing linguistic sophistication and grammatical variety

---

## 5. Lexical Richness

### What it measures
Multiple aspects of vocabulary diversity.

### Metrics included

#### Type-Token Ratio (TTR)
```
TTR = unique_words / total_words
```
- Range: 0-1
- **Higher is more diverse**
- Biased by text length

#### Root TTR (RTTR)
```
RTTR = unique_words / √(total_words)
```
- Less affected by text length
- More stable for comparison

#### Corrected TTR (CTTR)
```
CTTR = unique_words / √(2 × total_words)
```
- Further length correction
- Recommended for cross-text comparison

### Interpretation
- **TTR > 0.7**: Very diverse vocabulary (short texts)
- **TTR 0.4-0.6**: Moderate diversity (typical articles)
- **TTR < 0.4**: Repetitive vocabulary

### Use case
Measuring vocabulary richness independent of text length

---

## 6. Entity Specificity

### What it measures
The density of specific named entities (people, organizations, places).

### Formula
```
entity_specificity = #unique_entities / (total_tokens / 100)
```
Normalized per 100 words for comparability

### Entity types tracked
- **PERSON**: Individual names (e.g., "Barack Obama")
- **ORG**: Organizations (e.g., "NASA", "Harvard University")
- **GPE**: Geopolitical entities (e.g., "United States", "Tokyo")
- **LOC**: Locations (e.g., "Mount Everest")

### Interpretation
- **High specificity** (>5.0): Text references many specific entities
- **Low specificity** (<2.0): Abstract or general discussion
- **Zero**: No named entities (rare in informative text)

### Use case
Detecting concrete vs. abstract writing style, fact-density

---

## 7. Sentiment Variance

### What it measures
Emotional fluctuation and diversity across sentences.

### Formula
```
σ²_sentiment = (1/N) Σ (s_i - μ_s)²

where:
- s_i = sentiment score for sentence i (range: -1 to 1)
- μ_s = mean sentiment across all sentences
- N = number of sentences
```

### Additional metrics
- **Sentiment Mean**: Overall emotional tone
- **Sentiment Range**: max(s_i) - min(s_i)
- **Distribution**: % positive/negative/neutral sentences

### Interpretation
- **High variance** (>0.15): Emotional ups and downs, mixed perspectives
- **Low variance** (<0.05): Emotionally consistent, single-toned
- **Mean near 0**: Balanced or neutral tone
- **Mean > 0.3**: Overall positive tone
- **Mean < -0.3**: Overall negative tone

### Use case
Analyzing emotional balance in controversial topic coverage

---

## Combined Interpretation

### Diverse/Multi-perspective Text
Typically shows:
- ✓ High Shannon entropy (>4.0)
- ✓ High embedding variance (>0.3)
- ✓ Moderate-high sentiment variance (>0.08)
- ✓ High entity specificity (>4.0)
- ✓ High lexical richness (TTR >0.6)

### Narrow/Single-perspective Text
Typically shows:
- ✗ Low Shannon entropy (<3.5)
- ✗ Low embedding variance (<0.2)
- ✗ Low sentiment variance (<0.05)
- ✗ Low entity specificity (<2.0)
- ✗ Low lexical richness (TTR <0.5)

---

## Metric Selection Guide

### For comparing information sources
Use: **Shannon Entropy, Embedding Variance, Entity Specificity**

### For analyzing writing quality
Use: **Lexical Richness, Vocabulary Complexity, POS Features**

### For detecting bias/perspective diversity
Use: **Sentiment Variance, Embedding Variance**

### For assessing information density
Use: **Entity Specificity, Lexical Ratio, Shannon Entropy**

---

## Technical Notes

### Computation Requirements

| Metric | Requires | Speed |
|--------|----------|-------|
| Shannon Entropy | spaCy tokenization | Fast |
| POS Features | spaCy POS tagging | Fast |
| Lexical Richness | spaCy tokenization | Fast |
| Vocabulary Complexity | spaCy POS + lemmatization | Fast |
| Entity Specificity | spaCy NER | Fast |
| Embedding Variance | SentenceTransformer | Slow |
| Sentiment Variance | HuggingFace model | Slow |

### Model Dependencies
- **spaCy**: `en_core_web_sm` or larger
- **SentenceTransformers**: `all-MiniLM-L6-v2` (default)
- **Sentiment**: `distilbert-base-uncased-finetuned-sst-2-english`

### Scalability
- ✓ Short texts (<1000 words): All metrics fast
- ⚠ Medium texts (1000-5000 words): Embedding/sentiment may be slow
- ✗ Long texts (>5000 words): Consider batching or disabling slow metrics
