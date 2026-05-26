

# ctrLLM

**Controversial Topic Representation in LLM**

A comprehensive Python package for analyzing text diversity and controversial topic representation in language models.

---

## 🎯 Purpose

This toolkit helps researchers and developers analyze how different sources (Wikipedia, Britannica, LLMs) represent controversial topics by measuring:

- **Text diversity** (vocabulary, semantic, lexical)
- **Perspective multiplexity** (multiple viewpoints)
- **Content quality** (complexity, richness)
- **Rule-based discourse framing** (balance framing, narrative roles)

---

## 🚀 Quick Start

### Installation
```bash
# Install package
pip install -e .

# Install dependencies (including OpenAI)
pip install -r requirements.txt

# Download Stanza models
python -c "import stanza; stanza.download('en')"
```

### API Keys
This package relies on two external APIs:

```bash
# OpenAI — for LLM-based analysis (narrative roles, balanced pros and cons, stakeholder's claims)
export OPENAI_API_KEY='your-openai-api-key'

# Set WIBA API token for argument detection
# You can get an API key at https://wiba.dev/
export WIBA_API_TOKEN='your-wiba-api-token'
```

### Basic Usage

```python
from ctrllm import TextMetrics, print_summary

# Initialize
metrics = TextMetrics(lang='en')

# Analyze text
text = "Your controversial topic text here..."
results = metrics.compute_all(text)

# Print results
print_summary(results)
```

---

## 📦 Package Structure

```
ctrllm_package/
├── ctrllm/
│   ├── __init__.py
│   ├── metrics.py
│   ├── syntactic.py
│   ├── semantic.py
│   ├── entity.py
│   ├── sentiment.py
│   ├── argument.py
│   ├── rule_based.py
│   └── utils.py
├── demo.py
├── test.py
├── requirements.txt
├── setup.py
└── README.md

```

---

## 📊 Features

### Syntactic Analysis (`syntactic.py`)
- **Shannon Entropy** - Vocabulary distribution
- **POS Features** - Grammatical analysis
- **Vocabulary Complexity** - Word flexibility
- **Lexical Richness** - TTR, RTTR, CTTR

### Semantic Analysis (`semantic.py`)
- **Embedding Variance** - Semantic diversity
- **Pairwise Similarity** - Sentence similarity

### Entity Analysis (`entity.py`)
- **Entity Specificity** - Named entity density
- **Entity Distribution** - Person/Org/Location counts

### Sentiment Analysis (`sentiment.py`)
- **Sentiment Variance** - Emotional variability
- **Sentiment Distribution** - Positive/negative/neutral ratios

### **Argument Analysis (`argument.py`)**
- **Main vs. Fringe Perspective** - Main/fringe viewpoint ratio
- **Argument Diversity** - Argument topic diversity  
- **Argument Distinctness** - Cluster separation
- **Argumentativeness** - Argument density
- 
### **Rule-Based Analysis (`rule_based.py`)** 

A rule-based + LLM-assisted module for detecting higher-level discourse patterns not captured by syntactic or semantic metrics.

#### **1. Balanced Pro/Con Detection**
Identifies sentences that present **both pro and con viewpoints**, using LLM-based reasoning with contrastive patterns such as:

- “Some argue X, while others believe Y.”
- “On one hand A, on the other hand B.”
- “Proponents support X, whereas opponents claim Y.”

**Outputs:**
- `balanced_ratio` — proportion of two-sided sentences  
- `num_balanced_sentences` — count  
- `balanced_sentences` — extracted examples  

---

#### **2. Narrative Roles Extraction**
Classifies entities into narrative roles:

- **Hero** — protagonist / positive agent  
- **Villain** — antagonist / cause of harm  
- **Victim** — harmed or disadvantaged groups  

**Uses LLM-based detection** (requires OpenAI API)

### Utilities (`utils.py`)
- **API Management** - Environment variables + explicit keys
- **Save/Load** - Pickle serialization

---
 
#### **3. Interactivity (Stakeholder's Claim)**
Detects sentences that **attribute claims to specific sources** — i.e. quotes, citations, or reported speech — rather than presenting facts directly.
 
Examples of interactive sentences:
- *"According to the WHO, the virus spreads through droplets."*
- *"The CEO stated that layoffs were unavoidable."*
- *"Critics argue the policy will increase inequality."*
**Uses LLM-based detection** (requires OpenAI API)
 
**Outputs:**
- `interactive` — binary flag (1 if any interactive sentence found)  
- `interactive_ratio` — proportion of sentences with source attribution  
- `num_interactive_sentences` — count of attributed sentences  
- `total_citations` — total citation phrases extracted  
- `interaction_cues` — list of all citation phrases found  
- `sentence_breakdown` — per-sentence detail with extracted citation phrases  
---
## 💻 Usage Examples

### Example 1: Full Analysis

```python
from ctrllm import TextMetrics

metrics = TextMetrics(lang='en')
results = metrics.compute_all(text)

# Access specific metrics
print(results['shannon_entropy']['entropy'])
print(results['embedding_variance'])
print(results['entity_specificity'])
```

### Example 2: Individual Analyzers

```python
from ctrllm import SyntacticAnalyzer, SemanticAnalyzer, ArgumentAnalyzer

# Use individual classes
syntactic = SyntacticAnalyzer(lang='en')
syntactic.process_text(text)
entropy = syntactic.shannon_entropy()

semantic = SemanticAnalyzer()
semantic.set_sentences(sentences)
variance = semantic.embedding_variance()

# Argument analysis (LLM-based, requires OPENAI_API_KEY)
argument = ArgumentAnalyzer(
    lang='en',
    llm_model='gpt-4o-mini'
)
argument.process_text(text)
arg_results = argument.get_all_metrics(batch_delay=0.05)

print(f"Main ratio: {arg_results['main_vs_fringe']['main_ratio']:.3f}")
print(f"Diversity: {arg_results['argument_diversity']['arg_diversity']:.3f}")
```

---

## 📝 Running Demo

```bash
python demo.py
```

---

## 🎓 Use Cases

1. **Compare information sources** (Wikipedia vs. Britannica vs. LLM)
2. **Detect bias** in controversial topic coverage
3. **Measure perspective diversity** in content
4. **Assess text quality** and complexity
5. **Academic research** on LLM representations

---

## 📚 Dependencies

```
stanza                  # NLP processing
sentence-transformers   # Sentence embeddings
transformers           # Sentiment analysis
torch                  # PyTorch backend
numpy                  # Numerical operations
scikit-learn          # ML utilities
```

---

## 🔧 API Reference

### Main Class: `TextMetrics`

```python
TextMetrics(lang='en', embedding_model='...', sentiment_model='...')

# Methods
.compute_all(text, **options)      # All metrics
.compute_syntactic(text)            # Syntactic only
.compute_semantic(text)             # Semantic only
.compute_entity(text)               # Entity only
.compute_sentiment(text)            # Sentiment only
```

### Individual Analyzers

See source code docstrings for detailed API documentation.

---

## 📄 License

MIT License - Free for academic and commercial use

---


