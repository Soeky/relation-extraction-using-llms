# Biomedical Relation Extraction using LLMs

A pipeline for extracting biomedical relations from scientific text using Large Language Models (LLMs).

## Overview

This project implements a comprehensive pipeline for biomedical relation extraction that:

- Supports multiple prompting techniques (IO, CoT, RAG, ReAct)
- Works with various LLM providers via OpenRouter
- Evaluates extraction quality using multiple matching strategies
- Generates comparative analysis charts

## Features

- **Multiple Prompting Techniques**:
  - **IO (Input-Output)**: Direct extraction prompting
  - **CoT (Chain-of-Thought)**: Step-by-step reasoning
  - **RAG (Retrieval-Augmented Generation)**: Context-enhanced extraction
  - **ReAct**: Structured reasoning with OBSERVE-THINK-ACT-REFLECT-EXTRACT workflow

- **Prompt Complexity Levels**:
  - **Baseline**: Minimal prompting
  - **Improved**: Enhanced instructions with examples
  - **Full**: Comprehensive prompting with detailed guidelines

- **Evaluation Metrics**:
  - Precision, Recall, F1 Score
  - Exact match rate
  - Graph edit distance
  - Multiple entity matching strategies

## Installation

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Setup

1. Clone the repository:
```bash
git clone git@github.com:Soeky/relation-extraction-using-llms.git
cd relation-extraction-using-llms
```

2. Install dependencies:
```bash
uv sync
# or
pip install -e .
```

3. Create environment file:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Data Setup

### Download BioRED Dataset

```bash
python setup_scripts/download_biored.py
```

### Generate Clean Text Files

```bash
python setup_scripts/generate_clean_text.py
```

### Generate Gold Relations

```bash
python setup_scripts/generate_gold_relations.py
```

## Usage

### Basic Usage

Run extraction on the test split with default settings:

```python
from main import main

main(
    split="test",
    max_documents=10,  # Process 10 documents
    techniques=["IO", "CoT"],
    matching_strategies=["exact"],
)
```

### Command Line

```bash
python main.py
```

### Configuration

Edit `config.py` or use environment variables:

```python
# config.py settings
TEMPERATURE = 0.1          # LLM temperature
MAX_TOKENS = 4096          # Max output tokens
LOG_LEVEL = "INFO"         # Logging verbosity
```

### Available Techniques

```python
techniques = [
    "Baseline-IO", "Improved-IO", "IO",
    "Baseline-CoT", "Improved-CoT", "CoT",
    "Baseline-RAG", "Improved-RAG", "RAG",
    "Baseline-ReAct", "Improved-ReAct", "ReAct",
]
```

### Available Models

Configure models via OpenRouter:

```python
models = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "anthropic/claude-sonnet-4-20250514",
    "google/gemini-2.0-flash-001",
    "meta-llama/llama-3.3-70b-instruct",
    # ... more models
]
```

## Project Structure

```
relation-extraction-using-llms/
├── README.md                          # This file
├── .env.example                       # Environment variables template
├── .gitignore
├── pyproject.toml                     # Project configuration
├── config.py                          # Pipeline configuration
├── main.py                            # Main entry point
│
├── data/                              # BioRED dataset
│   └── README.md
│
├── clean_text/                        # Plain text documents
│   ├── train/
│   ├── dev/
│   ├── test/
│   └── README.md
│
├── gold_relations/                    # Ground truth annotations
│   ├── train/
│   ├── dev/
│   ├── test/
│   └── README.md
│
├── pipeline/                          # Core pipeline modules
│   ├── __init__.py
│   ├── types.py                       # Type definitions
│   ├── prompts.py                     # Prompt templates
│   ├── data/                          # Data loading
│   ├── llm_prompter/                  # Prompting techniques
│   ├── parsing/                       # Response parsing
│   ├── evaluation/                    # Evaluation metrics
│   ├── aggregation/                   # Result aggregation
│   ├── retrieval/                     # RAG retrieval
│   ├── cache/                         # LLM caching
│   └── visualization/                 # Chart generation
│
├── utils/                             # Utility functions
│   ├── __init__.py
│   ├── io.py
│   └── logging.py
│
├── setup_scripts/                     # Data preparation
│   ├── download_biored.py
│   ├── generate_clean_text.py
│   └── generate_gold_relations.py
│
└── rag_sources/                       # RAG document sources
    └── README.md
```

## Prompting Techniques

### IO (Input-Output)

Direct prompting that provides the text and asks for relation extraction:

```
Extract biomedical relations from the following text.
Text: [document]
Output JSON array of relations.
```

### CoT (Chain-of-Thought)

Step-by-step reasoning approach:

1. Identify all entities
2. Classify entity types
3. Identify relations between entities
4. Verify relations
5. Output final JSON

### RAG (Retrieval-Augmented Generation)

Enhances prompts with retrieved context:

1. Retrieve relevant documents from knowledge base
2. Include context in prompt
3. Extract relations with enhanced background

### ReAct (Reasoning and Acting)

Structured reasoning framework:

- **OBSERVE**: Identify all biomedical entities
- **THINK**: Consider relationships between entities
- **ACT**: Determine relation types
- **REFLECT**: Verify findings
- **EXTRACT**: Provide final JSON output

## Evaluation

### Relation Types

The pipeline extracts 8 relation types:

| Type | Description |
|------|-------------|
| Association | General association |
| Positive_Correlation | Positive correlation/causation |
| Negative_Correlation | Negative correlation/inhibition |
| Bind | Physical binding |
| Cotreatment | Combined treatment |
| Comparison | Comparative relationship |
| Drug_Interaction | Drug-drug interaction |
| Conversion | Biochemical conversion |

### Matching Strategies

- **exact**: Exact string matching
- **fuzzy**: Fuzzy string matching
- **jaccard**: Jaccard similarity
- **levenshtein**: Levenshtein distance
- **jaro_winkler**: Jaro-Winkler similarity
- **token**: Token-based matching
- **sbert**: Sentence-BERT embeddings
- **bertscore**: BERTScore matching

## BioRED Dataset

This project uses the BioRED dataset for evaluation.

**Link**: https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/

### Citation

```bibtex
@article{luo2022biored,
  title={BioRED: A Comprehensive Biomedical Relation Extraction Dataset},
  author={Luo, Ling and Lai, Po-Ting and Wei, Chih-Hsuan and Arighi, Cecilia N and Lu, Zhiyong},
  journal={Briefings in Bioinformatics},
  year={2022},
  publisher={Oxford University Press}
}
```

**PubMed**: https://pubmed.ncbi.nlm.nih.gov/36039520/

## License

MIT License
