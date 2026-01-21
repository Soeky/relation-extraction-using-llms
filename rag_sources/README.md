# RAG Sources

This directory contains source documents for Retrieval-Augmented Generation (RAG).

## Purpose

The RAG prompter uses these documents to provide additional context to the LLM
when extracting relations. This can help with:

- Entity disambiguation
- Relation type identification
- Background knowledge about biomedical concepts

## Usage

To use RAG prompting:

1. Populate this directory with relevant biomedical documents (optional)
2. The RAG prompter will automatically index and retrieve relevant passages

## Supported Formats

- Plain text files (`.txt`)
- JSON files with `text` field

## Directory Structure

```
rag_sources/
├── pubmed_abstracts/     # PubMed abstracts (optional)
├── knowledge_base/       # Domain knowledge (optional)
└── README.md
```

## Note

RAG functionality is optional. The pipeline works without RAG sources,
but performance may improve with relevant background documents.
