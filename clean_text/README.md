# Clean Text Files

This directory contains plain text versions of the BioRED documents.

## Generation Instructions

Generate clean text files from the BioRED dataset:

```bash
python setup_scripts/generate_clean_text.py
```

**Note**: You must first download the BioRED dataset:

```bash
python setup_scripts/download_biored.py
```

## Directory Structure

```
clean_text/
├── train/
│   ├── 10000001.txt
│   ├── 10000002.txt
│   └── ...
├── dev/
│   └── ...
├── test/
│   └── ...
└── README.md
```

## File Format

Each `.txt` file contains:

```
<Document Title>

<Document Body / Abstract>
```

For example:

```
BRCA1 mutations and breast cancer risk.

Mutations in the BRCA1 gene are associated with an increased risk
of breast and ovarian cancer. This study examines the relationship
between specific BRCA1 variants and cancer outcomes...
```

## Usage

These files serve as input documents for the relation extraction pipeline.
The LLM reads these plain text files and extracts biomedical relations.
