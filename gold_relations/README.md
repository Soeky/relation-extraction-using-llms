# Gold Relations

This directory contains the ground truth entity and relation annotations from BioRED.

## Generation Instructions

Generate gold relation files from the BioRED dataset:

```bash
python setup_scripts/generate_gold_relations.py
```

**Note**: You must first download the BioRED dataset:

```bash
python setup_scripts/download_biored.py
```

## Directory Structure

```
gold_relations/
├── train/
│   ├── 10000001.json
│   ├── 10000002.json
│   └── ...
├── dev/
│   └── ...
├── test/
│   └── ...
└── README.md
```

## JSON Format

Each `.json` file contains:

```json
{
  "doc_id": "10000001",
  "title": "Document title...",
  "body": "Document body/abstract...",
  "entities": [
    {
      "id": "MESH:D001943",
      "type": "DiseaseOrPhenotypicFeature",
      "mentions": [
        {
          "text": "breast cancer",
          "passage_index": 1,
          "passage_offset": 0,
          "char_offset": 45,
          "length": 13
        }
      ]
    }
  ],
  "relations": [
    {
      "id": "R1",
      "head_id": "NCBI:672",
      "tail_id": "MESH:D001943",
      "type": "Association",
      "novel": "No"
    }
  ]
}
```

## Entity Types

- `GeneOrGeneProduct` - Genes and proteins
- `DiseaseOrPhenotypicFeature` - Diseases and phenotypes
- `ChemicalEntity` - Chemicals and drugs
- `SequenceVariant` - Genetic variants and mutations
- `OrganismTaxon` - Species
- `CellLine` - Cell lines

## Relation Types

| Type | Description |
|------|-------------|
| Association | General association between entities |
| Positive_Correlation | Positive correlation or causation |
| Negative_Correlation | Negative correlation or inhibition |
| Bind | Physical binding interaction |
| Cotreatment | Combined treatment |
| Comparison | Comparative relationship |
| Drug_Interaction | Drug-drug interaction |
| Conversion | Biochemical conversion |

## Usage

These files serve as the ground truth for evaluating the relation extraction pipeline.
