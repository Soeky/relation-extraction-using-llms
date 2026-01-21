# BioRED Dataset

This directory should contain the BioRED (Biomedical Relation Extraction Dataset) files.

## Download Instructions

### Option 1: Use the setup script

```bash
python setup_scripts/download_biored.py
```

### Option 2: Manual download

1. Download `BIORED.zip` from the NCBI FTP server:
   - https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/BIORED.zip

2. Extract the JSON files from the ZIP archive and place them in this directory.

## Expected Files

After download, this directory should contain:

```
data/
├── Train.BioC.JSON
├── Dev.BioC.JSON
├── Test.BioC.JSON
└── README.md
```

## Dataset Information

BioRED is a comprehensive biomedical relation extraction dataset containing:

- **Entities**: Genes, diseases, chemicals, variants, species, cell lines
- **Relations**: 8 relation types (Association, Positive_Correlation, Negative_Correlation, Bind, Cotreatment, Comparison, Drug_Interaction, Conversion)
- **Novelty**: Relations are marked as "Novel" or "No" (previously known)

### Statistics

| Split | Documents | Relations |
|-------|-----------|-----------|
| Train | ~500      | ~5,000    |
| Dev   | ~100      | ~1,000    |
| Test  | ~100      | ~1,000    |

## Citation

If you use BioRED in your research, please cite:

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
