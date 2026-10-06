# Data Directory

Place raw EBA disclosure files here.

## Expected input for Exp 5 & 6

EBA 2025 EU-wide transparency exercise, reference date 30 June 2025
(released 4 December 2025), item 2520810: direct on-balance-sheet total
gross carrying amount of non-derivative financial assets to sovereign
counterparties, summed over maturities.

Download from: https://www.eba.europa.eu/risk-analysis-and-data/eu-wide-transparency-exercise

Filename convention used by `src/eba_loader.py`:
- `eba_2025_transparency_item2520810.csv`  (cleaned CSV from EBA XLSX)

If this file is absent, the loader falls back to:
- Exp 5: hard-coded six-bank Table 6 matrix (paper approximation)
- Exp 6: synthetic 116-institution matrix (seed=42)
