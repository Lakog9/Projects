# Project 2 — Drosophila Disease Model Analysis (FlyBase + MONDO)

## Summary
- Parse FlyBase disease model annotations
- Map DOID → MONDO
- Extract MONDO level-3 categories
- Build contingency tables per (category, DO qualifier)
- Fisher exact tests (+ optional BH-FDR)

## Data (download separately)
- FlyBase: disease_model_annotations_fb_2025_02.tsv.gz
- MONDO: mondo.json

## Run
python project2.py
