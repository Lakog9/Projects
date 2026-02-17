# Project 4 — Metagenome Complexity via k-mer Shannon Entropy

Streams a compressed FASTQ (`.fq.gz`), counts k-mers (skipping `N`),
computes Shannon entropy until convergence, and plots unique k-mers vs reads.

## Data
Download one `_clean.1.fq.gz` from the specified NCBI study (not included in repo).

## Run
python project4.py <file.fq.gz>        # multi-k (4..7)
python project4.py <file.fq.gz> 6      # single k
