# Project 7 — Byte Pair Encoding (BPE) for DNA Tokenization

Implements BPE training and tokenization for DNA sequences (A,C,G,T) and evaluates tokenization efficiency across different vocabulary sizes (K) and training/test subsequence lengths (M).

## Data (download separately)
- hg38 chromosome 20 FASTA:
  - `chr20.fa.gz` from UCSC
- Preprocess:
  - remove header line (`>chr20`)
  - uppercase sequence
  - join lines into one continuous string

## Run
> Adjust CLI arguments to match what `project7.py -h` shows.

```bash
python project7.py -h
