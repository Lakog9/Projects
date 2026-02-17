# Project 6 — Polypharmacy Side-Effect Association Network (Decagon/SNAP)

Streams the compressed CSV and (without networkx) finds:

**Among diseases whose drug-drug graph has exactly 1 connected component,
which disease has the largest component?**

Uses Union-Find (DSU) per disease while streaming.

## Data (download separately)
Download from SNAP:
- `ChChSe-Decagon_polypharmacy.csv.gz`

Do NOT commit it to the repo.

## Run
From this folder:
```bash
cd project-06-polypharmacy-network
python project6.py -h
python project6.py --csv_gz "C:\path\to\ChChSe-Decagon_polypharmacy.csv.gz"
