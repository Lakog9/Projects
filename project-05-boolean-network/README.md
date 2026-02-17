# Project 5 — Boolean Network Simulation (ER+ Breast Cancer Drug Response)

Simulates an ER+ breast cancer signaling Boolean network under drug perturbations (single and combinations).
Runs until reaching an attractor (repeated state) and scores the outcome:

**V = (# Apoptosis nodes ON) − (# Proliferation nodes ON)**

Higher V is better.

## Model file (not included)
Download `breast_cancer.txt` from:
https://github.com/CASCI-lab/CANA/blob/master/cana/datasets/breast_cancer.txt

Keep it local (do not commit).

## Run
Example:

```bash
python project5.py --file "C:\path\to\breast_cancer.txt" --runs 1000 --N 1
