# Project 5 — Boolean Networks for ER+ Breast Cancer Drug Response

Simulates a Boolean signaling network (ER+/PI3K/AKT/MAPK) under drug perturbations and searches for drug combinations that maximize:
**V = (#Apoptosis ON) − (#Proliferation ON)**

## Data (download separately)
- `breast_cancer.txt` (Boolean rules)

Place it here:
- `project-05-boolean-network/breast_cancer.txt`

> The dataset is not committed (ignored).

## Run
From this folder:
```bash
cd project-05-boolean-network
python project5.py -h
python project5.py --file "C:\path\to\breast_cancer.txt" --runs 1000 --mode synchronous -v final
