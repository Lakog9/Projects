RUN INSTRUCTIONS

1) Download the DDI-Bench repository (ZIP) from:
https://github.com/LARS-research/DDI-Bench

2) Extract it.

3) Put implementation.py in the extracted folder that contains the folder DDI_Ben.
The folder must look like this:

DDI-Bench-main/
  DDI_Ben/
  Real_Scene/
  implementation.py

4) Open a terminal inside DDI-Bench-main/ and install numpy:
pip install numpy

5) TRAIN MODE (evaluation):
python implementation.py --molecular_feats DDI_Ben/DDI_Ben/data/initial/drugbank/DB_molecular_feats.pkl --relation2id DDI_Ben/TextDDI/data/drugbank_random/relation2id.json --train DDI_Ben/DDI_Ben/data/drugbank_random/train.txt --test DDI_Ben/DDI_Ben/data/drugbank_random/test_S0.txt --cutoff 20 --mode train

6) INFERENCE MODE (prediction):
python implementation.py --molecular_feats DDI_Ben/DDI_Ben/data/initial/drugbank/DB_molecular_feats.pkl --relation2id DDI_Ben/TextDDI/data/drugbank_random/relation2id.json --train DDI_Ben/DDI_Ben/data/drugbank_random/train.txt --mode inference --drugbank_1 DB13231 --drugbank_2 DB00244
