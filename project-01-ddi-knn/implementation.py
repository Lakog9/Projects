import argparse
import json
import pickle
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np


# -----------------------------
# Data structures
# -----------------------------
@dataclass(frozen=True)
class DrugFeats:
    morgan: np.ndarray          # 0/1 vector
    morgan_sum: int             # number of 1s
    rd2d: np.ndarray            # float vector
    rd2d_norm: float            # L2 norm


# -----------------------------
# IO helpers
# -----------------------------
def load_pairs(path: str) -> List[Tuple[int, int, int]]:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a, b, r = line.split()
            pairs.append((int(a), int(b), int(r)))
    return pairs


def load_relation_map(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)  # keys are strings


def load_molecular_feats(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


# -----------------------------
# Feature preparation
# -----------------------------
def build_mappings_and_feats(feats_dict) -> Tuple[Dict[int, DrugFeats], Dict[str, int]]:
    """
    Returns:
      nodeid_to_feats: node_id -> DrugFeats
      drugbank_to_nodeid: "DBxxxx" -> node_id
    """
    node_ids = feats_dict["Node ID"]
    drugbank_ids = feats_dict["DrugBank ID"]
    morgan_list = feats_dict["Morgan_Features"]
    rd2d_list = feats_dict["RDKit2D_Features"]

    nodeid_to_feats: Dict[int, DrugFeats] = {}
    drugbank_to_nodeid: Dict[str, int] = {}

    for i in range(len(node_ids)):
        node_id = int(node_ids[i])
        db_id = str(drugbank_ids[i])

        # Morgan: force to 0/1 int array for fast dot product
        morgan = np.asarray(morgan_list[i], dtype=np.uint8)
        # Safety: ensure binary (some datasets store as bool already)
        if morgan.max() > 1:
            morgan = (morgan > 0).astype(np.uint8)
        morgan_sum = int(morgan.sum())

        # RDKit2D: float array
        rd2d = np.asarray(rd2d_list[i], dtype=np.float32)
        rd2d_norm = float(np.linalg.norm(rd2d))

        nodeid_to_feats[node_id] = DrugFeats(
            morgan=morgan,
            morgan_sum=morgan_sum,
            rd2d=rd2d,
            rd2d_norm=rd2d_norm,
        )
        drugbank_to_nodeid[db_id] = node_id

    return nodeid_to_feats, drugbank_to_nodeid


# -----------------------------
# Distances
# -----------------------------
def tanimoto_distance(a: DrugFeats, b: DrugFeats) -> float:
    """
    Tanimoto/Jaccard distance for binary vectors:
      sim = |A∩B| / |A∪B|
      dist = 1 - sim
    Using dot for intersection since vectors are 0/1.
    """
    inter = int(np.dot(a.morgan, b.morgan))
    union = a.morgan_sum + b.morgan_sum - inter
    if union == 0:
        return 0.0  # identical "all-zero" case
    sim = inter / union
    return 1.0 - sim


def cosine_distance(a: DrugFeats, b: DrugFeats) -> float:
    """
    Cosine distance:
      dist = 1 - (a·b / (||a|| ||b||))
    """
    denom = a.rd2d_norm * b.rd2d_norm
    if denom == 0.0:
        return 1.0
    sim = float(np.dot(a.rd2d, b.rd2d) / denom)
    # numeric safety
    if sim > 1.0:
        sim = 1.0
    elif sim < -1.0:
        sim = -1.0
    return 1.0 - sim


def pair_distance(
    test_a: DrugFeats,
    test_b: DrugFeats,
    train_a: DrugFeats,
    train_b: DrugFeats,
    mf_weight: float,
    rd_weight: float,
) -> float:
    mf = (tanimoto_distance(test_a, train_a) + tanimoto_distance(test_b, train_b)) / 2.0
    rd = (cosine_distance(test_a, train_a) + cosine_distance(test_b, train_b)) / 2.0
    return mf_weight * mf + rd_weight * rd


# -----------------------------
# k-NN prediction
# -----------------------------
def predict_knn(
    test_pair: Tuple[int, int],
    train_pairs: List[Tuple[int, int, int]],
    nodeid_to_feats: Dict[int, DrugFeats],
    k: int,
    mf_weight: float,
    rd_weight: float,
) -> int:
    ta, tb = test_pair
    test_a = nodeid_to_feats[ta]
    test_b = nodeid_to_feats[tb]

    if k <= 1:
        best_d = float("inf")
        best_label = None
        for a, b, lab in train_pairs:
            d = pair_distance(test_a, test_b, nodeid_to_feats[a], nodeid_to_feats[b], mf_weight, rd_weight)
            if d < best_d:
                best_d = d
                best_label = lab
        return int(best_label)

    # k > 1: keep k best
    best: List[Tuple[float, int]] = []  # (distance, label)

    for a, b, lab in train_pairs:
        d = pair_distance(test_a, test_b, nodeid_to_feats[a], nodeid_to_feats[b], mf_weight, rd_weight)
        if len(best) < k:
            best.append((d, lab))
            if len(best) == k:
                best.sort(key=lambda x: x[0])
        else:
            # best is sorted by distance; last is worst
            if d < best[-1][0]:
                best[-1] = (d, lab)
                best.sort(key=lambda x: x[0])

    labels = [lab for _, lab in best]
    counts = Counter(labels)
    max_count = max(counts.values())
    tied = [lab for lab, c in counts.items() if c == max_count]
    if len(tied) == 1:
        return int(tied[0])

    # tie-breaker: choose the tied label with smallest distance among best
    for d, lab in best:
        if lab in tied:
            return int(lab)

    # fallback
    return int(best[0][1])


# -----------------------------
# Modes
# -----------------------------
def run_train(args):
    rel_map = load_relation_map(args.relation2id)
    feats_dict = load_molecular_feats(args.molecular_feats)
    nodeid_to_feats, _ = build_mappings_and_feats(feats_dict)

    train_pairs = load_pairs(args.train)
    test_pairs = load_pairs(args.test)

    total = len(test_pairs) if args.cutoff is None else min(args.cutoff, len(test_pairs))
    correct = 0

    for i in range(total):
        a, b, real = test_pairs[i]
        pred = predict_knn(
            (a, b),
            train_pairs,
            nodeid_to_feats,
            k=args.k,
            mf_weight=args.mf_weight,
            rd_weight=args.rd_weight,
        )
        if pred == real:
            correct += 1
        print(f"Sample: {i+1}/{len(test_pairs)}, Predicted: {pred}, Real: {real}")

    wrong = total - correct
    acc = correct / total if total > 0 else 0.0
    print(f"Total: {total}, correct: {correct}  wrong: {wrong}")
    print(f"Accuracy: {acc}")

    # optional: show example mapping for the last prediction
    if total > 0:
        print("\nExample predicted relation text:")
        print(rel_map.get(str(pred), f"(missing text for relation id {pred})"))


def run_inference(args):
    rel_map = load_relation_map(args.relation2id)
    feats_dict = load_molecular_feats(args.molecular_feats)
    nodeid_to_feats, drugbank_to_nodeid = build_mappings_and_feats(feats_dict)

    train_pairs = load_pairs(args.train)

    if args.drugbank_1 not in drugbank_to_nodeid:
        raise ValueError(f"Unknown DrugBank ID: {args.drugbank_1}")
    if args.drugbank_2 not in drugbank_to_nodeid:
        raise ValueError(f"Unknown DrugBank ID: {args.drugbank_2}")

    a_node = drugbank_to_nodeid[args.drugbank_1]
    b_node = drugbank_to_nodeid[args.drugbank_2]

    pred = predict_knn(
        (a_node, b_node),
        train_pairs,
        nodeid_to_feats,
        k=args.k,
        mf_weight=args.mf_weight,
        rd_weight=args.rd_weight,
    )

    text = rel_map.get(str(pred), f"(missing text for relation id {pred})")
    print("Prediction:", text)


def main():
    parser = argparse.ArgumentParser(description="k-NN DDI relation prediction (DrugBank)")
    parser.add_argument("--molecular_feats", required=True, help="Path to DB_molecular_feats.pkl")
    parser.add_argument("--relation2id", required=True, help="Path to relation2id.json")
    parser.add_argument("--train", required=True, help="Path to train.txt")
    parser.add_argument("--mode", required=True, choices=["train", "inference"], help="train or inference")

    parser.add_argument("--test", help="Path to test file (required for mode=train)")
    parser.add_argument("--cutoff", type=int, default=None, help="Max number of test samples to process")
    parser.add_argument("--k", type=int, default=1, help="k in k-NN (default 1)")
    parser.add_argument("--mf_weight", type=float, default=0.5, help="Weight for Morgan (Tanimoto) distance")
    parser.add_argument("--rd_weight", type=float, default=0.5, help="Weight for RDKit2D (cosine) distance")

    parser.add_argument("--drugbank_1", help="First DrugBank ID (required for inference)")
    parser.add_argument("--drugbank_2", help="Second DrugBank ID (required for inference)")

    args = parser.parse_args()

    if args.mode == "train":
        if not args.test:
            raise ValueError("--test is required when --mode train")
        run_train(args)
    else:
        if not args.drugbank_1 or not args.drugbank_2:
            raise ValueError("--drugbank_1 and --drugbank_2 are required when --mode inference")
        run_inference(args)


if __name__ == "__main__":
    main()
