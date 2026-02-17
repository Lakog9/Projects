import json
import re
from collections import deque
from typing import Dict, List, Any, Set


MONDO_URL_RE = re.compile(r"^http://purl\.obolibrary\.org/obo/MONDO_\d+$")
DEFAULT_ROOT = "http://purl.obolibrary.org/obo/MONDO_0000001"  # disease


def part_1(mondo_json_path: str) -> Dict[str, Any]:
    """
    Read mondo.json and build a tree-like structure with:
      - labels: MONDO_URL -> label
      - children_of: parent_url -> [child_url, ...]
      - parents_of: child_url -> [parent_url, ...]
      - level: node_url -> level from root (root level = 1)
      - root: chosen root URL
    Uses only edges with pred == 'is_a' and MONDO URLs for sub/obj.
    """
    with open(mondo_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    g0 = data["graphs"][0]
    raw_nodes = g0.get("nodes", [])
    raw_edges = g0.get("edges", [])

    # labels
    labels: Dict[str, str] = {}
    for n in raw_nodes:
        node_id = n.get("id", "")
        if isinstance(node_id, str) and MONDO_URL_RE.match(node_id):
            lbl = n.get("lbl", "")
            if isinstance(lbl, str):
                labels[node_id] = lbl

    # edges (child -> parent) for is_a only
    children_of: Dict[str, List[str]] = {}
    parents_of: Dict[str, List[str]] = {}

    for e in raw_edges:
        if e.get("pred") != "is_a":
            continue
        sub = e.get("sub", "")
        obj = e.get("obj", "")
        if not (isinstance(sub, str) and isinstance(obj, str)):
            continue
        if MONDO_URL_RE.match(sub) and MONDO_URL_RE.match(obj):
            children_of.setdefault(obj, []).append(sub)   # parent -> child
            parents_of.setdefault(sub, []).append(obj)    # child -> parent(s)

    # choose root
    if DEFAULT_ROOT in labels:
        root = DEFAULT_ROOT
    else:
        all_nodes = set(labels.keys())
        roots = [n for n in all_nodes if n not in parents_of]
        root = roots[0] if roots else None

    # compute levels from root with BFS (root level = 1)
    level: Dict[str, int] = {}
    if root is not None:
        q = deque([root])
        level[root] = 1
        while q:
            cur = q.popleft()
            cur_level = level[cur]
            for child in children_of.get(cur, []):
                if child not in level:
                    level[child] = cur_level + 1
                    q.append(child)

    return {
        "labels": labels,
        "children_of": children_of,
        "parents_of": parents_of,
        "level": level,
        "root": root,
    }


def part_2(mondo_tree: Dict[str, Any], mondo_id_short: str, target_level: int = 3) -> List[str]:
    """
    Given the tree object from part_1 and a MONDO ID like 'MONDO_0011719',
    return labels of ancestors that are exactly at 'target_level' from the root.
    """
    start = f"http://purl.obolibrary.org/obo/{mondo_id_short}"

    labels: Dict[str, str] = mondo_tree["labels"]
    parents_of: Dict[str, List[str]] = mondo_tree["parents_of"]
    level: Dict[str, int] = mondo_tree["level"]

    if start not in labels:
        return []

    q = deque([start])
    visited: Set[str] = {start}

    found: Set[str] = set()

    while q:
        cur = q.popleft()
        for p in parents_of.get(cur, []):
            if p not in visited:
                visited.add(p)
                q.append(p)
            if level.get(p) == target_level:
                lbl = labels.get(p, "")
                if lbl:
                    found.add(lbl)

    # deterministic output
    return sorted(found)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mondo_json", required=True, help="Full path to mondo.json")
    parser.add_argument("--mondo_id", required=True, help="MONDO id like MONDO_0011719")
    parser.add_argument("--level", type=int, default=3, help="Target level (default 3)")
    args = parser.parse_args()

    mondo_tree = part_1(args.mondo_json)
    print(part_2(mondo_tree, args.mondo_id, target_level=args.level))


