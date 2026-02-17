
"""
PROJECT 5 – Boolean Network (ER+ breast cancer)

- Roots: κόμβοι χωρίς “γονείς”/κανόνα → αρχικοποιούνται τυχαία και μένουν σταθεροί.
- Drugs: όλα OFF εκτός από τα επιλεγμένα ON → μένουν σταθερά σε κάθε πείραμα.
- Result nodes (Apoptosis*, Proliferation*): αρχικά πάντα OFF.
- Run: επαναλαμβάνω ενημερώσεις μέχρι να ξαναεμφανιστεί ίδια κατάσταση. Τότε V = #ApoptosisON − #ProliferationON.
- Για κάθε συνδυασμό drugs κάνω 1000 runs και αναφέρω AVERAGE(V). (--seed για αναπαραγωγιμότητα)
"""

import re
import time
import random
import argparse
from itertools import combinations
from typing import Dict, List, Set, Tuple, Optional

# ----------------------------
# Constants (per assignment)
# ----------------------------
DRUGS = [
    "Fulvestrant",
    "Alpelisib",
    "Everolimus",
    "Trametinib",
    "Ipatasertib",
    "Palbociclib",
    "Neratinib",
]

APOP_PREFIX = "Apoptosis"
PROLIF_PREFIX = "Proliferation"
BOOL_TOKENS = {"and", "or", "not", "True", "False"}

MAX_STEPS_SYNC = 20000
MAX_STEPS_ASYNC = 50000

DRUG_CANON = {d.casefold(): d for d in DRUGS}


# ----------------------------
# Utilities
# ----------------------------
def format_seconds(sec: float) -> str:
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def strip_outer_parens(s: str) -> str:
    s = s.strip()
    while len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        inner = s[1:-1].strip()
        if inner.count("(") != inner.count(")"):
            break
        s = inner
    return s


def is_identity_rule(node: str, expr: str) -> bool:
    return strip_outer_parens(expr) == node


def canonicalize_name(name: str) -> str:
    """Canonicalize known drug node names case-insensitively."""
    return DRUG_CANON.get(name.casefold(), name)


def canonicalize_expr(expr: str) -> str:
    """Canonicalize drug tokens inside expressions case-insensitively."""
    out = expr
    for canon in DRUGS:
        # word-boundary safe: won't touch e.g. Ipatasertib_2
        out = re.sub(rf"\b{re.escape(canon)}\b", canon, out, flags=re.IGNORECASE)
    return out


def progress_line(prefix: str, done: int, total: int, elapsed: float, eta: float) -> str:
    pct = (done / total) * 100 if total else 100.0
    return f"{prefix} {done}/{total} ({pct:5.1f}%) | elapsed {format_seconds(elapsed)} | eta {format_seconds(eta)}"


# ----------------------------
# Task 1: Parse rules (supports Node* = expr and Node: expr + continuations)
# ----------------------------
def load_rules(filename: str) -> Dict[str, str]:
    start_re = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(?:\*\s*=\s*|:\s*)(.*)$")

    rules: Dict[str, str] = {}
    current: Optional[str] = None
    buf: List[str] = []
    current_raw_name: Optional[str] = None

    with open(filename, "r", encoding="utf-8-sig", errors="replace") as f:
        for raw in f:
            raw = raw.split("#", 1)[0]
            line = raw.strip()
            if not line:
                continue
            if line.upper().startswith("PROJECT"):
                continue

            m = start_re.match(line)
            if m:
                if current is not None:
                    expr = " ".join(buf).strip()
                    expr = canonicalize_expr(expr)
                    rules[current] = expr

                current_raw_name = m.group(1)
                current = canonicalize_name(current_raw_name)
                buf = [m.group(2)]
            else:
                if current is None:
                    continue
                buf.append(line)

    if current is not None:
        expr = " ".join(buf).strip()
        expr = canonicalize_expr(expr)
        rules[current] = expr

    if not rules:
        raise ValueError("Parsed 0 rules. Expected lines like `Node* = expr` or `Node: expr`.")
    return rules


# ----------------------------
# Task 2: Build graph info
# ----------------------------
def extract_identifiers(expr: str) -> Set[str]:
    tokens = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expr))
    return {t for t in tokens if t not in BOOL_TOKENS}


def build_network(rules: Dict[str, str]) -> Tuple[List[str], Dict[str, Set[str]], Set[str], Set[str]]:
    parents_map: Dict[str, Set[str]] = {}
    all_nodes: Set[str] = set(rules.keys())

    for node, expr in rules.items():
        deps = extract_identifiers(expr)
        all_nodes |= deps

        # identity nodes treated as having NO parents (root/input)
        if is_identity_rule(node, expr):
            parents_map[node] = set()
        else:
            parents_map[node] = deps

    nodes_without_rule = {n for n in all_nodes if n not in rules}
    nodes_with_no_parents = {n for n, ps in parents_map.items() if len(ps) == 0}

    roots = nodes_without_rule | nodes_with_no_parents
    drug_nodes_present = {d for d in DRUGS if d in all_nodes}

    return sorted(all_nodes), parents_map, roots, drug_nodes_present


def get_result_nodes(all_nodes: List[str]) -> Tuple[List[str], List[str]]:
    apoptosis_nodes = [n for n in all_nodes if n.startswith(APOP_PREFIX)]
    proliferation_nodes = [n for n in all_nodes if n.startswith(PROLIF_PREFIX)]
    return apoptosis_nodes, proliferation_nodes


def warn_case_duplicates(all_nodes: List[str]) -> None:
    by_fold: Dict[str, List[str]] = {}
    for n in all_nodes:
        by_fold.setdefault(n.casefold(), []).append(n)
    dups = {k: v for k, v in by_fold.items() if len(v) > 1}
    if dups:
        print("\nWARNING: Case-duplicate node names detected (this WILL change results):")
        for k, v in sorted(dups.items()):
            print(f"  {k}: {v}")
        print("Tip: this is usually caused by inconsistent capitalization in the rules (e.g. ipatasertib vs Ipatasertib).\n")


# ----------------------------
# Task 3: compiled rule evaluation
# ----------------------------
def compile_rules(rules: Dict[str, str]) -> Dict[str, object]:
    return {node: compile(expr, f"<rule:{node}>", "eval") for node, expr in rules.items()}


def eval_rule(compiled_expr: object, state: Dict[str, bool]) -> bool:
    return bool(eval(compiled_expr, {"__builtins__": {}}, state))


def get_next_state(
    current_state: Dict[str, bool],
    compiled_rules: Dict[str, object],
    update_nodes: List[str],
    mode: str,
) -> Dict[str, bool]:
    nxt = current_state.copy()
    if mode == "synchronous":
        for node in update_nodes:
            nxt[node] = eval_rule(compiled_rules[node], current_state)
    elif mode == "asynchronous":
        if update_nodes:
            node = random.choice(update_nodes)
            nxt[node] = eval_rule(compiled_rules[node], current_state)
    else:
        raise ValueError("mode must be 'synchronous' or 'asynchronous'")
    return nxt


# ----------------------------
# Tasks 3-4: init, run until repeat, compute V
# ----------------------------
def compute_V(state: Dict[str, bool], apoptosis_nodes: List[str], proliferation_nodes: List[str]) -> int:
    a = sum(1 for n in apoptosis_nodes if state.get(n, False))
    p = sum(1 for n in proliferation_nodes if state.get(n, False))
    return a - p


def initialize_state(
    all_nodes: List[str],
    roots: Set[str],
    drug_nodes_present: Set[str],
    active_drugs: Set[str],
    apoptosis_nodes: List[str],
    proliferation_nodes: List[str],
) -> Tuple[Dict[str, bool], Dict[str, bool], Dict[str, bool]]:
    state: Dict[str, bool] = {n: random.choice([True, False]) for n in all_nodes}

    # Results OFF initially
    for n in apoptosis_nodes:
        state[n] = False
    for n in proliferation_nodes:
        state[n] = False

    # Drugs fixed for this experiment
    for d in drug_nodes_present:
        state[d] = (d in active_drugs)

    roots_only = set(roots) - set(drug_nodes_present)
    fixed_roots = {r: state[r] for r in roots_only}
    fixed_drugs = {d: state[d] for d in drug_nodes_present}
    return state, fixed_roots, fixed_drugs


def clamp_fixed(state: Dict[str, bool], fixed_roots: Dict[str, bool], fixed_drugs: Dict[str, bool]) -> None:
    for k, v in fixed_roots.items():
        state[k] = v
    for k, v in fixed_drugs.items():
        state[k] = v


def state_signature(state: Dict[str, bool], node_order: List[str]) -> Tuple[bool, ...]:
    return tuple(state[n] for n in node_order)


def run_until_repeat(
    all_nodes: List[str],
    roots: Set[str],
    drug_nodes_present: Set[str],
    compiled_rules: Dict[str, object],
    active_drugs: Set[str],
    mode: str,
    v_mode: str,  # final or attractor_mean
) -> float:
    apoptosis_nodes, proliferation_nodes = get_result_nodes(all_nodes)

    state, fixed_roots, fixed_drugs = initialize_state(
        all_nodes, roots, drug_nodes_present, active_drugs, apoptosis_nodes, proliferation_nodes
    )

    frozen = set(fixed_roots.keys()) | set(fixed_drugs.keys())
    update_nodes = [n for n in compiled_rules.keys() if n not in frozen]

    node_order = all_nodes
    max_steps = MAX_STEPS_SYNC if mode == "synchronous" else MAX_STEPS_ASYNC

    seen_index: Dict[Tuple[bool, ...], int] = {}
    V_series: List[int] = []

    for step in range(max_steps):
        sig = state_signature(state, node_order)

        if sig in seen_index:
            cycle_start = seen_index[sig]
            if v_mode == "final":
                return float(compute_V(state, apoptosis_nodes, proliferation_nodes))
            elif v_mode == "attractor_mean":
                cycle_Vs = V_series[cycle_start:]
                return sum(cycle_Vs) / len(cycle_Vs)
            else:
                raise ValueError("v_mode must be 'final' or 'attractor_mean'")

        seen_index[sig] = step
        V_series.append(compute_V(state, apoptosis_nodes, proliferation_nodes))

        nxt = get_next_state(state, compiled_rules, update_nodes, mode)
        clamp_fixed(nxt, fixed_roots, fixed_drugs)
        state = nxt

    raise RuntimeError(f"Exceeded max_steps={max_steps} without reaching a repeated state.")


# ----------------------------
# Tasks 5-7
# ----------------------------
def average_V(
    all_nodes: List[str],
    roots: Set[str],
    drug_nodes_present: Set[str],
    compiled_rules: Dict[str, object],
    combo: Tuple[str, ...],
    runs: int,
    mode: str,
    v_mode: str,
) -> float:
    total = 0.0
    active = set(combo)
    for _ in range(runs):
        total += run_until_repeat(all_nodes, roots, drug_nodes_present, compiled_rules, active, mode, v_mode)
    return total / runs


def main():
    parser = argparse.ArgumentParser(description="Project 5 Boolean Network simulation (Tasks 1-7 + BONUS)")
    parser.add_argument("--file", default="breast_cancer.txt", help="Model file path")
    parser.add_argument("--runs", type=int, default=1000, help="Runs per combination (Task 5 requires 1000)")
    parser.add_argument("--mode", choices=["synchronous", "asynchronous"], default="synchronous", help="Update mode (BONUS)")
    parser.add_argument("--v", choices=["final", "attractor_mean"], default="final", help="V mode: final or attractor_mean")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (use -1 to disable)")
    parser.add_argument("--print-rules", action="store_true", help="Print all parsed rules (as in sample screenshots)")
    parser.add_argument("--print-all-task7", action="store_true", help="Print ALL combinations for Task 7 (like sample)")
    args = parser.parse_args()

    if args.seed != -1:
        random.seed(args.seed)

    rules = load_rules(args.file)
    all_nodes, parents_map, roots, drug_nodes_present = build_network(rules)
    compiled_rules = compile_rules(rules)

    apoptosis_nodes, proliferation_nodes = get_result_nodes(all_nodes)

    print("PROJECT 5\n")
    print(f"Parsed rules: {len(rules)}")
    print(f"Total nodes: {len(all_nodes)}")
    print(f"Drug nodes present: {sorted(drug_nodes_present)}")
    print(f"Roots detected (no parents / no rule): {len(roots)}\n")

    warn_case_duplicates(all_nodes)

    if args.print_rules:
        for k in rules:
            print(f"{k}: {rules[k]}")
        print()

    # -------- Task 5 --------
    print("TASK 5:")
    best1 = (None, float("-inf"))
    for d in DRUGS:
        avg = average_V(all_nodes, roots, drug_nodes_present, compiled_rules, (d,), args.runs, args.mode, args.v)
        print(f"Drug: {d:<12} -> Average V: {avg:.4f}")
        if avg > best1[1]:
            best1 = ((d,), avg)
    print(f"Best drug: {best1[0][0]} with Max Average V = {best1[1]:.4f}\n")

    # -------- Task 6 (N=2) --------
    print("TASK 6 (N=2):")
    best2 = (None, float("-inf"))
    for combo in combinations(DRUGS, 2):
        avg = average_V(all_nodes, roots, drug_nodes_present, compiled_rules, combo, args.runs, args.mode, args.v)
        print(f"Drug: {' + '.join(combo)} -> Average V: {avg:.4f}")
        if avg > best2[1]:
            best2 = (combo, avg)
    print(f"Best drug 2 : {' + '.join(best2[0])} with Max Average V = {best2[1]:.4f}\n")

    # -------- Task 7 (N=1..7) --------
    print("TASK 7:")
    print("TASK 7: Optimal Therapy (N=1 to 7)")

    overall_best = (None, float("-inf"))

    for n in range(1, 8):
        combos = list(combinations(DRUGS, n))
        best_n = (None, float("-inf"))

        start = time.time()
        for i, combo in enumerate(combos, 1):
            avg = average_V(all_nodes, roots, drug_nodes_present, compiled_rules, combo, args.runs, args.mode, args.v)

            if args.print_all_task7:
                print(f"Drug: {' + '.join(combo):<60} -> Average V: {avg:.4f}")

            if avg > best_n[1]:
                best_n = (combo, avg)

            # progress only if not printing every combo
            if (not args.print_all_task7) and (i % max(1, len(combos) // 10) == 0 or i == len(combos)):
                elapsed = time.time() - start
                eta = (elapsed / i) * (len(combos) - i)
                print(progress_line(f"  N={n} progress:", i, len(combos), elapsed, eta))

        # report best per N
        if not args.print_all_task7:
            print(f"Best for N={n}: {' + '.join(best_n[0])} -> Max Average V = {best_n[1]:.4f}")

        if best_n[1] > overall_best[1]:
            overall_best = best_n

    print(f"\nFINAL DRUG: {' + '.join(overall_best[0])}--> Max Average V = {overall_best[1]:.4f}")

    # -------- BONUS quick check --------
    print("\nBONUS (500 runs check on FINAL DRUG):")
    for m in ("synchronous", "asynchronous"):
        avg = average_V(all_nodes, roots, drug_nodes_present, compiled_rules, overall_best[0], 500, m, args.v)
        print(f"Mode: {m:<12} -> Average V: {avg:.4f}")


if __name__ == "__main__":
    main()
