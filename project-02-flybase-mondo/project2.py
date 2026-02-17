
"""
PROJECT 2 — Drosophila Disease Model Analysis (Task 1–6 + Bonus 1–2)

Αρχεία (στον ίδιο φάκελο):
  • disease_model_annotations_fb_2025_02.tsv.gz (FlyBase)
  • mondo.json (MONDO ontology)

Τρέξιμο:
  python project2.py
Εξαρτήσεις:
  pip install scipy statsmodels

Τι κάνει ο κώδικας (σύνοψη):
  1) Διαβάζει FlyBase και κρατά (FBgn, DO qualifier, DOID) + μοναδικά (qualifier, DOID, term).
  2) Χαρτογραφεί DOID → MONDO με STRICT xrefs (όχι free-text).
  3) Χτίζει το is_a γράφημα της MONDO και βρίσκει “Categories” ως ancestors στο level 3 από MONDO:0000001.
  4) Φτιάχνει gene-level presence/absence σύνολα για κάθε (Category, Qualifier).
  5) Τρέχει Fisher exact (two-sided) και υπολογίζει expected + fold-change (Bonus 1).
  6) Τυπώνει Top-10 μικρότερα p-values.
  Bonus 2: Κάνει BH-FDR (alpha=0.05) και τυπώνει όλα τα significant (sorted by padj).
Σημείωση output: οι πίνακες είναι pipe-separated χωρίς κενά για compact terminal.
"""


import gzip, json, re
from collections import defaultdict, deque

from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


# =========================
# CONFIG
# =========================
FLY_FILE = "disease_model_annotations_fb_2025_02.tsv.gz"
MONDO_FILE = "mondo.json"
TOPK, TARGET_LEVEL, ALPHA = 10, 3, 0.05
PRINT_UNMAPPED_DOIDS = True
SHOW_SIGNIFICANT_LIMIT = None  # None => show all


# =========================
# SMALL HELPERS
# =========================
DOID_RE = re.compile(r"^(?:DOID|DO)[:_](\d+)$")
MONDO_URI_RE = re.compile(r"^http://purl\.obolibrary\.org/obo/MONDO_\d+$")

def hr(ch="=", n=90): return ch * n
def sci(x, d=6): return f"{x:.{d}e}"
def f4(x): return f"{x:.4f}"
def odds_str(o): return "inf" if o == float("inf") else ("0" if o == 0.0 else f"{o:.6g}")
def mondo_curie(uri): return uri.rsplit("/", 1)[-1].replace("MONDO_", "MONDO:")

def norm_doid(s: str):
    m = DOID_RE.match((s or "").strip())
    return f"DOID:{m.group(1)}" if m else None

def is_is_a(pred: str) -> bool:
    return isinstance(pred, str) and pred and (
        pred == "is_a" or pred == "subClassOf" or pred == "rdfs:subClassOf" or pred.endswith("#subClassOf")
    )

def print_table(headers, rows, maxw=60):
    def t(x):
        s = "" if x is None else str(x)
        return (s[:maxw - 1] + "…") if len(s) > maxw else s
    H = [t(h) for h in headers]
    R = [[t(c) for c in r] for r in rows]
    w = [len(h) for h in H]
    for r in R:
        for i, c in enumerate(r):
            w[i] = max(w[i], len(c))
    print(" | ".join(H[i].ljust(w[i]) for i in range(len(w))))
    print("-+-".join("-" * w[i] for i in range(len(w))))
    for r in R:
        print(" | ".join(r[i].ljust(w[i]) for i in range(len(w))))

def xref_doids(meta: dict) -> set[str]:
    """STRICT: DOIDs only from xref-like fields (no free-text mining)."""
    out = set()
    if not isinstance(meta, dict):
        return out

    for x in meta.get("xrefs", []) if isinstance(meta.get("xrefs", []), list) else []:
        val = (x.get("val") or x.get("id") or x.get("xref")) if isinstance(x, dict) else x
        if isinstance(val, str):
            d = norm_doid(val)
            if d: out.add(d)

    definition = meta.get("definition")
    if isinstance(definition, dict):
        for v in definition.get("xrefs", []) if isinstance(definition.get("xrefs", []), list) else []:
            if isinstance(v, str):
                d = norm_doid(v)
                if d: out.add(d)

    for it in meta.get("basicPropertyValues", []) if isinstance(meta.get("basicPropertyValues", []), list) else []:
        if isinstance(it, dict) and isinstance(it.get("val"), str):
            d = norm_doid(it["val"])
            if d: out.add(d)

    return out


# =========================
# TASK 1: FlyBase parsing (streaming)
# =========================
task1_rows, gene_rows = set(), set()
header_found = False

with gzip.open(FLY_FILE, "rt", encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.rstrip("\n")
        if not header_found:
            if ("DO qualifier" in line) and ("DO ID" in line) and ("DO term" in line):
                h = line.lstrip("#").strip().split("\t")
                i_fbgn = h.index("FBgn ID")
                i_q    = h.index("DO qualifier")
                i_do   = h.index("DO ID")
                i_term = h.index("DO term")
                header_found = True
            continue

        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if max(i_fbgn, i_q, i_do, i_term) >= len(parts):
            continue

        fbgn = parts[i_fbgn].strip()
        qual = parts[i_q].strip()
        doid = norm_doid(parts[i_do].strip()) or parts[i_do].strip()
        term = parts[i_term].strip()

        task1_rows.add((qual, doid, term))
        gene_rows.add((fbgn, qual, doid))

unique_doids = {d for _, d, _ in task1_rows}
unique_quals = sorted({q for q, _, _ in task1_rows})
unique_genes = {g for g, _, _ in gene_rows}

print(hr("="))
print("TASK 1 — FlyBase parsing")
print(hr("-"))
print(f"Unique rows (qualifier, DOID, term): {len(task1_rows)}")
print(f"Unique DO IDs: {len(unique_doids)}")
print(f"Unique DO qualifiers ({len(unique_quals)}): {unique_quals}")
print(f"Unique genes: {len(unique_genes)}")
print(hr("="))


# =========================
# Load MONDO once
# =========================
with open(MONDO_FILE, "r", encoding="utf-8") as f:
    g = json.load(f)["graphs"][0]
nodes, edges = g.get("nodes", []), g.get("edges", [])


# =========================
# TASK 2: DOID → MONDO (strict xrefs)
# =========================
doid_to_mondos = defaultdict(set)  # DOID -> set(MONDO_URI)
for n in nodes:
    nid = n.get("id", "")
    if not (isinstance(nid, str) and MONDO_URI_RE.match(nid)):
        continue
    for d in (xref_doids(n.get("meta", {}) or {}) & unique_doids):
        doid_to_mondos[d].add(nid)

mapped_doids = set(doid_to_mondos)
all_mondo_uris = set().union(*doid_to_mondos.values()) if doid_to_mondos else set()
unmapped = sorted(unique_doids - mapped_doids)

print("TASK 2 — DOID → MONDO mapping (strict xrefs)")
print(hr("-"))
print(f"DOIDs in FlyBase: {len(unique_doids)}")
print(f"DOIDs mapped to MONDO: {len(mapped_doids)}")
print(f"Unique MONDO terms hit: {len(all_mondo_uris)}")
if PRINT_UNMAPPED_DOIDS:
    print(f"Unmapped DOIDs: {len(unmapped)} (showing up to 20: {unmapped[:20]})")

sample = list(doid_to_mondos.items())[:5]
if sample:
    print("\nSample mappings:")
    print_table(
        ["DOID", "MONDO (first up to 3)"],
        [[d, ", ".join(mondo_curie(x) for x in list(ms)[:3])] for d, ms in sample],
        maxw=70
    )
print(hr("="))


# =========================
# TASK 3: MONDO categories (ancestors exactly at TARGET_LEVEL)
# =========================
labels = {}
for n in nodes:
    nid = n.get("id", "")
    if isinstance(nid, str) and MONDO_URI_RE.match(nid):
        labels[nid] = n.get("lbl", "") or ""

parents = defaultdict(list)
children = defaultdict(list)
for e in edges:
    if not is_is_a(e.get("pred", "")):
        continue
    child = e.get("subj") or e.get("sub") or ""
    parent = e.get("obj") or ""
    if (isinstance(child, str) and isinstance(parent, str)
        and MONDO_URI_RE.match(child) and MONDO_URI_RE.match(parent)):
        parents[child].append(parent)
        children[parent].append(child)

root = "http://purl.obolibrary.org/obo/MONDO_0000001"
level = {}
if root in labels:
    level[root] = 1
    q = deque([root])
    while q:
        cur = q.popleft()
        for ch in children.get(cur, []):
            if ch not in level:
                level[ch] = level[cur] + 1
                q.append(ch)

cat_cache = {}
def categories(mondo_uri: str) -> set[str]:
    if mondo_uri in cat_cache:
        return cat_cache[mondo_uri]
    if mondo_uri not in labels or not level:
        cat_cache[mondo_uri] = set()
        return cat_cache[mondo_uri]

    cats, seen = set(), {mondo_uri}
    dq = deque([mondo_uri])
    while dq:
        cur = dq.popleft()
        for p in parents.get(cur, []):
            if p not in seen:
                seen.add(p)
                dq.append(p)
            if level.get(p) == TARGET_LEVEL:
                lbl = labels.get(p, "")
                if lbl:
                    cats.add(lbl)

    cat_cache[mondo_uri] = cats
    return cats

print("TASK 3 — MONDO categories (Project 3 style)")
print(hr("-"))
print(f"Root: {mondo_curie(root)} (level=1)")
print(f"Target ancestor level for categories: {TARGET_LEVEL}")
if all_mondo_uris:
    ex = next(iter(all_mondo_uris))
    ex_cats = sorted(categories(ex))
    print(f"Example MONDO: {mondo_curie(ex)}")
    print(f"Categories (n={len(ex_cats)}): {ex_cats[:15]}")
else:
    print("No MONDO terms available; mapping failed upstream.")
print(hr("="))


# =========================
# TASK 4: gene-level sets
# =========================
gene_to_qual = defaultdict(set)
gene_to_cat  = defaultdict(set)

for fbgn, qual, doid in gene_rows:
    gene_to_qual[fbgn].add(qual)
    for m in doid_to_mondos.get(doid, ()):
        gene_to_cat[fbgn].update(categories(m))

genes = set(gene_to_qual)
N = len(genes)
all_quals = sorted({q for s in gene_to_qual.values() for q in s})
all_cats  = sorted({c for s in gene_to_cat.values() for c in s})

genes_with_qual = {q: set() for q in all_quals}
for g, qs in gene_to_qual.items():
    for q in qs: genes_with_qual[q].add(g)

genes_with_cat = {c: set() for c in all_cats}
for g, cs in gene_to_cat.items():
    for c in cs: genes_with_cat[c].add(g)

print("TASK 4 — Contingency table setup (gene-level presence/absence)")
print(hr("-"))
print(f"Genes (N): {N}")
print(f"Qualifiers: {len(all_quals)} -> {all_quals}")
print(f"Categories: {len(all_cats)}")
print(hr("="))


# =========================
# TASK 5: Fisher (optimized set ops) + Bonus 1
# =========================
results, pvals = [], []
for cat in all_cats:
    A = genes_with_cat[cat]
    nA = len(A)
    for qual in all_quals:
        B = genes_with_qual[qual]
        nB = len(B)

        a = len(A & B)
        b = nA - a
        c = nB - a
        d = N - nA - nB + a

        odds, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
        expected = (nA * nB) / N if N else 0.0
        fold = (a / expected) if expected > 0 else None

        results.append({"p": p, "odds": odds, "cat": cat, "qual": qual,
                        "a": a, "expected": expected, "fold": fold})
        pvals.append(p)

print("TASK 5 — Fisher exact tests")
print(hr("-"))
print(f"Total tests: {len(results)} (= {len(all_cats)} categories × {len(all_quals)} qualifiers)")
print(hr("="))


# =========================
# TASK 6: TOPK
# =========================
topk = sorted(results, key=lambda r: r["p"])[:TOPK]
print(f"TASK 6 — TOP {TOPK} lowest p-values (Bonus 1 included)")
print(hr("-"))
print_table(
    ["rank", "p", "odds", "a", "expected", "fold", "qualifier", "category"],
    [[str(i), sci(r["p"]), odds_str(r["odds"]), str(r["a"]),
      f4(r["expected"]), (f"{r['fold']:.2f}" if r["fold"] is not None else "NA"),
      r["qual"], r["cat"]] for i, r in enumerate(topk, 1)],
    maxw=60
)
print(hr("="))


# =========================
# BONUS 2: BH FDR (sorted by padj)
# =========================
reject, padj, _, _ = multipletests(pvals, alpha=ALPHA, method="fdr_bh")
sig = sorted(((padj_i, r) for r, rej, padj_i in zip(results, reject, padj) if rej), key=lambda x: x[0])

print("BONUS 2 — Significant after FDR (BH)")
print(hr("-"))
print(f"alpha={ALPHA} | significant tests: {len(sig)} / {len(results)}")

lim = len(sig) if SHOW_SIGNIFICANT_LIMIT is None else min(SHOW_SIGNIFICANT_LIMIT, len(sig))
if sig:
    print_table(
        ["p", "padj(BH)", "odds", "a", "expected", "fold", "qualifier", "category"],
        [[sci(r["p"]), sci(padj_i), odds_str(r["odds"]), str(r["a"]),
          f4(r["expected"]), (f"{r['fold']:.2f}" if r["fold"] is not None else "NA"),
          r["qual"], r["cat"]] for padj_i, r in sig[:lim]],
        maxw=60
    )
    if SHOW_SIGNIFICANT_LIMIT is not None and len(sig) > SHOW_SIGNIFICANT_LIMIT:
        print(f"\n(Showing first {SHOW_SIGNIFICANT_LIMIT} of {len(sig)}. Set SHOW_SIGNIFICANT_LIMIT=None to show all.)")
else:
    print("No significant results at the given alpha.")
print(hr("="))