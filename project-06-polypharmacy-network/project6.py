import argparse
import os
import gzip
import csv
import sys
import time


# 1. ΒΟΗΘΗΤΙΚΕΣ ΣΥΝΑΡΤΗΣΕΙΣ (UNION-FIND)


def find_root(parents, node):
    
    #Εντοπισμός Ρίζας (Root Finding) με Path Compression.
    
    if parents[node] != node:
        parents[node] = find_root(parents, parents[node])
    return parents[node]

def union_nodes(parents, sizes, node_a, node_b):

    #Ένωση Συνόλων (Union) με Union by Size.
    
    root_a = find_root(parents, node_a)
    root_b = find_root(parents, node_b)

    if root_a == root_b:
        return False # Κύκλος: Ήδη στην ίδια συνιστώσα

    # Προσάρτηση του μικρότερου δέντρου στο μεγαλύτερο
    if sizes[root_a] < sizes[root_b]:
        parents[root_a] = root_b
        sizes[root_b] += sizes[root_a]
    else:
        parents[root_b] = root_a
        sizes[root_a] += sizes[root_b]
    
    return True

# 2. ΚΥΡΙΩΣ ΠΡΟΓΡΑΜΜΑ (MAIN SCRIPT)

def main():
    parser = argparse.ArgumentParser(
        description="Polypharmacy side-effect association network (Union-Find)"
    )
    parser.add_argument(
        "--csv_gz",
        required=True,
        help="Full path to ChChSe-Decagon_polypharmacy.csv.gz",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path (default: <script_dir>/sorted_results.csv)",
    )
    args = parser.parse_args()

    input_file = args.csv_gz

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = args.out if args.out else os.path.join(script_dir, "sorted_results.csv")

    if not os.path.isfile(input_file):
        print(f"Σφάλμα: Το αρχείο εισόδου δεν βρέθηκε: {input_file}")
        return

    
    print(f"1. Ανάγνωση αρχείου stream: {input_file} ...")
    start_time = time.time()

    diseases = {}

    try:
        # Streaming ανάγνωση (Memory efficient)
        with gzip.open(input_file, 'rt', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader) # Skip header

            for row in reader:
                if len(row) < 4: continue

                drug1, drug2, d_id, d_name = row[0], row[1], row[2], row[3]

                # Αρχικοποίηση Δομής Ασθένειας
                if d_id not in diseases:
                    diseases[d_id] = {
                        'name': d_name,
                        'parents': {},
                        'sizes': {},
                        'num_components': 0,
                        'node_count': 0 
                    }
                
                state = diseases[d_id]
                parents = state['parents']
                sizes = state['sizes']

                # Lazy Initialization Κόμβων
                if drug1 not in parents:
                    parents[drug1] = drug1
                    sizes[drug1] = 1
                    state['num_components'] += 1
                    state['node_count'] += 1
                
                if drug2 not in parents:
                    parents[drug2] = drug2
                    sizes[drug2] = 1
                    state['num_components'] += 1
                    state['node_count'] += 1

                # Δημιουργία Ακμής (Union)
                if union_nodes(parents, sizes, drug1, drug2):
                    state['num_components'] -= 1

    except FileNotFoundError:
        print("Σφάλμα: Το αρχείο εισόδου δεν βρέθηκε.")
        return

    print(f"2. Επεξεργασία ολοκληρώθηκε σε {time.time() - start_time:.2f} sec.")
    print("3. Ταξινόμηση και εξαγωγή αποτελεσμάτων...\n")

    
    # 3. ΤΑΞΙΝΟΜΗΣΗ ΚΑΙ ΕΓΓΡΑΦΗ (SORTING & EXPORT)

    # Λίστα για να αποθηκεύσουμε τα επεξεργασμένα δεδομένα πριν την εγγραφή
    results_list = []
    
    # Μεταβλητές για το Main Question
    max_single_comp_size = -1
    winning_disease = None

    for d_id, state in diseases.items():
        n_comp = state['num_components']
        total_nodes = state['node_count']
        
        # Υπολογισμός Μέσου Μεγέθους (Bonus)
        avg_size = total_nodes / n_comp if n_comp > 0 else 0

        # Αποθήκευση στη λίστα
        results_list.append({
            'id': d_id,
            'name': state['name'],
            'components': n_comp,
            'total_nodes': total_nodes,
            'avg_size': avg_size
        })

        # Έλεγχος για το Main Question (Single Component Winner)
        if n_comp == 1:
            if total_nodes > max_single_comp_size:
                max_single_comp_size = total_nodes
                winning_disease = (d_id, state['name'], total_nodes)

    # ΤΑΞΙΝΟΜΗΣΗ (SORTING)
    # Ταξινομούμε τη λίστα με βάση το 'total_nodes' (Συνολικά Φάρμακα)
    # reverse=True σημαίνει Φθίνουσα σειρά (από το μεγαλύτερο στο μικρότερο)
    results_list.sort(key=lambda x: x['total_nodes'], reverse=True)

    # ΕΓΓΡΑΦΗ ΣΕ CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as out_f:
        writer = csv.writer(out_f)
        # Επικεφαλίδες
        writer.writerow(['Rank', 'Disease ID', 'Name', 'Total Drugs (Nodes)', 'Components', 'Avg Component Size'])

        rank = 1
        for res in results_list:
            writer.writerow([
                rank,
                res['id'],
                res['name'],
                res['total_nodes'],
                res['components'],
                f"{res['avg_size']:.2f}"
            ])
            rank += 1

    
    # 4. (TERMINAL OUTPUT)
    

    print("-" * 60)
    print(f"Τα αποτελέσματα ταξινομήθηκαν και αποθηκεύτηκαν στο: {output_file}")
    print("-" * 60)

    if winning_disease:
        w_id, w_name, w_size = winning_disease
        print("\n=== ΤΕΛΙΚΗ ΑΠΑΝΤΗΣΗ (MAIN PROJECT QUESTION) ===")
        print(f"Η ασθένεια με 1 συνεκτική συνιστώσα και το μεγαλύτερο μέγεθος:")
        print(f"Όνομα:   {w_name}")
        print(f"ID:      {w_id}")
        print(f"Μέγεθος: {w_size} φάρμακα")
    else:
        print("\nΔεν βρέθηκε ασθένεια με μία μόνο συνεκτική συνιστώσα.")

if __name__ == "__main__":
    main()