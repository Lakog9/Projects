

import gzip
import random

# ΒΟΗΘΗΤΙΚΕΣ ΣΥΝΑΡΤΗΣΕΙΣ (Τα εργαλεία μας)

def get_most_frequent_pair(tokens):
    
    #Διαβάζει τη λίστα των tokens και βρίσκει ποιο ζευγάρι εμφανίζεται τις περισσότερες φορές.
    
    pairs_count = {}
    
    # Πάμε από το 0 μέχρι το προτελευταίο στοιχείο
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i+1]) # Φτιάχνουμε το ζευγάρι
        
        if pair in pairs_count:
            pairs_count[pair] += 1
        else:
            pairs_count[pair] = 1
            
    # Αν δεν βρήκαμε κανένα ζευγάρι (πολύ μικρή λίστα), επιστρέφουμε none
    if not pairs_count:
        return None

    # Βρίσκουμε το ζευγάρι με το μεγαλύτερο count
    best_pair = None
    max_count = -1
    
    for pair, count in pairs_count.items():
        if count > max_count:
            max_count = count
            best_pair = pair
            
    return best_pair

def merge_pair_in_sequence(tokens, pair_to_merge, new_token):
    
    #Διασχίζει τη λίστα και όπου βρει το pair_to_merge, το αντικαθιστά με το new_token.
    
    new_sequence = []
    i = 0
    
    while i < len(tokens):
        # Ελέγχουμε αν είμαστε στο ζευγάρι που ψάχνουμε
        # Πρέπει να μην είμαστε στο τελευταίο στοιχείο για να ελέγξουμε το επόμενο
        if i < len(tokens) - 1 and tokens[i] == pair_to_merge[0] and tokens[i+1] == pair_to_merge[1]:
            new_sequence.append(new_token)
            i += 2  # Προχωράμε 2 βήματα γιατί "φάγαμε" δύο στοιχεία
        else:
            new_sequence.append(tokens[i])
            i += 1  # Προχωράμε κανονικά
            
    return new_sequence


# PART 1: Εκπαίδευση (Train BPE)

def train_bpe(sequence, K):
    # Μετατρέπουμε το string σε λίστα χαρακτήρων: ['A', 'C', 'G', ...]
    tokens = list(sequence)
    
    # Εδώ θα αποθηκεύουμε τους κανόνες μας. Π.χ. ('A', 'C') -> 'AC'
    rules = [] 
    
    # Το λεξιλόγιο αρχικά είναι τα 4 γράμματα
    vocab = set(tokens)
    
    # Επαναλαμβάνουμε μέχρι να φτάσουμε τα K tokens στο λεξιλόγιο
    while len(vocab) < K:
        best_pair = get_most_frequent_pair(tokens)
        
        if best_pair is None:
            break # Δεν υπάρχουν άλλα ζευγάρια να ενώσουμε
            
        # Φτιάχνουμε το νέο token ενώνοντας τα δύο μέρη
        new_token = best_pair[0] + best_pair[1]
        
        # Εφαρμόζουμε την αλλαγή στη λίστα μας
        tokens = merge_pair_in_sequence(tokens, best_pair, new_token)
        
        # Αποθηκεύουμε τον κανόνα και το νέο token
        rules.append((best_pair, new_token))
        vocab.add(new_token)
        
    return rules


# PART 2: Tokenization (Χρήση του BPE)


def tokenize_dna(sequence, rules):
    # Ξεκινάμε πάλι με απλά γράμματα
    tokens = list(sequence)
    
    # Εφαρμόζουμε τους κανόνες ΜΕ ΤΗ ΣΕΙΡΑ που τους μάθαμε
    for pair, new_token in rules:
        tokens = merge_pair_in_sequence(tokens, pair, new_token)
        
    return tokens


# PART 3 & 4: Αξιολόγηση και Πειράματα


def load_data(filename):
    print("Φόρτωση αρχείου (μπορεί να πάρει λίγο χρόνο)...")
    with gzip.open(filename, 'rt') as f:
        # Διαβάζουμε όλες τις γραμμές
        lines = f.readlines()
        
    # Η πρώτη γραμμή είναι header (>chr20), την πετάμε
    dna_lines = lines[1:]
    
    # Ενώνουμε τις γραμμές και αφαιρούμε τα 'new line' (\n)
    # Επίσης τα κάνουμε όλα κεφαλαία για σιγουριά
    full_sequence = "".join(dna_lines).replace('\n', '').upper()
    
    # Κρατάμε μόνο τα A, C, G, T (αν υπάρχει κανένα N το πετάμε)
    full_sequence = "".join([c for c in full_sequence if c in 'ACGT'])
    
    print(f"Το αρχείο φορτώθηκε! Μήκος DNA: {len(full_sequence)} βάσεις.")
    return full_sequence

def run_experiments(full_sequence):
    # Παράμετροι που ζητάει η άσκηση
    # Μπορείς να προσθέσεις κι άλλα νούμερα στις λίστες αν θες
    K_values = [10, 50, 100, 200, 500, 1000] 
    M_values = [1000, 5000, 10000, 50000]
    
    print("\nStarting Experiments...")
    print(f"{'K':<10} {'M':<10} {'Average # Tokens':<20}")
    print("-" * 40)
    
    for K in K_values:
        for M in M_values:
            total_tokens_length = 0
            trials = 100 # Η άσκηση λέει 100 επαναλήψεις
            
            for _ in range(trials):
                # 1. Διαλέγουμε τυχαίο κομμάτι για ΕΚΠΑΙΔΕΥΣΗ (Training)
                start = random.randint(0, len(full_sequence) - M)
                train_seq = full_sequence[start : start + M]
                
                # 2. Εκπαιδεύουμε το μοντέλο
                learned_rules = train_bpe(train_seq, K)
                
                # 3. Διαλέγουμε τυχαίο κομμάτι για ΕΛΕΓΧΟ (Testing)
                test_start = random.randint(0, len(full_sequence) - M)
                test_seq = full_sequence[test_start : test_start + M]
                
                # 4. Κάνουμε tokenize
                final_tokens = tokenize_dna(test_seq, learned_rules)
                
                # 5. Μετράμε πόσα tokens βγήκαν
                total_tokens_length += len(final_tokens)
            
            # Υπολογίζουμε τον μέσο όρο
            avg_tokens = total_tokens_length / trials
            print(f"{K:<10} {M:<10} {avg_tokens:<20.2f}")


# MAIN - Εδώ ξεκινάνε όλα


if __name__ == "__main__":
    # Το όνομα του αρχείου που έχεις κατεβάσει
    FILENAME = "chr20.fa.gz"
    
    try:
        # Φορτώνουμε τα δεδομένα
        dna_data = load_data(FILENAME)
        
        # Τρέχουμε τα πειράματα
        run_experiments(dna_data)
        
    except FileNotFoundError:
        print(f"Σφάλμα: Δεν βρέθηκε το αρχείο {FILENAME} στον φάκελο.")