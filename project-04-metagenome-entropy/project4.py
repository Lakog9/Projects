

# Project 4

# ----------------------------------------------------------------------------
# Περιγραφή: Υπολογισμός Shannon Entropy και k-mer diversity σε δεδομένα metagenomics.
#
# Χρήση:
#   1. Για k=4,5,6,7: python project4.py MH0026_081224.clean.1.fq.gz
#   2. Για συγκεκριμένο k: python project4.py MH0026_081224.clean.1.fq.gz <k_value>
# π.χ. για k=7 run: python project4.py MH0026_081224.clean.1.fq.gz 7

import gzip
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import sys


def shannon_entropy(counts):
    #Υπολογισμός Shannon entropy από τα counts των k-mers.
    total = sum(counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total       # πιθανότητα εμφάνισης k-mer
            entropy -= p * np.log2(p)
    return entropy


def metagenome_entropy(fastq_file, k=None):
    
    #Αν k=None -> τρέχουμε για k = 4,5,6,7.
    #Αλλιώς τρέχουμε μόνο για το δοσμένο k.
    
    if k is None:
        k_values = range(4, 8)
    else:
        k_values = [k]

    # Καινούριο figure για όλα τα k μαζί
    plt.figure()

    for current_k in k_values:
        kmer_counts = Counter()
        previous_entropy = None
        current_entropy = 0.0
        reads_processed = 0
        total_kmers = 0

        # ιστορικό για το γράφημα
        reads_history = []
        unique_history = []

        # ανοίγουμε το συμπιεσμένο FASTQ
        with gzip.open(fastq_file, "rt") as f:
            line_num = 0
            for line in f:
                # sequence line: 2η γραμμή σε κάθε 4άδα
                if line_num % 4 == 1:
                    sequence = line.strip()
                    seq_len = len(sequence)

                    # δημιουργία k-mers με sliding window
                    i = 0
                    while i <= seq_len - current_k:
                        kmer = sequence[i:i + current_k]
                        if "N" not in kmer:  # αγνοούμε ambiguous βάσεις
                            kmer_counts[kmer] += 1
                            total_kmers += 1
                        i += 1

                    reads_processed += 1

                    # κάθε 1000 reads: υπολογισμός entropy + αποθήκευση στο history
                    if reads_processed % 1000 == 0:
                        current_entropy = shannon_entropy(kmer_counts)

                        reads_history.append(reads_processed)
                        unique_history.append(len(kmer_counts))

                        if previous_entropy is not None:
                            diff = abs(current_entropy - previous_entropy)
                            if diff < 0.00001:
                                # σύγκλιση -> σταματάμε
                                break
                        previous_entropy = current_entropy

                line_num += 1

        # αν δεν προλάβαμε να υπολογίσουμε entropy μέσα στο loop
        if previous_entropy is None:
            current_entropy = shannon_entropy(kmer_counts)

        # φρόντισε να υπάρχει και το τελικό σημείο στο history
        if not reads_history or reads_history[-1] != reads_processed:
            reads_history.append(reads_processed)
            unique_history.append(len(kmer_counts))

        # εκτύπωση αποτελεσμάτων για το current_k
        print(
            "k: " + str(current_k) +
            ", Reads: " + str(reads_processed) +
            ", Total k-mers: " + str(total_kmers) +
            ", Unique k-mers: " + str(len(kmer_counts)) +
            ", Shannon Entropy: " + str(round(current_entropy, 4))
        )

        # γράφημα για αυτό το k (ιστορικό, όχι σταθερή γραμμή)
        plt.plot(reads_history, unique_history, label="k=" + str(current_k))

    # ρυθμίσεις τελικού γραφήματος
    plt.xlabel("Number of Reads")
    plt.ylabel("Number of Unique k-mers")
    plt.title("k-mer Diversity")
    plt.legend()
    plt.tight_layout()
    plt.savefig("kmer_diversity.svg")  # vector format, καλή ποιότητα
    plt.show()


# Main program
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Χρήση: python project4.py <fastq_file.fq.gz> [k]")
        sys.exit(1)

    file_to_process = sys.argv[1]
    k_param = int(sys.argv[2]) if len(sys.argv) > 2 else None

    metagenome_entropy(file_to_process, k_param)