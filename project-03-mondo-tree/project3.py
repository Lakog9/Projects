
#Project 3

import json
import re
#είναι σαν μια double-ended ουρά που βγάζει στοιχείο από την αρχή(popleft()) και βάζει στοιχείο στο τέλος(append())
from collections import deque



with open("mondo.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    
#pairnoume to graph dict
g0 = data["graphs"][0]

# Παίρνω τη λίστα με nodes και edges από το πρώτο graph
raw_nodes = g0["nodes"]
raw_edges = g0["edges"]


# regex για MONDO URL
mondo_url_re = re.compile(r"^http://purl\.obolibrary\.org/obo/MONDO_\d+$")

#φτιάχνω ένα dictionary για να κρατάω τα labels
mondo_labels = {}  # key:MONDO url -> value: lbl(the 'name') 
#π.χ.:mondo_labels["http://purl.obolibrary.org/obo/MONDO_0002816"] = "adrenal cortex disorder"

#ψάχνοθμε σε όλα τα node του JSON.Κάθε n είναι κάπως έτσι:{"id": "...", "lbl": "...", "type": "...", ...}
for n in raw_nodes:
    node_id = n.get("id", "") #αν υπάρχει id,κράτα το αλλιώς γύρνα ""
    if isinstance(node_id, str) and mondo_url_re.match(node_id):
        lbl = n.get("lbl", "") #το ίδιο με πριν αλλά για lbl
        if isinstance(lbl, str):
            #στο τέλος θα έχει όλα τα MONDO nodes που μας ενδιαφέρουν από το JSON αρχείο
            mondo_labels[node_id] = lbl


#Φτιάχνω μία λίστα για τα filtered edge
filtered_edges = []  #list of (child_url, parent_url)
#περνάμε από όλα τα edges του JSON
for e in raw_edges:
    #κάθε e είναι ένα dict της μορφής:{"sub": "...", "pred": "is_a", "obj": "...", "meta": {...}}
    if e.get("pred") != "is_a": #αν το pred δεν είναι 'is_a' τότε skip αυτό το edge
        continue
    #κρατάμε αυτά που μας ενδιαφέρουν
    sub = e.get("sub", "")
    obj = e.get("obj", "")

    #σιγουρευόμαστε οτι είναι strings
    if not (isinstance(sub, str) and isinstance(obj, str)):
        continue
    #σιγουρευόμαστε οτι είναι και τα 2 MONDO URLs
    if mondo_url_re.match(sub) and mondo_url_re.match(obj):
        filtered_edges.append((sub, obj)) #κρατάμε τα edges που θέλουμε


#Φτιάχνω δομές tree: parent->children και child->parent
children_of = {}   # parent_url -> list of child_url  Example: children_of[parent] = [child1, child2, ...]
parents_of = {}     # child_url -> parent_url  Example: parents_of[child] = [parent1,parent2,...]

for child, parent in filtered_edges: #διαβάζουμε edges ενα-ενα.Η λίστα filtered_edges έιναι κάπως έτσι:[(child_url, parent_url),(child_url, parent_url),...]

    #σιγουρευόμαστε οτι υπάρχει μια λίστα για τον parent μέσα στο dict children_of.Μετά, κλανουμε append το child σε αυτή τη λίστα
    children_of.setdefault(parent, []).append(child)
    #αντίστοιχα
    parents_of.setdefault(child, []).append(parent) 


#Βρίσκω roots: κόμβοι που δεν είναι child σε κανένα edge
#Δηλαδή: nodes που δεν έχουν parent
all_nodes = set(mondo_labels.keys()) #αποθηκέυω όλα τα MONDO node URLs που έχουμε κρατήσει πριν σε σετ
#root->node χωρις parent, δεν εμφανίσονται σαν sub, μονο σαν obj.
roots = [n for n in all_nodes if n not in parents_of]

#ορίζω main_root αυτό το MONDO που είναι το disease γιατί απο αυτό ξεκινάμε συνήθως
main_root = "http://purl.obolibrary.org/obo/MONDO_0000001"  # disease

#ελέγχω αν υπάρχει αυτό το URL στα nodes που κρατήσαμε, 
if main_root in mondo_labels:
    root = main_root #αν υπάρχει, το κρατάμε σαν το roots μας. Δηλαδή οταν μετράμε levels, θα ξεκινάμε από εδώ
else:
    #Αν δεν υπάρχει, τοτε επιλέγουμε το 1ο root που βρήκαμε πριν στα roots.Αν δεν υπάρχουν καθόλου roots, τοτε βάζουμε ως root=None
    root = roots[0] if roots else None


level = {}  #dictionary που θα αποθηκεύει π.χ. level[node_url]=1,2,3,.. node_url -> level number.Δηλαδή το root θα έχει level 1, children level 2 etc.

#safety check.Αν δεν είχαμε root, σταματάμε
if root is not None:
    q = deque([root]) #φτιάχνω ουρά μόνο με το root μέσα στην αρχή
    level[root] = 1  #level 1 = root

    while q: #τρέχει οσο υπάρχουν nodes
        cur = q.popleft() #πάρε το 1ο node απο το queue.(cur->current node)
        cur_level = level[cur]

        #Αν το cur έχει children-> retrun the list.Αλλιώς->return empty list
        for child in children_of.get(cur, []):
            if child not in level:           #αν δεν το έχουμε δει ξανά
                level[child] = cur_level + 1 #level child = level parent + 1
                q.append(child)


def part_2(mondo_id_short):

    start=f"http://purl.obolibrary.org/obo/{mondo_id_short}"

    #αν δεν υπάρχει στο graph μας, επιστρέφουμε []
    if start not in mondo_labels:
        return []
    
    #μας ενδιαφέρουν οι κόμβοι στο level 3 από το root
    TARGET_LEVEL = 3

    #ουρά BFS που ξεκινά με το start
    q=deque([start])
    #set για να αποφύγουμε διπλές επαναλήψεις
    visited = {start}

    #η λίστα αποτελεσμάτων
    mondo_categories = []
    #για να μη βάλουμε το ίδιο label 2 φορές(αν το συναντήσουμε ξανά)
    seen_labels = set()

    while q:
        cur = q.popleft() #πάρε τον 1ο κόμβο από την ουρά

        for p in parents_of.get(cur,[]): #παίρνουμε όλους τους γονείς του cur.Αν δεν έχει δίνει άδεια λίστα
            #αν δεν έχει εξεταστεί ήδη.Τον μαρκαρουμε ως visited
            if p not in visited:
                visited.add(p)
                q.append(p) #τον βάζουμε στην ουρά για να συνεχίσουμε να ανεβαίνουμε προς τα πάνω

            #αν ο πρόγονος είναι level 3,τότε μας κάνει
            if level.get(p)==TARGET_LEVEL:
                lbl = mondo_labels.get(p,"") #παίρνω το lbl του p
                if lbl and lbl not in seen_labels:
                    mondo_categories.append(lbl)
                    seen_labels.add(lbl)

    return mondo_categories


mondo_categories=part_2("MONDO_0011719")
print(mondo_categories)