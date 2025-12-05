import pandas as pd
import numpy as np
import time
from collections import Counter
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import psutil
from itertools import combinations
import matplotlib.pyplot as plt
####################################################################################################################

df = pd.read_csv("heart.csv")

# Handle missing values
def handle_missing_values(df):
    for col in df.columns:
        if df[col].isna().any() or (df[col].dtype in [np.float64, np.int64] and not np.isfinite(df[col]).all()):
            print(f"Column {col} has missing or non-finite values")
            if df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna("Unknown")
    return df

df = handle_missing_values(df)

# Discretize numeric columns into bins
def discretize_column(df, column, bins=3):
    labels = [f"{column}_low", f"{column}_medium", f"{column}_high"]
    df[column] = pd.cut(df[column], bins=bins, labels=labels)
    return df

numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
for col in numerical_cols:
    if col in df.columns:
        df = discretize_column(df, col, bins=3)

# Transform the dataset into transaction format
def transform_to_transactions(df):
    transactions = []
    for i in range(len(df)):
        row = [f"{col}={val}" for col, val in df.iloc[i].items() if pd.notna(val)]
        transactions.append(row)
    return transactions

transactions = transform_to_transactions(df)

# Global dynamic minsup from the full transactions
def calculate_dynamic_minsup_weighted(transactions):
    total_transactions = len(transactions)
    item_counts = Counter(item for transaction in transactions for item in transaction)
    counts = np.array(list(item_counts.values()))
    Q1 = np.percentile(counts, 25)
    Q2 = np.percentile(counts, 50)
    weighted_percentile = (Q1 + Q2) / 2.0
    minsup = weighted_percentile / total_transactions
    print(f"Dynamic minimum support (weighted percentile, full transactions): {minsup:.4f}")
    return max(0.01, min(0.5, minsup))  

# Recalculate dynamic minsup based on current iteration's frequent itemsets
def calculate_dynamic_minsup_current_weighted(frequent_itemsets, transactions):
    if frequent_itemsets:
        supports = [calculate_support(itemset, transactions) for itemset in frequent_itemsets]
        Q1 = np.percentile(supports, 25)
        Q2 = np.percentile(supports, 50)
        weighted_percentile = (Q1 + Q2) / 2.0
        new_minsup = weighted_percentile
        print(f"Dynamic minimum support recalculated (current iteration, weighted percentile): {new_minsup:.4f}")
        return max(0.01, new_minsup)
    else:
        return 0.01


# Transaction Encoding for mlxtend's Apriori
def encode_transactions(transactions):
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    return pd.DataFrame(te_array, columns=te.columns_)

df_encoded = encode_transactions(transactions)
dynamic_minsup = calculate_dynamic_minsup_weighted(transactions)


# Utility Functions
def measure_memory_usage(func, *args, **kwargs):
    process = psutil.Process()
    memory_before = process.memory_info().rss  # in bytes
    result = func(*args, **kwargs)
    memory_after = process.memory_info().rss
    memory_used = (memory_after - memory_before) / (1024 * 1024)  # in MB
    return result, memory_used

def calculate_support(itemset, transactions):
    count = sum(1 for transaction in transactions if itemset.issubset(set(transaction)))
    return count / len(transactions)

def generate_candidates(transactions, size):
    # Extract all unique items
    items = [item for transaction in transactions for item in transaction]
    return [set(comb) for comb in combinations(set(items), size)]


# Custom Apriori Algorithm with Dynamic Minimum Support (Weighted Percentile)
def apriori_dynamic_weighted(transactions):
    frequent_itemsets = []
    size = 1
    minsup = calculate_dynamic_minsup_weighted(transactions)
    iteration = 1
    while True:
        print(f"\nIteration {iteration}:")
        print(f"Using minimum support threshold: {minsup:.4f}")
        candidates = generate_candidates(transactions, size)
        frequent = [itemset for itemset in candidates if calculate_support(itemset, transactions) >= minsup]
        if not frequent:
            break
        print(f"Found {len(frequent)} frequent itemsets for size {size}.")
        frequent_itemsets.extend(frequent)
        size += 1
        # Recalculate minsup based on current iteration's frequent itemsets
        minsup = calculate_dynamic_minsup_current_weighted(frequent, transactions)
        iteration += 1
    return frequent_itemsets


##########################################################Algorithm Classic############################################################
start_time_fixed = time.time()
frequent_itemsets_fixed, memory_usage_fixed = measure_memory_usage(
    apriori, df_encoded, min_support=0.3, use_colnames=True
)
end_time_fixed = time.time()
time_taken_fixed = end_time_fixed - start_time_fixed

print(f"\nFixed Minimum Support (0.3):")
print(f"Time taken: {time_taken_fixed:.4f} seconds")
print(f"Memory usage: {memory_usage_fixed:.4f} MB")
print("\nFrequent Itemsets (Fixed):")
print(frequent_itemsets_fixed)
frequent_itemsets_fixed.to_csv("frequent_itemsets_fixed.csv", index=False)

########################################################Dynamic algorithm###########################################################################

start_time_dynamic = time.time()
frequent_itemsets_dynamic_weighted, memory_usage_dynamic = measure_memory_usage(
    apriori_dynamic_weighted, transactions
)
end_time_dynamic = time.time()
time_taken_dynamic = end_time_dynamic - start_time_dynamic

print(f"\nDynamic Minimum Support (Weighted Percentile):")
print(f"Time taken: {time_taken_dynamic:.4f} seconds")
print(f"Memory usage: {memory_usage_dynamic:.4f} MB")
print("\nFrequent Itemsets (Dynamic Weighted):")
print(frequent_itemsets_dynamic_weighted)

# Compute support for each dynamically found itemset and store in a DataFrame for rule extraction
dynamic_supports = [calculate_support(itemset, transactions) for itemset in frequent_itemsets_dynamic_weighted]
frequent_itemsets_dynamic_df = pd.DataFrame({
    "itemsets": frequent_itemsets_dynamic_weighted,
    "support": dynamic_supports
})
frequent_itemsets_dynamic_df.to_csv("frequent_itemsets_dynamic_weighted.csv", index=False)

#############################################Association Rules#########################33######################

# Association rules for fixed minsup (0.3)
rules_fixed = association_rules(frequent_itemsets_fixed, metric="lift", min_threshold=1.0)
# Association rules for dynamic weighted minsup
rules_dynamic = association_rules(frequent_itemsets_dynamic_df, metric="lift", min_threshold=1.0)

avg_lift_fixed = rules_fixed["lift"].mean() if not rules_fixed.empty else 0
avg_lift_dynamic = rules_dynamic["lift"].mean() if not rules_dynamic.empty else 0
avg_conf_fixed = rules_fixed["confidence"].mean() if not rules_fixed.empty else 0
avg_conf_dynamic = rules_dynamic["confidence"].mean() if not rules_dynamic.empty else 0

print("\nComparison of Association Rules:")
print(f"Fixed minsup (0.3) - Number of Itemsets: {len(frequent_itemsets_fixed)}")
print(f"Dynamic weighted minsup - Number of Itemsets: {len(frequent_itemsets_dynamic_weighted)}")
print(f"Fixed minsup (0.3) - Average Lift: {avg_lift_fixed:.4f}")
print(f"Dynamic weighted minsup - Average Lift: {avg_lift_dynamic:.4f}")
print(f"Fixed minsup (0.3) - Average Confidence: {avg_conf_fixed:.4f}")
print(f"Dynamic weighted minsup - Average Confidence: {avg_conf_dynamic:.4f}")



labels = ['Fixed minsup (0.3)', 'Dynamic Weighted']
num_itemsets = [len(frequent_itemsets_fixed), len(frequent_itemsets_dynamic_weighted)]
avg_lift = [avg_lift_fixed, avg_lift_dynamic]
avg_conf = [avg_conf_fixed, avg_conf_dynamic]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Number of Frequent Itemsets
axes[0].bar(labels, num_itemsets, color=['blue', 'green'])
axes[0].set_title('Number of Frequent Itemsets')
axes[0].set_ylabel('Count')

# Plot 2: Average Lift
axes[1].bar(labels, avg_lift, color=['blue', 'green'])
axes[1].set_title('Average Lift of Association Rules')
axes[1].set_ylabel('Lift')

# Plot 3: Average Confidence
axes[2].bar(labels, avg_conf, color=['blue', 'green'])
axes[2].set_title('Average Confidence of Association Rules')
axes[2].set_ylabel('Confidence')

plt.tight_layout()
plt.show()

#Graph showing the number of association rules vs static minsup thresholds

static_thresholds = np.linspace(0.05, 0.5, 10)
static_rules_counts = []

for thresh in static_thresholds:
    freq_itemsets = apriori(df_encoded, min_support=thresh, use_colnames=True)
    if not freq_itemsets.empty:
        rules = association_rules(freq_itemsets, metric="lift", min_threshold=1.0)
        static_rules_counts.append(len(rules))
    else:
        static_rules_counts.append(0)

plt.figure(figsize=(8, 5))
plt.plot(static_thresholds, static_rules_counts, marker='o', color='blue')
plt.title("Number of Association Rules vs Static Minsup")
plt.xlabel("Static Minsup")
plt.ylabel("Number of Rules")
plt.grid(True)
plt.show()


##############################################################################################################################################################################
"""
Cette approche est meilleure pour plusieurs raisons :

1.Adaptée à la distribution des occurrences :
   - Elle ajuste dynamiquement le minsup en fonction des fréquences des items, utilisant les percentiles (Q1 et Q2) 
   pour refléter la tendance centrale des données.

2.Robustesse avec les percentiles pondérés:
   - La combinaison de Q1 et Q2 réduit l'impact des valeurs aberrantes, donnant un minsup plus représentatif.

3.Adaptabilité à la variabilité des données:
   - Si la variabilité des données est élevée, le minsup est plus bas pour conserver des motifs diversifiés ; 
   sinon, il est plus élevé pour filtrer les bruits.

5.Souplesse selon la taille des données:
   - Il ajuste le minsup en fonction du nombre total de transactions, ce qui le rend applicable à des ensembles de données de différentes tailles.

6. **Plage de clamping (0.01 à 0.5)** :
   - Le minsup est limité pour éviter des valeurs extrêmes, assurant un bon compromis entre sévérité et flexibilité.


   Prsk l median whdo mytritich les cas rares
   W 3la hadi zdna khdmna b 25%
Apr combinina binathm w dinan
Donc l median rah ydi l cas général w 25% tzidlnanles cas rares w btali la moyenne t3hm ha tkoun afdal mn median whdha w haka ntocho items aktr 
0.75+0.5/2 Less Balance Between Lower and Middle Data:

"""


"""

2. Lift moyen des règles d'association (Graphique du milieu)
Le lift mesure l’intérêt des règles d’association (plus il est grand, plus la règle est utile).
Avec un minsup fixe (0.3), le lift est plus élevé qu'avec la méthode dynamique.
Cela peut indiquer que le seuil fixe retient des règles qui ont un fort pouvoir explicatif, mais potentiellement trop nombreuses.

3. Confiance moyenne des règles d'association (Graphique de droite)
La confiance indique à quelle fréquence la conséquence d’une règle se réalise lorsqu'on observe la condition.
Ici, la méthode dynamique pondérée donne des règles avec une confiance plus élevée que celles issues du minsup=0.3.
Cela suggère que la sélection des motifs dans l’approche dynamique privilégie les règles plus fiables, bien que moins nombreuses.
L'approche avec minsup fixe extrait plus de motifs et de règles, mais au prix d’une confiance moyenne plus faible.
En revanche, la méthode dynamique pondérée extrait moins de règles, mais ces règles ont une meilleure fiabilité (confiance plus élevée). Cela montre que la sélection dynamique peut être utile pour filtrer les motifs non pertinents et se concentrer sur ceux qui sont les plus significatifs.
Forte décroissance initiale :
Lorsque minsup est très faible (environ 0.05), le nombre de règles est extrêmement élevé (~360 000).
Cela s’explique par le fait qu’un seuil trop bas inclut presque toutes les combinaisons possibles d’items, même celles apparaissant rarement.
Diminution rapide autour de minsup=0.1 :

En augmentant légèrement minsup, le nombre de règles chute brutalement (~50 000 règles).
Cela signifie que de nombreuses règles reposaient sur des motifs peu fréquents, qui sont éliminés avec un seuil plus strict.
Stabilisation à partir de minsup=0.2 :

Pour minsup entre 0.2 et 0.5, le nombre de règles reste faible et presque constant.
Cela indique que la majorité des règles fortement supportées ont déjà été extraites, et qu’augmenter davantage minsup n’a que peu d’impact.
Interprétation globale :
Un minsup trop faible génère une explosion combinatoire de règles, souvent non pertinentes.
Un minsup trop élevé réduit drastiquement le nombre de règles, risquant d’exclure des relations intéressantes.

"""