import pandas as pd
import numpy as np

# Load the tiers
t1 = pd.read_csv(r"d:\BOOM BAAM\Grafestt\tier1.csv")
t2 = pd.read_csv(r"d:\BOOM BAAM\Grafestt\tier2.csv")
t3 = pd.read_csv(r"d:\BOOM BAAM\Grafestt\tier3.csv")

# Basic stats
print(f"Tier 1 rows: {len(t1)}")
print(f"Tier 2 rows: {len(t2)}")
print(f"Tier 3 rows: {len(t3)}")

# Distribution comparison
print("\nClass Counts Tier 1:")
print(t1['label'].value_counts())
print("\nClass Counts Tier 2:")
print(t2['label'].value_counts())
print("\nClass Counts Tier 3:")
print(t3['label'].value_counts())

# Agreement stats
# Merging to see how often they agree
merged = t1.merge(t2, on='image_id', suffixes=('_t1', '_t2')).merge(t3, on='image_id')
merged.rename(columns={'label': 'label_t3'}, inplace=True)

all_agree = (merged['label_t1'] == merged['label_t2']) & (merged['label_t2'] == merged['label_t3'])
print(f"\nAll 3 agree: {all_agree.sum()} / {len(merged)}")

t1_t3_agree = (merged['label_t1'] == merged['label_t3'])
t2_t3_agree = (merged['label_t2'] == merged['label_t3'])
t1_t2_agree = (merged['label_t1'] == merged['label_t2'])

print(f"T1 and T3 agree: {t1_t3_agree.sum()}")
print(f"T2 and T3 agree: {t2_t3_agree.sum()}")
print(f"T1 and T2 agree: {t1_t2_agree.sum()}")

# Does T3 follow the majority? (Consensus)
def consensus(row):
    labels = [row['label_t1'], row['label_t2'], row['label_t3']]
    # Count most common
    counts = pd.Series(labels).value_counts()
    return counts.idxmax()

merged['consensus'] = merged.apply(consensus, axis=1)
t3_is_consensus = (merged['label_t3'] == merged['consensus'])
print(f"T3 matches consensus: {t3_is_consensus.sum()} / {len(merged)}")

# Are there many cases where T1 and T2 agree but T3 disagrees?
t1_t2_agree_but_t3_differs = t1_t2_agree & (merged['label_t3'] != merged['label_t1'])
print(f"T1 and T2 agree, but T3 is DIFFERENT: {t1_t2_agree_but_t3_differs.sum()}")

# Samples where T3 is unique
if t1_t2_agree_but_t3_differs.sum() > 0:
    print("\nSamples where T3 differs from T1 & T2:")
    print(merged[t1_t2_agree_but_t3_differs].head(10))
