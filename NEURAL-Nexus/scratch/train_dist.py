"""
Utility to check class distribution of training data.
"""
import pandas as pd

train = pd.read_csv(r"d:\BOOM BAAM\Grafestt\train.csv")
print("Train Class Counts:")
print(train['label'].value_counts())
