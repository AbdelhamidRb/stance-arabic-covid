import pandas as pd
import numpy as np

# Test avec keep_default_na=False
df_train = pd.read_csv("data/Mawqif_AllTargets_Train.csv", keep_default_na=False, encoding="utf-8-sig")
df_test = pd.read_csv("data/Mawqif_AllTargets_Test.csv", keep_default_na=False, encoding="utf-8-sig")

print("=== VERIFICATION DE L'ARTEFACT DE LECTURE ===")
print("Train stance value counts (avec keep_default_na=False):")
print(df_train['stance'].value_counts(dropna=False))
print("\nTest stance value counts (avec keep_default_na=False):")
print(df_test['stance'].value_counts(dropna=False))

print("\nY a-t-il de vraies valeurs manquantes dans train sur text, target, stance ?")
print(f"text vides: {(df_train['text'] == '').sum()}")
print(f"target vides: {(df_train['target'] == '').sum()}")
print(f"stance vides: {(df_train['stance'] == '').sum()}")
