import os
import pandas as pd
import numpy as np
import re

train_path = "data/Mawqif_AllTargets_Train.csv"
test_path = "data/Mawqif_AllTargets_Test.csv"

print("--- TENTATIVE DE LECTURE STANDARD AVEC PANDAS ---")
try:
    df_train = pd.read_csv(train_path)
    print("Succès standard train!")
    print("Shape:", df_train.shape)
    print("Colonnes:", df_train.columns.tolist())
    print("\nValeurs manquantes par colonne dans Train:")
    print(df_train.isnull().sum())
    print("\nValeurs uniques de 'stance':")
    print(df_train['stance'].value_counts(dropna=False))
except Exception as e:
    print("Erreur lecture standard:", e)

# Inspection brute ligne par ligne pour détecter des artefacts (séparateurs, guillemets, encodage, sauts de ligne)
with open(train_path, "r", encoding="utf-8", errors="replace") as f:
    raw_lines = f.readlines()

print(f"\nNombre de lignes brutes dans le fichier: {len(raw_lines)}")
header = raw_lines[0].strip()
print("Header brut:", header)
