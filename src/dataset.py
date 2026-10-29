import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def load_and_preprocess_mawqif(
    train_path="data/Mawqif_AllTargets_Train.csv",
    test_path="data/Mawqif_AllTargets_Test.csv",
    val_size=0.10,
    random_state=42,
    resolve_conflicts=True
):
    """
    Charge les données Mawqif en corrigeant l'artefact de lecture CSV ('None' interprété comme NaN),
    traite les doublons/conflits selon la politique justifiée,
    et crée un split train/val stratifié à 10% par (target x stance).
    """
    # 1. Chargement sans transformer 'None' en NaN
    train_df = pd.read_csv(train_path, keep_default_na=False, encoding="utf-8-sig")
    test_df = pd.read_csv(test_path, keep_default_na=False, encoding="utf-8-sig")

    # Vérification d'intégrité
    assert train_df['stance'].isnull().sum() == 0, "Erreur: valeurs NaN résiduelles dans train!"
    assert test_df['stance'].isnull().sum() == 0, "Erreur: valeurs NaN résiduelles dans test!"

    # 2. Résolution des doublons contradictoires dans le Train
    if resolve_conflicts:
        # Trier par confiance décroissante pour privilégier l'annotation la plus sûre
        if 'stance:confidence' in train_df.columns:
            train_df = train_df.sort_values(by='stance:confidence', ascending=False)
        # Supprimer les doublons stricts (text, target) en gardant la première occurrence (plus haute confiance)
        train_df = train_df.drop_duplicates(subset=['text', 'target'], keep='first').reset_index(drop=True)

    # 3. Stratification par Target x Stance
    train_df['strat_key'] = train_df['target'].astype(str) + "___" + train_df['stance'].astype(str)

    train_sub, val_df = train_test_split(
        train_df,
        test_size=val_size,
        stratify=train_df['strat_key'],
        random_state=random_state
    )

    # Nettoyage de la clé temporaire
    train_sub = train_sub.drop(columns=['strat_key']).reset_index(drop=True)
    val_df = val_df.drop(columns=['strat_key']).reset_index(drop=True)

    return train_sub, val_df, test_df

def init_results_log(filepath="results/results.csv"):
    """Initialise le journal de résultats s'il n'existe pas déjà."""
    columns = [
        "Experiment_ID",
        "Section",
        "Model",
        "Target",
        "Seed",
        "Threshold",
        "F1_Favor",
        "F1_Against",
        "F1_None",
        "F1_StanceEval",
        "Macro_F1",
        "Split_Evaluated",
        "Details"
    ]
    if not os.path.exists(filepath):
        pd.DataFrame(columns=columns).to_csv(filepath, index=False, encoding="utf-8")
        print(f"Journal initialisé : {filepath}")
    else:
        print(f"Journal existant trouvé : {filepath}")

if __name__ == "__main__":
    train_sub, val_df, test_df = load_and_preprocess_mawqif()
    print("=== CHARGEMENT ET SPLIT STRATIFIE 10% ===")
    print(f"Train sub-split: {len(train_sub)} exemples")
    print(f"Validation split: {len(val_df)} exemples")
    print(f"Test split (scellé): {len(test_df)} exemples")
    
    print("\nDistribution dans Train sub:")
    print(pd.crosstab(train_sub['target'], train_sub['stance'], normalize='index').round(3))
    
    print("\nDistribution dans Val:")
    print(pd.crosstab(val_df['target'], val_df['stance'], normalize='index').round(3))

    init_results_log()
