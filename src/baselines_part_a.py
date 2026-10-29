import pandas as pd
import numpy as np
import os
import re
import sys
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Forcer encodage UTF-8 pour les prints
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from dataset import load_and_preprocess_mawqif, init_results_log
from metrics import compute_stance_metrics

def log_experiment(exp_id, section, model_name, target, seed, threshold, metrics, split="Val", details=""):
    """Ajoute les résultats dans results/results.csv."""
    results_path = "results/results.csv"
    if not os.path.exists(results_path):
        init_results_log(results_path)
    
    row = {
        "Experiment_ID": exp_id,
        "Section": section,
        "Model": model_name,
        "Target": target,
        "Seed": seed,
        "Threshold": threshold,
        "F1_Favor": metrics["F1_Favor"],
        "F1_Against": metrics["F1_Against"],
        "F1_None": metrics["F1_None"],
        "F1_StanceEval": metrics["F1_StanceEval"],
        "Macro_F1": metrics["Macro_F1"],
        "Split_Evaluated": split,
        "Details": details
    }
    df = pd.DataFrame([row])
    df.to_csv(results_path, mode='a', header=False, index=False, encoding='utf-8')
    print(f"-> Enregistré dans results.csv : {exp_id} ({model_name}) | F1_StanceEval: {metrics['F1_StanceEval']}")

def run_part_a():
    print("=" * 60)
    print("EXECUTION PARTIE A : BASELINES ET RACCOURCIS")
    print("=" * 60)
    
    # 1. Chargement des données stratifiées
    train_df, val_df, test_df = load_and_preprocess_mawqif()
    y_val = val_df['stance'].values
    
    # --- A.1 : Distribution des labels ---
    print("\n--- A.1 : DISTRIBUTION DES LABELS DANS VAL ---")
    dist_val = pd.crosstab(val_df['target'], val_df['stance'], normalize='index') * 100
    print(dist_val.round(1))
    
    # --- A.2 : Classifieur trivial « Toujours Favor » ---
    print("\n--- A.2 : CLASSIFIEUR TRIVIAL (TOUJOURS FAVOR) ---")
    y_pred_trivial = ["Favor"] * len(val_df)
    m_trivial = compute_stance_metrics(y_val, y_pred_trivial)
    print("Métriques Trivial (Val) :", m_trivial)
    log_experiment(
        exp_id="A_Trivial_Favor",
        section="Partie A",
        model_name="Toujours Favor (Trivial)",
        target="All",
        seed=42,
        threshold="N/A",
        metrics=m_trivial,
        split="Val",
        details="Prédit systématiquement Favor pour tout tweet"
    )

    # --- A.3 : Raccourci 1 (Règle : Cible -> Position majoritaire) ---
    print("\n--- A.3.1 : RACCOURCI 1 (CIBLE -> POSITION MAJORITAIRE) ---")
    # Calculer la position majoritaire par cible sur le TRAIN uniquement
    target_majorities = {}
    for target in train_df['target'].unique():
        majority_stance = train_df[train_df['target'] == target]['stance'].value_counts().idxmax()
        target_majorities[target] = majority_stance
    print("Positions majoritaires apprises sur Train :", target_majorities)
    
    y_pred_target_rule = val_df['target'].map(target_majorities).values
    m_target_rule = compute_stance_metrics(y_val, y_pred_target_rule)
    print("Métriques Règle Cible (Val) :", m_target_rule)
    log_experiment(
        exp_id="A_Rule_Target_Majority",
        section="Partie A",
        model_name="Règle Cible -> Majorité",
        target="All",
        seed=42,
        threshold="N/A",
        metrics=m_target_rule,
        split="Val",
        details=f"Majorités Train: {target_majorities}"
    )

    # --- A.3.2 : Raccourci 2 (Règle : Contient un hashtag de soutien -> Favor) ---
    print("\n--- A.3.2 : RACCOURCI 2 (HASHTAG DE SOUTIEN -> FAVOR) ---")
    # Identifier les hashtags fortement associés à Favor dans le train
    hashtag_pattern = re.compile(r'#\w+')
    favor_hashtags = {}
    other_hashtags = {}
    
    for _, row in train_df.iterrows():
        tags = hashtag_pattern.findall(row['text'])
        for t in tags:
            if row['stance'] == 'Favor':
                favor_hashtags[t] = favor_hashtags.get(t, 0) + 1
            else:
                other_hashtags[t] = other_hashtags.get(t, 0) + 1
                
    # Sélectionner les hashtags avec forte précision de soutien (au moins 3 occurrences et > 75% Favor)
    support_tags = set()
    for tag, count in favor_hashtags.items():
        other_cnt = other_hashtags.get(tag, 0)
        total = count + other_cnt
        if total >= 3 and (count / total) >= 0.75:
            support_tags.add(tag)
            
    print(f"Nombre de hashtags de soutien détectés : {len(support_tags)}")
    print("Exemples de hashtags de soutien :", list(support_tags)[:8])
    
    # Règle : si contient hashtag de soutien -> Favor, sinon règle de cible majoritaire
    def predict_hashtag_rule(row):
        tags = set(hashtag_pattern.findall(row['text']))
        if len(tags.intersection(support_tags)) > 0:
            return "Favor"
        return target_majorities[row['target']]
        
    y_pred_tag_rule = val_df.apply(predict_hashtag_rule, axis=1).values
    m_tag_rule = compute_stance_metrics(y_val, y_pred_tag_rule)
    print("Métriques Règle Hashtags (Val) :", m_tag_rule)
    log_experiment(
        exp_id="A_Rule_Hashtags_Support",
        section="Partie A",
        model_name="Règle Hashtag Soutien -> Favor",
        target="All",
        seed=42,
        threshold="N/A",
        metrics=m_tag_rule,
        split="Val",
        details=f"{len(support_tags)} hashtags retenus"
    )

    # --- A.4 : Baseline TF-IDF (mots 1-2 + car 2-5) + Régression Logistique ---
    print("\n--- A.4 : BASELINE TF-IDF + REGRESSION LOGISTIQUE ---")
    # Entrée : Cible + [SEP] + Tweet
    X_train_text = train_df['target'] + " [SEP] " + train_df['text']
    y_train = train_df['stance'].values
    
    X_val_text = val_df['target'] + " [SEP] " + val_df['text']
    
    # Feature Union : Mots (1,2) + Caractères (2,5)
    union_tfidf = FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=25000
        )),
        ("char_tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=2,
            max_features=35000
        ))
    ])
    
    print("Extraction TF-IDF (fit sur Train)...")
    X_train_tfidf = union_tfidf.fit_transform(X_train_text)
    X_val_tfidf = union_tfidf.transform(X_val_text)
    print(f"Dimension matrice TF-IDF : {X_train_tfidf.shape}")
    
    # Modèle avec class_weight='balanced'
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        C=1.0,
        random_state=42
    )
    clf.fit(X_train_tfidf, y_train)
    
    # Prédiction standard (argmax)
    y_pred_tfidf_std = clf.predict(X_val_tfidf)
    m_tfidf_std = compute_stance_metrics(y_val, y_pred_tfidf_std)
    print("Métriques TF-IDF (seuil standard argmax) :", m_tfidf_std)
    log_experiment(
        exp_id="A_TFIDF_LogReg_Standard",
        section="Partie A",
        model_name="TF-IDF (mots 1-2 + car 2-5) + LogReg",
        target="All",
        seed=42,
        threshold="argmax (1.0, 1.0, 1.0)",
        metrics=m_tfidf_std,
        split="Val",
        details="class_weight=balanced, C=1.0"
    )
    
    # --- A.5 : Optimisation des seuils sur la Validation ---
    print("\n--- A.5 : OPTIMISATION DU SEUIL DE DECISION SUR VAL ---")
    val_probs = clf.predict_proba(X_val_tfidf)
    classes = list(clf.classes_) # ['Against', 'Favor', 'None']
    print(f"Ordre des classes : {classes}")
    
    best_f1_stance = m_tfidf_std['F1_StanceEval']
    best_weights = (1.0, 1.0, 1.0)
    best_metrics = m_tfidf_std
    
    # Recherche par grille de multiplicateurs de probabilités sur Val
    # p_pondéré = P * w, puis argmax
    for w_against in [0.8, 1.0, 1.2, 1.5, 1.8]:
        for w_favor in [0.7, 0.85, 1.0, 1.15]:
            for w_none in [0.5, 0.7, 1.0, 1.3]:
                w_vec = np.array([
                    w_against if c == 'Against' else (w_favor if c == 'Favor' else w_none)
                    for c in classes
                ])
                weighted_probs = val_probs * w_vec
                preds_idx = np.argmax(weighted_probs, axis=1)
                preds = [classes[i] for i in preds_idx]
                
                m = compute_stance_metrics(y_val, preds)
                if m['F1_StanceEval'] > best_f1_stance:
                    best_f1_stance = m['F1_StanceEval']
                    best_weights = (w_against, w_favor, w_none)
                    best_metrics = m
                    
    print(f"Poids optimaux sur Val (Against, Favor, None) : {best_weights}")
    print("Métriques TF-IDF (seuils optimisés sur Val) :", best_metrics)
    log_experiment(
        exp_id="A_TFIDF_LogReg_TunedThreshold",
        section="Partie A",
        model_name="TF-IDF + LogReg (Seuils optimisés)",
        target="All",
        seed=42,
        threshold=f"weights={best_weights}",
        metrics=best_metrics,
        split="Val",
        details=f"Gain F1_StanceEval: +{round(best_metrics['F1_StanceEval'] - m_tfidf_std['F1_StanceEval'], 4)}"
    )
    
    print("\n" + "=" * 60)
    print("PARTIE A TERMINEE AVEC SUCCES !")
    print("=" * 60)

if __name__ == "__main__":
    run_part_a()
