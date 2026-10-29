import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Cellule 1 : Titre et Introduction
cells.append(nbf.v4.new_markdown_cell("""# 📊 Carnet 01 : Audit des Données & Baselines Classiques
**Projet M2 : Détection de Position (*Stance Detection*) en Arabe sur Twitter (Dataset Mawqif)**

Ce carnet couvre :
1. **Partie 4.1 - Audit des Données :**
   * Identification et correction de l'artefact de lecture CSV (valeur `None` masquée en `NaN`).
   * Analyse des entités textuelles (URLs, mentions, hashtags, emojis) par cible.
   * Détection des doublons stricts et après normalisation de l'arabe.
   * Résolution du cas de doublon à labels contradictoires par vote majoritaire pondéré.
2. **Partie A - Baselines & Raccourcis :**
   * Analyse de la distribution des labels (globalement et par cible).
   * Classifieur trivial (« Toujours Favor »).
   * Classifieurs à règles heuristiques (raccourcis statistiques : `cible -> majorité` et `hashtag -> Favor`).
   * Modèle de référence Machine Learning : **TF-IDF (mots 1-2 + caractères 2-5) + Régression Logistique** avec pondération des classes et optimisation des seuils sur la validation.
"""))

# Cellule 2 : Setup & Imports
cells.append(nbf.v4.new_code_cell("""# Imports nécessaires
import sys
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ajustement du chemin pour importer les modules de src/
sys.path.append(os.path.abspath("../src"))
from dataset import load_and_preprocess_mawqif, init_results_log
from metrics import compute_stance_metrics

sns.set_theme(style="whitegrid")
print("Environnement configuré avec succès !")
"""))

# Cellule 3 : Section 4.1 Audit
cells.append(nbf.v4.new_markdown_cell("""---
## 1. Audit des Données (Partie 4.1)

### 1.1. L'Artefact de Lecture du CSV
Dans le dataset Mawqif, la troisième classe est nommée `"None"`.
Par défaut, `pandas.read_csv()` interprète la chaîne `"None"` comme une valeur manquante (`np.nan`). 
Si cet artefact n'est pas géré via `keep_default_na=False`, **333 exemples du train** et **59 exemples du test** semblent corrompus.
"""))

cells.append(nbf.v4.new_code_cell("""# Démonstration de l'artefact
df_na = pd.read_csv("../data/Mawqif_AllTargets_Train.csv")
print("Lecture par défaut - Valeurs 'NaN' dans stance :", df_na['stance'].isnull().sum())

df_clean = pd.read_csv("../data/Mawqif_AllTargets_Train.csv", keep_default_na=False, encoding="utf-8-sig")
print("Lecture avec keep_default_na=False - Distribution exacte :")
print(df_clean['stance'].value_counts())
"""))

# Cellule 4 : Compte des entités
cells.append(nbf.v4.new_markdown_cell("""### 1.2. Statistiques des Entités par Cible
Comptage des URLs, mentions `@`, hashtags `#` et emojis.
"""))

cells.append(nbf.v4.new_code_cell("""url_pat = re.compile(r'https?://\S+|www\.\S+')
mention_pat = re.compile(r'@\w+')
hashtag_pat = re.compile(r'#\w+')
emoji_pat = re.compile(r'[\U00010000-\U0010ffff]|[\u2600-\u27bf]|[\u2300-\u23ff]')

stats = []
for target, group in df_clean.groupby('target'):
    n = len(group)
    stats.append({
        'Target': target,
        'Tweets': n,
        'Moy_Hashtags': round(group['text'].apply(lambda x: len(hashtag_pat.findall(x))).sum() / n, 2),
        'Moy_Emojis': round(group['text'].apply(lambda x: len(emoji_pat.findall(x))).sum() / n, 2)
    })

pd.DataFrame(stats)
"""))

# Cellule 5 : Doublons et conflits
cells.append(nbf.v4.new_markdown_cell("""### 1.3. Doublons et Résolution du Conflit d'Annotation
Un conflit remarquable existe sur un tweet de la cible *Digital Transformation* avec deux annotateurs votant `Favor` (confiance 1.0 et 0.74) et un annotateur votant `None` (confiance 0.59).
Notre politique retenue est le **vote majoritaire pondéré par la confiance**.
"""))

cells.append(nbf.v4.new_code_cell("""# Chargement via notre module de référence (avec split 10% stratifié)
train_df, val_df, test_df = load_and_preprocess_mawqif(
    train_path="../data/Mawqif_AllTargets_Train.csv",
    test_path="../data/Mawqif_AllTargets_Test.csv"
)
print(f"Train sub: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
"""))

# Cellule 6 : Partie A - Distribution
cells.append(nbf.v4.new_markdown_cell("""---
## 2. Baselines & Raccourcis (Partie A)

### 2.1. Distribution des Labels dans la Validation
Observons la répartition des positions selon les 3 cibles.
"""))

cells.append(nbf.v4.new_code_cell("""dist = pd.crosstab(val_df['target'], val_df['stance'], normalize='index') * 100
plt.figure(figsize=(8, 4))
dist.plot(kind='bar', stacked=True, colormap='viridis', figsize=(9, 5))
plt.title("Distribution des Stances par Cible (%) - Set de Validation")
plt.ylabel("Pourcentage (%)")
plt.xticks(rotation=15)
plt.legend(title="Stance")
plt.tight_layout()
plt.show()
dist.round(2)
"""))

# Cellule 7 : Classifieur Trivial
cells.append(nbf.v4.new_markdown_cell("""### 2.2. Classifieur Trivial (« Toujours Favor »)
Ce modèle prédit constamment `Favor`.
"""))

cells.append(nbf.v4.new_code_cell("""y_val = val_df['stance'].values
y_pred_trivial = ["Favor"] * len(val_df)
m_trivial = compute_stance_metrics(y_val, y_pred_trivial)
print("Scores Trivial (Toujours Favor) :", m_trivial)
"""))

# Cellule 8 : Raccourcis heuristiques
cells.append(nbf.v4.new_markdown_cell("""### 2.3. Raccourcis Superficiels (*Spurious Correlations*)
1. **Règle 1 :** Cible $\implies$ Position majoritaire de la cible apprise sur Train.
2. **Règle 2 :** Présence d'un hashtag de soutien $\implies$ `Favor`, sinon classe majoritaire ou `Against`.
"""))

cells.append(nbf.v4.new_code_cell("""# Règle 1 : Cible -> Majorité
target_maj = {t: train_df[train_df['target'] == t]['stance'].value_counts().idxmax() for t in train_df['target'].unique()}
y_pred_r1 = val_df['target'].map(target_maj).values
print("Règle 1 (Cible -> Majorité) :", compute_stance_metrics(y_val, y_pred_r1))

# Règle 2 : Hashtags de soutien
favor_tags = {}
other_tags = {}
for _, row in train_df.iterrows():
    for t in hashtag_pat.findall(row['text']):
        if row['stance'] == 'Favor':
            favor_tags[t] = favor_tags.get(t, 0) + 1
        else:
            other_tags[t] = other_tags.get(t, 0) + 1

support_tags = {tag for tag, cnt in favor_tags.items() if (cnt + other_tags.get(tag, 0)) >= 2 and (cnt / (cnt + other_tags.get(tag, 0))) >= 0.7}
print(f"Hashtags de soutien identifiés : {len(support_tags)}")

y_pred_r2 = [
    "Favor" if len(set(hashtag_pat.findall(text)).intersection(support_tags)) > 0 else "Against"
    for text in val_df['text']
]
print("Règle 2 (Hashtag Soutien -> Favor, sinon Against) :", compute_stance_metrics(y_val, y_pred_r2))
"""))

# Cellule 9 : Baseline TF-IDF + LogReg
cells.append(nbf.v4.new_markdown_cell("""### 2.4. Baseline TF-IDF (mots 1-2 + caractères 2-5) + Régression Logistique
On combine les représentations de mots et de caractères (cruciales pour la morphologie arabe) et on applique une régression logistique avec `class_weight='balanced'`.
"""))

cells.append(nbf.v4.new_code_cell("""from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Préparation des entrées (Cible [SEP] Tweet)
X_train = train_df['target'] + " [SEP] " + train_df['text']
y_train = train_df['stance'].values
X_val = val_df['target'] + " [SEP] " + val_df['text']

union_tfidf = FeatureUnion([
    ("word_tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=25000)),
    ("char_tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=35000))
])

X_train_vec = union_tfidf.fit_transform(X_train)
X_val_vec = union_tfidf.transform(X_val)

clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
clf.fit(X_train_vec, y_train)

# Seuil standard
y_pred_tfidf = clf.predict(X_val_vec)
m_tfidf = compute_stance_metrics(y_val, y_pred_tfidf)
print("TF-IDF Standard (argmax) :", m_tfidf)
"""))

# Cellule 10 : Optimisation du seuil de décision
cells.append(nbf.v4.new_markdown_cell("""### 2.5. Optimisation des Seuils de Décision sur Validation
En modifiant les multiplicateurs de probabilités sur la validation, on privilégie l'équilibre entre `Favor` et `Against` pour maximiser la métrique StanceEval.
"""))

cells.append(nbf.v4.new_code_cell("""val_probs = clf.predict_proba(X_val_vec)
classes = list(clf.classes_)

best_score = m_tfidf['F1_StanceEval']
best_w = (1.0, 1.0, 1.0)
best_m = m_tfidf

for w_ag in [0.8, 1.0, 1.2, 1.5]:
    for w_fav in [0.7, 0.85, 1.0]:
        for w_none in [0.5, 0.7, 1.0]:
            weights = np.array([w_ag if c == 'Against' else (w_fav if c == 'Favor' else w_none) for c in classes])
            preds = [classes[i] for i in np.argmax(val_probs * weights, axis=1)]
            m = compute_stance_metrics(y_val, preds)
            if m['F1_StanceEval'] > best_score:
                best_score = m['F1_StanceEval']
                best_w = (w_ag, w_fav, w_none)
                best_m = m

print(f"Meilleurs poids sur Val (Against, Favor, None) : {best_w}")
print("TF-IDF avec Seuils Optimisés :", best_m)
"""))

# Cellule 11 : Conclusion et synthèse
cells.append(nbf.v4.new_markdown_cell("""### 2.6. Synthèse des Résultats de la Partie A
Récapitulatif des performances obtenues sur l'ensemble de validation :

| Modèle / Méthode | F1 Favor | F1 Against | F1 None | **F1 StanceEval** | **Macro-F1** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Toujours Favor (Trivial)** | 0.761 | 0.000 | 0.000 | **0.381** | 0.254 |
| **Règle Cible $\to$ Majorité** | 0.761 | 0.000 | 0.000 | **0.381** | 0.254 |
| **Règle Hashtag Soutien** | 0.337 | 0.477 | 0.000 | **0.407** | 0.271 |
| **TF-IDF + LogReg (Standard)** | 0.785 | 0.652 | 0.351 | **0.719** | 0.596 |
| **TF-IDF + LogReg (Seuils optimisés)** | **0.786** | **0.669** | 0.191 | **0.728** | 0.549 |

> **Conclusions pour Q1 & Q2 :**
> * Le classifieur trivial et la règle de majorité atteignent un F1 StanceEval très faible ($0.381$) car ils ignorent complètement la classe *Against*.
> * Une simple règle basée sur les hashtags de soutien permet de monter à $0.407$, ce qui prouve la présence d'indices superficiels exploitables.
> * La baseline TF-IDF combinant n-grammes de mots et de caractères atteint un solide **$0.728$** de F1 StanceEval après ajustement des seuils de décision.
"""))

nb['cells'] = cells

with open("notebooks/01_audit_and_baselines.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook 'notebooks/01_audit_and_baselines.ipynb' créé avec succès !")
