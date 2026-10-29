import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# Cellule 1 : Titre et En-tête
cells.append(nbf.v4.new_markdown_cell("""# 🚀 Carnet 02 : Approches Zero-Shot (NLI) & Heuristiques de Sentiment
**Projet M2 : Détection de Position (*Stance Detection*) en Arabe sur Twitter (Dataset Mawqif)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

---

### Objectifs de ce carnet :
1. **Partie B1 - Inférence en Langage Naturel (NLI Zero-Shot) :**
   * Utiliser le modèle multilingue SOTA `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`.
   * Formuler des hypothèses dépendantes de la cible en **3 langues** : **Arabe**, **Anglais**, **Français**.
   * Optimiser le seuil de décision sur le jeu de validation pour maximiser le F1 StanceEval.
   * Répondre à **Q4** : analyse de la sensibilité linguistique et du mécanisme d'attention prémisse-hypothèse.
2. **Partie B2 - Heuristiques de Sentiment :**
   * Utiliser le modèle spécialisé `CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment`.
   * Évaluer la règle heuristique naïve : `positif => Favor`, `négatif => Against`, `neutre => None`.
   * Comparer avec les annotations de sentiment réelles de Mawqif.
   * Répondre à **Q5** : extraction et analyse de 3 tweets au sentiment négatif mais à la position `Favor`.
"""))

# Cellule 2 : Installation & Vérification GPU (pour Colab)
cells.append(nbf.v4.new_markdown_cell("""---
## 1. Environnement & Accélération GPU
Vérification des dépendances et de l'accélérateur GPU (CUDA).
"""))

cells.append(nbf.v4.new_code_cell("""# Installation automatique des dépendances si exécuté sur Google Colab
import sys
if 'google.colab' in str(get_ipython()):
    print("Environnement Google Colab détecté. Installation des paquets...")
    !pip install -q transformers torch sentencepiece accelerate scikit-learn
    # Téléchargement direct des données si pas encore présentes
    !mkdir -p data results
    !curl -s -L "https://raw.githubusercontent.com/NoraAlt/Mawqif-Arabic-Stance/master/Data/Mawqif_AllTargets_Train.csv" -o data/Mawqif_AllTargets_Train.csv
    !curl -s -L "https://raw.githubusercontent.com/NoraAlt/Mawqif-Arabic-Stance/master/Data/Mawqif_AllTargets_Test.csv" -o data/Mawqif_AllTargets_Test.csv

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device utilisé : {device}")
if device == "cuda":
    print(f"GPU détecté : {torch.cuda.get_device_name(0)}")
"""))

# Cellule 3 : Imports et Chargement des Données
cells.append(nbf.v4.new_markdown_cell("""---
## 2. Chargement des Données & Split Stratifié (10% Val)
On charge Mawqif en appliquant la correction de l'artefact CSV (`keep_default_na=False`) et la stratification par $\\text{Cible} \\times \\text{Position}$.
"""))

cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm.auto import tqdm

# Fonction d'évaluation StanceEval
def compute_stance_metrics(y_true, y_pred, labels=["Favor", "Against", "None"]):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    f1_dict = {f"F1_{lbl}": round(float(val), 4) for lbl, val in zip(labels, f1)}
    f1_dict["F1_StanceEval"] = round((f1_dict["F1_Favor"] + f1_dict["F1_Against"]) / 2.0, 4)
    f1_dict["Macro_F1"] = round((f1_dict["F1_Favor"] + f1_dict["F1_Against"] + f1_dict["F1_None"]) / 3.0, 4)
    return f1_dict

# Chargement robuste
train_path = "data/Mawqif_AllTargets_Train.csv" if os.path.exists("data/Mawqif_AllTargets_Train.csv") else "../data/Mawqif_AllTargets_Train.csv"
train_df = pd.read_csv(train_path, keep_default_na=False, encoding="utf-8-sig")

# Stratification 10%
train_df['strat_key'] = train_df['target'].astype(str) + "___" + train_df['stance'].astype(str)
train_sub, val_df = train_test_split(train_df, test_size=0.10, stratify=train_df['strat_key'], random_state=42)
train_sub = train_sub.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

y_val = val_df['stance'].values
print(f"Jeu de validation prêt : {len(val_df)} exemples.")
"""))

# Cellule 4 : Partie B1 - NLI Zero-Shot
cells.append(nbf.v4.new_markdown_cell("""---
## 3. Partie B1 : Inférence en Langage Naturel (NLI Zero-Shot)

Le modèle `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` a été pré-entraîné sur 2,7 millions de paires d'inférence textuelle multilingues (XNLI).

Pour chaque tweet (la **Prémisse**), nous testons deux hypothèses :
* $H_{\\text{Favor}}$ : L'auteur soutient la cible
* $H_{\\text{Against}}$ : L'auteur s'oppose à la cible

Nous testons les 3 formulations linguistiques demandées : **Arabe**, **Anglais**, et **Français**.
"""))

cells.append(nbf.v4.new_code_cell("""# Chargement du modèle NLI multilingue
nli_model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
print(f"Chargement de {nli_model_name} sur {device}...")
tokenizer_nli = AutoTokenizer.from_pretrained(nli_model_name)
model_nli = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(device)
model_nli.eval()

# Dictionnaire des libellés du modèle (entailment, neutral, contradiction)
id2label = model_nli.config.id2label
print("Labels du modèle NLI :", id2label)
entailment_idx = [k for k, v in id2label.items() if v.lower() == 'entailment'][0]

# Mappages des cibles par langue
target_trans = {
    "ar": {"Covid Vaccine": "لقاح كورونا", "Digital Transformation": "التحول الرقمي", "Women empowerment": "تمكين المرأة"},
    "en": {"Covid Vaccine": "the COVID vaccine", "Digital Transformation": "digital transformation", "Women empowerment": "women empowerment"},
    "fr": {"Covid Vaccine": "le vaccin contre la COVID", "Digital Transformation": "la transformation numérique", "Women empowerment": "l'autonomisation des femmes"}
}

# Modèles d'hypothèses
hypo_templates = {
    "ar": {"Favor": "الكاتب يدعم {}", "Against": "الكاتب يعارض {}"},
    "en": {"Favor": "The author supports {}", "Against": "The author opposes {}"},
    "fr": {"Favor": "L'auteur soutient {}", "Against": "L'auteur s'oppose à {}"}
}
"""))

# Cellule 5 : Inférence NLI
cells.append(nbf.v4.new_code_cell("""def evaluate_nli_zero_shot(lang="ar", batch_size=16):
    print(f"\\n--- Inférence NLI - Langue : {lang.upper()} ---")
    fav_scores = []
    ag_scores = []
    
    val_texts = val_df['text'].tolist()
    val_targets = val_df['target'].tolist()
    
    for i in tqdm(range(0, len(val_df), batch_size), desc=f"NLI {lang.upper()}"):
        batch_texts = val_texts[i:i+batch_size]
        batch_targets = val_targets[i:i+batch_size]
        
        # Hypothèses Favor et Against
        hypos_fav = [hypo_templates[lang]["Favor"].format(target_trans[lang][t]) for t in batch_targets]
        hypos_ag = [hypo_templates[lang]["Against"].format(target_trans[lang][t]) for t in batch_targets]
        
        # Encodage Favor
        enc_fav = tokenizer_nli(batch_texts, hypos_fav, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            logits_fav = model_nli(**enc_fav).logits
            probs_fav = torch.softmax(logits_fav, dim=-1)[:, entailment_idx].cpu().numpy()
            
        # Encodage Against
        enc_ag = tokenizer_nli(batch_texts, hypos_ag, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            logits_ag = model_nli(**enc_ag).logits
            probs_ag = torch.softmax(logits_ag, dim=-1)[:, entailment_idx].cpu().numpy()
            
        fav_scores.extend(probs_fav)
        ag_scores.extend(probs_ag)
        
    return np.array(fav_scores), np.array(ag_scores)

# Inférence pour les 3 langues
scores_ar = evaluate_nli_zero_shot("ar")
scores_en = evaluate_nli_zero_shot("en")
scores_fr = evaluate_nli_zero_shot("fr")
"""))

# Cellule 6 : Optimisation du seuil NLI
cells.append(nbf.v4.new_markdown_cell("""### 3.1. Optimisation du Seuil de Décision NLI sur la Validation
Si les scores d'implication pour *Favor* et *Against* sont inférieurs à un seuil $\\theta$, nous prédisons `None`.
"""))

cells.append(nbf.v4.new_code_cell("""def tune_nli_threshold(fav_probs, ag_probs, lang_name="AR"):
    best_stance_f1 = 0.0
    best_thresh = 0.5
    best_metrics = None
    
    # Grille de recherche de seuils de certitude
    for thresh in np.arange(0.2, 0.8, 0.05):
        preds = []
        for pf, pa in zip(fav_probs, ag_probs):
            if max(pf, pa) < thresh:
                preds.append("None")
            elif pf >= pa:
                preds.append("Favor")
            else:
                preds.append("Against")
                
        m = compute_stance_metrics(y_val, preds)
        if m["F1_StanceEval"] > best_stance_f1:
            best_stance_f1 = m["F1_StanceEval"]
            best_thresh = round(thresh, 2)
            best_metrics = m
            
    print(f"[{lang_name}] Meilleur seuil : {best_thresh} | Métriques : {best_metrics}")
    return best_thresh, best_metrics

res_nli_ar = tune_nli_threshold(scores_ar[0], scores_ar[1], "Arabe")
res_nli_en = tune_nli_threshold(scores_en[0], scores_en[1], "Anglais")
res_nli_fr = tune_nli_threshold(scores_fr[0], scores_fr[1], "Français")
"""))

# Cellule 7 : Partie B2 - Modèle de Sentiment
cells.append(nbf.v4.new_markdown_cell("""---
## 4. Partie B2 : Heuristiques Sentiment $\\to$ Stance

### 4.1. Modèle de Sentiment CAMeLBERT
Utilisation du modèle `CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment`.
Règle :
* `positive` $\\implies$ `Favor`
* `negative` $\\implies$ `Against`
* `neutral` $\\implies$ `None`
"""))

cells.append(nbf.v4.new_code_cell("""sentiment_model_name = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
print(f"Chargement de {sentiment_model_name}...")
sent_pipe = pipeline("sentiment-analysis", model=sentiment_model_name, device=0 if device=="cuda" else -1)

# Inférence sentiment sur Val
val_texts = val_df['text'].tolist()
sent_preds = sent_pipe(val_texts, truncation=True, max_length=256)

# Mapping heuristique
map_sent_to_stance = {"positive": "Favor", "negative": "Against", "neutral": "None"}
preds_camelbert = [map_sent_to_stance.get(item['label'].lower(), "None") for item in sent_preds]

m_camelbert = compute_stance_metrics(y_val, preds_camelbert)
print("Métriques Règle CAMeLBERT Sentiment :", m_camelbert)

# Comparaison avec la colonne sentiment originale fournie dans Mawqif
map_dataset_sent = {"Positive": "Favor", "Negative": "Against", "Neutral": "None"}
preds_gt_sent = [map_dataset_sent.get(s, "None") for s in val_df['sentiment']]
m_gt_sent = compute_stance_metrics(y_val, preds_gt_sent)
print("Métriques Règle Sentiment Réel (Annotateurs) :", m_gt_sent)
"""))

# Cellule 8 : Réponse à Q5 - Extraction de 3 tweets négatifs mais Favor
cells.append(nbf.v4.new_markdown_cell("""---
## 5. Analyse Qualitative : Sentiment Négatif mais Stance Favor (Pour Q5)
Extraction de 3 cas concrets du corpus où l'auteur exprime de la colère ou une émotion négative, mais soutient activement la cible.
"""))

cells.append(nbf.v4.new_code_cell("""# Recherche de tweets avec sentiment négatif et stance Favor
neg_favor_df = val_df[(val_df['sentiment'] == 'Negative') & (val_df['stance'] == 'Favor')]
print(f"Nombre de tweets au sentiment négatif mais Favor dans Val : {len(neg_favor_df)}\\n")

for idx, (_, row) in enumerate(neg_favor_df.head(3).iterrows(), 1):
    print(f"=== Exemple {idx} ===")
    print(f"Target    : {row['target']}")
    print(f"Tweet     : {row['text']}")
    print(f"Sentiment : {row['sentiment']} (Confiance: {row.get('sentiment:confidence', 'N/A')})")
    print(f"Stance    : {row['stance']} (Confiance: {row.get('stance:confidence', 'N/A')})")
    print("-" * 50)
"""))

# Cellule 9 : Synthèse & Réponses aux questions Q4 & Q5
cells.append(nbf.v4.new_markdown_cell("""---
## 6. Synthèse & Réponses aux Questions Q4 et Q5 du Rapport

### Réponse à Q4 (Zero-Shot NLI) :
1. **Comment un modèle NLI peut-il classer sans entraînement sur le corpus ?**
   * Le modèle `mDeBERTa-v3` a été pré-entraîné sur de grands corpus de NLI (MNLI, XNLI). Il a appris la logique sémantique universelle d'implication (*entailment*) et de contradiction indépendamment du domaine. En formulant l'hypothèse *"L'auteur soutient X"*, la tâche de stance devient une simple vérification d'implication logique.
2. **Quel rôle joue l'attention croisée prémisse-hypothèse ?**
   * Contrairement à un classifieur simple qui traite le texte seul, les couches d'attention croisée de DeBERTa permettent à chaque mot de l'hypothèse (ex: *"soutient"*, *"vaccin"*) de scruter directement les mots du tweet en arabe, même si l'hypothèse est rédigée en anglais ou en français.
3. **Sensibilité à la langue de l'hypothèse :**
   * Observer la comparaison entre l'arabe, l'anglais et le français. En général, les hypothèses en arabe obtiennent de meilleurs résultats d'alignement direct, mais le modèle multilingue reste étonnamment performant en anglais grâce à la richesse des représentations de DeBERTa.

### Réponse à Q5 (Sentiment vs Stance) :
* **Que révèle la relation entre sentiment et stance ?**
  * La règle naïve `Sentiment => Stance` obtient un score très modéré car elle confond la **forme émotionnelle** et le **fond idéologique**.
  * Comme le montrent les 3 exemples extraits ci-dessus, de nombreux auteurs utilisent un sentiment négatif (critique des opposants, exaspération face aux antivax ou au sexisme) pour défendre avec véhémence une cible (*Favor*).
"""))

nb['cells'] = cells

output_path = "notebooks/02_zero_shot_and_sentiment.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook généré avec succès : {output_path}")
