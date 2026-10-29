# Guide Méthodologique & Concepts Clés
## Projet M2 : Détection de Position (*Stance Detection*) en Arabe sur Twitter (Dataset Mawqif)

Ce document rassemble l'ensemble des définitions théoriques, des concepts mathématiques, des architectures neuronales et du protocole expérimental requis pour le projet. Il sert de document de référence pour la réalisation du code, la rédaction du rapport et la préparation de la soutenance (slides).

---

## Sommaire
1. [La Tâche & Les Spécificités Linguistiques](#1-la-tâche--les-spécificités-linguistiques)
2. [Les Métriques d'Évaluation](#2-les-métriques-dévaluation)
3. [Protocole Expérimental & Rigueur Statistique](#3-protocole-expérimental--rigueur-statistique)
4. [Modèles & Architectures (Du Trivial au SOTA)](#4-modèles--architectures-du-trivial-au-sota)
5. [Tests de Robustesse & Analyse d'Erreurs](#5-tests-de-robustesse--analyse-derreurs)
6. [Décision de Mise en Production (Compromis)](#6-décision-de-mise-en-production-compromis)
7. [Matrice de Correspondance avec les Questions du Rapport (Q1 à Q12)](#7-matrice-de-correspondance-avec-les-questions-du-rapport-q1-à-q12)

---

## 1. La Tâche & Les Spécificités Linguistiques

### 1.1. Définition de la Détection de Position (*Stance Detection*)
La détection de position consiste à déterminer l'attitude ou la posture de l'auteur d'un texte vis-à-vis d'une **cible prédéfinie (*Target*)**.
* **Classes cibles :**
  * `Favor` (Pour / En faveur)
  * `Against` (Contre / Opposé)
  * `None` (Neutre, indifférent ou sans opinion explicite envers la cible)
* **Les 3 cibles du dataset Mawqif :**
  1. Le vaccin contre la COVID-19 (`Covid-19 vaccine`)
  2. La transformation numérique (`Digital Transformation`)
  3. L'autonomisation des femmes (`Women Empowerment`)

### 1.2. Distinction Fondamentale : *Stance* (Position) vs *Sentiment*
> **Règle d'or :** Le sentiment mesure l'état affectif ou la valence émotionnelle du texte ; la stance mesure l'alignement idéologique envers une cible précise.

* **Exemple concret :**
  * *Tweet :* « يلعن كورونا والمرض، تعبنا من الحجر، روحوا خذوا اللقاح وخلصونا ! » *(Maudit soit le Covid et la maladie, nous sommes épuisés par le confinement, allez vous faire vacciner et libérez-nous !)*
  * **Sentiment :** Très négatif (colère, exaspération, vocabulaire péjoratif : *maudit*, *maladie*, *épuisés*).
  * **Stance (Cible = Vaccin) :** `Favor` (l'auteur appelle activement à la vaccination).
* **Conséquence :** Une règle simpliste `Positif => Favor` et `Négatif => Against` échoue sur les cas où une émotion négative sert à défendre une cause.

### 1.3. La Dépendance à la Cible
Un texte n'a pas de position intrinsèque absolue : sa position n'a de sens que **relativement à une cible**.
* Un tweet soutenant le télétravail pour les femmes peut être `Favor` pour *Women Empowerment*, `Favor` pour *Digital Transformation*, mais `None` pour le *Vaccin Covid*.
* Le modèle doit donc prendre en entrée le couple **(Cible, Tweet)** et non le tweet isolé.

### 1.4. Les Spécificités de l'Arabe sur Twitter
1. **Diglossie et Dialectes :** Les tweets mélangent l'Arabe Standard Moderne (MSA) et divers dialectes régionaux (Golfe, Levant, Égypte, Maghreb) sans orthographe normalisée.
2. **Morphologie Agglutinante :** Articles, prépositions, pronoms conjoints et conjonctions sont directement attachés au mot (ex. « وسيعلمون » = *et ils sauront*).
3. **Absence de Diacritiques (*Tashkeel*) :** L'omission des voyelles courtes crée de nombreuses homographies levées uniquement par le contexte.

---

## 2. Les Métriques d'Évaluation

### 2.1. Précision, Rappel et F1-Score
Pour une classe $c \in \{\text{Favor}, \text{Against}, \text{None}\}$ :
* **Précision ($P_c$) :** Proportion de prédictions correctes parmi tous les exemples prédits comme appartenant à la classe $c$ :
  $$P_c = \frac{TP_c}{TP_c + FP_c}$$
* **Rappel ($R_c$) :** Proportion d'exemples de la classe $c$ correctement identifiés par le modèle :
  $$R_c = \frac{TP_c}{TP_c + FN_c}$$
* **F1-Score ($F_{1,c}$) :** Moyenne harmonique entre précision et rappel :
  $$F_{1,c} = 2 \cdot \frac{P_c \cdot R_c}{P_c + R_c}$$

### 2.2. Macro-F1 (3 classes)
Moyenne non pondérée des F1-scores des 3 classes :
$$\text{Macro-F1} = \frac{F_{1,\text{Favor}} + F_{1,\text{Against}} + F_{1,\text{None}}}{3}$$
* **Pourquoi pas l'Accuracy ?** L'Accuracy est fortement biaisée lorsque les classes sont déséquilibrées (un modèle prédisant toujours la classe majoritaire obtiendrait une fausse impression de haute performance). Le Macro-F1 attribue le même poids à chaque classe indépendamment de son effectif.

### 2.3. Métrique Officielle StanceEval : F1 Moyen (Favor & Against)
$$\text{F1}_{\text{StanceEval}} = \frac{F_{1,\text{Favor}} + F_{1,\text{Against}}}{2}$$
* **Pourquoi exclure la classe `None` ?**
  Dans la tradition des benchmarks de détection de position (SemEval-2016, StanceEval-2024), l'objectif principal est de mesurer la capacité d'un système à arbitrer le débat d'opinion (*Pour* vs *Contre*). La classe `None` regroupe souvent des textes hors-sujet ou ambigus ; l'inclure dans la métrique principale risquerait de masquer de mauvaises performances sur les classes d'intérêt.

---

## 3. Protocole Expérimental & Rigueur Statistique

### 3.1. Partitionnement Stratifié (Split 10% Validation)
* Le jeu de données Train ($3\,502$ tweets) est divisé en **Train interne ($90\%$)** et **Validation ($10\%$)**.
* **Stratification par $\text{Cible} \times \text{Position}$ :** On garantit que chaque couple possible (ex. *Vaccin $\times$ Against*, *Femmes $\times$ Favor*) est représenté dans les mêmes proportions dans les deux sous-ensembles.

### 3.2. Le Principe du "Test Scellé" & Prévention de la Fuite de Données (*Data Leakage*)
* Le jeu de **Test ($619$ tweets)** ne doit être manipulé et évalué **qu'en toute fin de chaîne (Partie F)**.
* **Aucun choix d'hyperparamètre, seuil ou vocabulaire** ne doit être optimisé sur le test.
* Toute transformation dépendante des données (vocabulaire, calculs IDF de TF-IDF, normalisation) doit être apprise (`fit`) sur le Train uniquement, puis appliquée (`transform`) sur Val et Test.

### 3.3. Gestion du Déséquilibre des Classes
1. **Perte Pondérée (*Weighted Cross-Entropy Loss*) :** On applique un poids inversement proportionnel à la fréquence de la classe :
   $$w_c = \frac{N}{K \cdot N_c}$$
   *(où $N$ est le nombre total d'exemples, $K$ le nombre de classes, $N_c$ le nombre d'exemples de la classe $c$).*
2. **Sur-échantillonnage (*Oversampling*) :** Duplication aléatoire d'exemples des classes minoritaires (réalisée **uniquement sur le Train** après le split de validation pour éviter toute fuite).
3. **Optimisation des Seuils de Décision (*Threshold Tuning*) :** Ajustement des seuils de probabilité sur le split de validation pour maximiser la métrique StanceEval plutôt que d'utiliser la règle naive $\text{argmax}$.

### 3.4. Validation Statistique Avancée
* **Intervalles de Confiance à 95% par Bootstrap ($B = 1000$) :**
  Tirage avec remise de 1000 sous-échantillons du jeu de test. Calcul de la métrique sur chaque réplique pour déduire les percentiles 2.5% et 97.5%.
* **Test de McNemar :**
  Test statistique d'hypothèse pour données appariées comparant deux classifieurs $M_1$ et $M_2$ sur leur table de contingence des erreurs :
  $$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$$
  *(où $b$ est le nombre d'exemples réussis par $M_1$ et échoués par $M_2$, et $c$ l'inverse).* Si $p < 0.05$, la différence de performance est statistiquement significative.
* **Prior Shift (Décalage de distribution de labels) :**
  Différence entre la distribution marginale $P(Y)$ sur le jeu d'entraînement/validation et celle observée sur le test.

---

## 4. Modèles & Architectures (Du Trivial au SOTA)

### 4.1. Baselines & Analyse des Raccourcis (Partie A)
* **Classifieur Trivial :** Prédit systématiquement la classe `Favor`. Établit la borne inférieure absolue.
* **TF-IDF + Régression Logistique :**
  * N-grammes de mots (1 à 2) : capture le vocabulaire lexical et les expressions figées.
  * N-grammes de caractères (2 à 5) : capture les sous-unités morphologiques, racines sémitiques et variations dialectales/orthographiques.
* **Raccourcis Superficiels (*Spurious Correlations*) :**
  Règles heuristiques directes (ex. association cible $\to$ label majoritaire, ou détection de hashtags explicites de soutien). Elles révèlent si le corpus contient des biais exploitables sans véritable compréhension sémantique.

### 4.2. Inférence en Langage Naturel Zero-Shot (*NLI Zero-Shot* - Partie B)
* **Principe du NLI :** Déterminer la relation logique entre une **Prémisse** (le tweet) et une **Hypothèse** :
  * *Entailment* (Implication) $\implies$ `Favor`
  * *Contradiction* $\implies$ `Against`
  * *Neutral* $\implies$ `None`
* **Modèle utilisé :** `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`.
* **Mécanisme :** L'attention croisée bidirectionnelle permet au modèle d'aligner chaque élément de l'hypothèse avec les indices du tweet, sans avoir été entraîné sur la tâche de stance spécifique.

### 4.3. Réseaux Neuronaux Conçus "From Scratch" (Partie C)

#### Vocabulaire & Taux OOV (*Out-Of-Vocabulary*)
* Construction du vocabulaire sur le Train (mots ayant une fréquence $\ge 2$).
* Les mots absents sont associés au token spécial `<UNK>`. Le taux OOV mesure la proportion de tokens inconnus dans Val et Test.

#### Embeddings Statiques
* **Aléatoires :** Poids initialisés selon une loi normale, appris de zéro.
* **AraVec (Twitter) :** Modèle Word2Vec pré-entraîné sur des millions de tweets arabes.
  * *Mode figé (frozen) :* Les vecteurs restent constants.
  * *Mode dégelé (fine-tuned) :* Les vecteurs sont ajustés par rétropropagation.

#### Modèles Neuronaux Spécifiques
1. **C1 - BiLSTM (Bidirectional LSTM) :**
   Traite la séquence dans les deux sens (droite-gauche et gauche-droite). Concatène les états cachés $\vec{h}_t$ et $\overleftarrow{h}_t$ pour capturer le contexte passé et futur.
2. **C2 - BiLSTM + Mécanisme d'Attention :**
   Au lieu de ne conserver que le dernier état caché, le mécanisme d'attention calcule un score d'alignement pour chaque pas de temps $t$ :
   $$\alpha_t = \frac{\exp(e_t)}{\sum_{k} \exp(e_k)} \quad \text{avec} \quad e_t = v^\top \tanh(W h_t + b)$$
   Le vecteur de contexte $c = \sum_t \alpha_t h_t$ pondère dynamiquement les mots clés du tweet.
3. **C3 - Mini-Encodeur Transformer (2 couches, $d_{model}=128$, 4 têtes) :**
   Repose exclusivement sur l'auto-attention (*Self-Attention*).
   * **Encodage Positionnel (*Positional Encoding*) :** Vecteurs trigonométriques injectant l'ordre des mots dans une architecture par nature invariante par permutation.
   * **Ablation :** Suppression de l'encodage positionnel pour mesurer empiriquement la sensibilité du modèle à la syntaxe séquentielle.
4. **C4 - Char-CNN :**
   Applique des bancs de convolutions 1D sur les séquences de caractères. Extrêmement robuste aux dialectes non standardisés, abréviations et fautes de frappe.

### 4.4. Modèles Pré-entraînés (PLMs) & Fine-Tuning (Parties D et E)

#### AraBERT vs MARBERTv2
* **`aubmindlab/bert-base-arabertv02-twitter` :** Modèle BERT pré-entraîné principalement sur de l'arabe moderne et adapté sur un corpus Twitter.
* **`UBC-NLP/MARBERTv2` :** Pré-entraîné sur 1 milliard de tweets arabes diversifiés ; modèle de référence pour les dialectes et l'arabe conversationnel informel.

#### Format d'Entrée en Paire
Le texte est formaté avec des tokens spéciaux pour conditionner la représentation :
`[CLS] Cible [SEP] Tweet [SEP]`
L'état de sortie du token `[CLS]` capture l'interaction croisée entre la cible et le texte grâce aux couches de *Self-Attention*.

#### Apprentissage Multi-Tâche (*Multi-Task Learning* - E2)
Le réseau partage son corps Transformer et dispose de plusieurs têtes de classification indépendantes en sortie :
1. Tête 1 : Position (*Favor / Against / None*)
2. Tête 2 : Sentiment (*Positive / Negative / Neutral*)
3. Tête 3 : Sarcasme (*Sarcastic / Not Sarcastic*)

* **Fonction de Perte Totale :**
  $$\mathcal{L}_{\text{globale}} = \mathcal{L}_{\text{stance}} + \lambda_1 \mathcal{L}_{\text{sentiment}} + \lambda_2 \mathcal{L}_{\text{sarcasme}}$$
* **Objectif :** L'apprentissage de représentations auxiliaires riches (comprendre l'ironie et l'état émotionnel) sert de régularisateur et guide la décision de position.

---

## 5. Tests de Robustesse & Analyse d'Erreurs

### 5.1. Évaluation Cible Non Vue (*Cross-Target Stance Detection* - F2)
* Entraînement sur 2 cibles et évaluation sur la 3ème cible exclue (3 permutations complètes).
* Mesure la capacité du modèle à apprendre une sémantique générale de l'argumentation plutôt qu'une mémorisation lexicale propre à un sujet.

### 5.2. Test d'Ablation de la Cible (F4)
* Suppression explicite des mentions de la cible (mots-clés, hashtags) dans les tweets du test initialement bien classés.
* Mesure la chute de performance pour vérifier si le modèle exploite de véritables arguments implicites ou de simples corrélations lexicales directes.

### 5.3. Typologie des Erreurs (Taxonomie des 30 Cas - F6)
L'inspection qualitative manuelle de 30 erreurs est catégorisée selon :
1. **Position Implicite :** Opinion exprimée via des proverbes, métaphores ou sous-entendus sans lexique direct.
2. **Ironie / Sarcasme :** Formulations louangeuses employées à des fins de moquerie ou de rejet.
3. **Sentiment Opposé :** Valence émotionnelle inverse de la position idéologique.
4. **Cible Mal Identifiée :** Tweets traitant de sujets connexes où la cible réelle n'est pas le focus de l'auteur.
5. **Bruit d'Annotation :** Erreurs ou ambiguïtés intrinsèques aux étiquettes humaines du corpus d'origine.

---

## 6. Décision de Mise en Production (Compromis)

Le choix final du modèle pour un environnement de production repose sur une analyse multidimensionnelle :
* **Efficacité Algorithmique :** F1 StanceEval et Macro-F1.
* **Robustesse :** Résistance à la suppression de cibles et généralisation cross-target.
* **Coût Matériel & Empreinte :** Taille des poids sur disque (Mo) et consommation de mémoire vive (RAM / VRAM).
* **Latence Opérationnelle :** Temps moyen de traitement par tweet sur architecture CPU (ms/tweet).

---

## 7. Matrice de Correspondance avec les Questions du Rapport (Q1 à Q12)

| Question | Thème Principal | Section du Guide Associée |
| :--- | :--- | :--- |
| **Q1** | Distribution train/val/test, classifieur trivial, choix de la métrique | § 2.2, § 2.3, § 3.1 |
| **Q2** | Raccourcis superficiels, indices de surface, validité des scores | § 4.1 |
| **Q3** | Choix de prétraitement par famille de modèles, étude d'ablation | § 1.4, § 4.1, § 4.3 |
| **Q4** | NLI zero-shot, sensibilité aux hypothèses, attention prémisse-hypothèse | § 4.2 |
| **Q5** | Règle Sentiment $\to$ Stance, analyse de tweets négatifs mais Favor | § 1.2 |
| **Q6** | Comparaison perte pondérée vs sur-échantillonnage vs seuil, fuite de données | § 3.2, § 3.3 |
| **Q7** | Comparaison C1, C2, C3, C4, rôle de l'attention et de l'encodage positionnel | § 4.3 |
| **Q8** | PLM gelé vs fine-tuné, AraBERT vs MARBERTv2 (temps vs performance) | § 4.4 |
| **Q9** | Apprentissage multi-tâche (sentiment, sarcasme), rôle du paramètre $\lambda$ | § 4.4 |
| **Q10** | Décalage de prior ($P(Y)$ val vs test), calibration de seuil en production | § 3.3, § 3.4 |
| **Q11** | Évaluation cross-target (3 permutations), déploiement sur nouvelle cible | § 5.1 |
| **Q12** | Analyse d'erreurs (30 cas), synthèse qualité / robustesse / coût / latence | § 5.3, § 6 |
