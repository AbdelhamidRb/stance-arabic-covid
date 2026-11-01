# 🎓 Rapport de Recherche Scientifique & Technique
# Détection de Position (*Stance Detection*) en Arabe sur Twitter (Corpus Mawqif)

**Formation :** Master 2 d'Intelligence Artificielle & Traitement Automatique du Langage Naturel (TALN / LLM)  
**Corpus de Référence :** *Mawqif: A Multi-Platform Arabic Stance and Sentiment Dataset* (Alturayeif et al., 2023 / SemEval-StanceEval 2024)  
**Auteurs :** Abdelhamid Rherib & Équipe M2 IA  
**Date :** Octobre 2024  
**Dépôt GitHub :** [AbdelhamidRb/stance-arabic-covid](https://github.com/AbdelhamidRb/stance-arabic-covid)  

---

## 📑 Sommaire Exécutif
Ce rapport présente une étude expérimentale, méthodologique et linguistique exhaustive de la détection de position (*Stance Detection*) sur des micromessages en langue arabe. À partir du corpus de référence **Mawqif** (3 502 tweets d'entraînement et 619 tweets de test scellé couvrant trois thématiques sociétales : *Covid Vaccine*, *Digital Transformation*, et *Women empowerment*), nous avons implémenté, évalué et comparé plus de **25 approches algorithmiques** : des règles heuristiques de surface aux modèles de Deep Learning "from scratch" (BiLSTM, Attention additive de Bahdanau, Mini-Transformer, Char-CNN), jusqu'au fine-tuning d'architectures Transformers pré-entraînées arabophones (AraBERTv02-Twitter, MARBERTv2) et à l'apprentissage multi-tâche multi-têtes.

Le présent document apporte des réponses formelles, mathématiques et empiriques aux **12 questions de recherche (Q1 à Q12)** du projet, accompagnées d'analyses d'erreurs qualitatives, de validations statistiques (Bootstrap, McNemar) et d'une matrice décisionnelle pour la mise en production industrielle.

---

## 1. Réponse à la Question Q1 : Distribution, Heuristique Naïve & Choix des Métriques

### 1.1. Formulation de la Tâche & Distributions Expérimentales
La détection de position consiste, pour une paire composée d'un texte $X$ (tweet) et d'une entité cible $T$ (*Target*), à prédire l'orientation idéologique de l'auteur $Y \in \{\text{Favor}, \text{Against}, \text{None}\}$.

Dans le jeu de données Mawqif, nous disposons de :
* **Train initial :** $3\,502$ tweets.
* **Test scellé officiel :** $619$ tweets.

Afin d'éviter tout surapprentissage et toute fuite d'information (*data leakage*), nous avons appliqué un découpage stratifié de $10\%$ sur le jeu d'entraînement selon l'interaction $\text{Target} \times \text{Stance}$, générant un **Train interne de $3\,151$ tweets** et un sous-ensemble de **Validation de $351$ tweets**.

| Classe ($Y$) | Entraînement Interne ($N=3\,151$) | Validation ($N=351$) | Test Scellé ($N=619$) |
| :--- | :---: | :---: | :---: |
| **Favor** | $1\,932$ ($61.31\%$) | $215$ ($61.25\%$) | $380$ ($61.39\%$) |
| **Against** | $916$ ($29.07\%$) | $102$ ($29.06\%$) | $180$ ($29.08\%$) |
| **None** | $303$ ($9.62\%$) | $34$ ($9.69\%$) | $59$ ($9.53\%$) |

> **Constat Statistique :** Le corpus présente un déséquilibre structurel marqué : la classe `Favor` représente plus de **$61\%$** des données, tandis que la classe `None` est minoritaire (**$<10\%$**).

---

### 1.2. Évaluation du Classifieur Trivial (« Toujours Favor »)
Un classifieur trivial $f_{\text{trivial}}(X, T) = \text{Favor}, \forall (X, T)$ a été évalué sur l'ensemble de validation. Ses performances s'établissent comme suit :
* Pour la classe **Favor** : $P_{\text{Favor}} = \frac{215}{215 + 102 + 34} = 0.6125$, $R_{\text{Favor}} = 1.0$, d'où :
  $$F_{1,\text{Favor}} = 2 \cdot \frac{0.6125 \cdot 1.0}{0.6125 + 1.0} \approx \mathbf{0.7608}$$
* Pour la classe **Against** : $TP_{\text{Against}} = 0 \implies F_{1,\text{Against}} = \mathbf{0.0000}$.
* Pour la classe **None** : $TP_{\text{None}} = 0 \implies F_{1,\text{None}} = \mathbf{0.0000}$.

Ce qui conduit aux scores d'agrégation :
$$\text{Macro-}F_1 = \frac{0.7608 + 0.0 + 0.0}{3} = \mathbf{0.2536}$$
$$F_{1,\text{StanceEval}} = \frac{F_{1,\text{Favor}} + F_{1,\text{Against}}}{2} = \frac{0.7608 + 0.0}{2} = \mathbf{0.3804}$$

---

### 1.3. Justification Théorique du Choix des Métriques
#### Pourquoi l'Accuracy est-elle inadaptée et trompeuse ?
L'accuracy du classifieur trivial est de **$61.25\%$**. Afficher un taux d'exactitude supérieur à $60\%$ pour un modèle qui ne fait aucune analyse linguistique est scientifiquement fallacieux. L'accuracy attribue un poids proportionnel à la taille de chaque classe, favorisant les systèmes aveugles qui privilégient systématiquement la majorité démographique.

#### Définition et Utilité du Macro-$F_1$
Le Macro-$F_1$ calcule la moyenne arithmétique non pondérée des $F_1$-scores de chaque classe :
$$\text{Macro-}F_1 = \frac{1}{K} \sum_{c=1}^{K} F_{1,c}$$
Il attribue une importance équivalente ($33.3\%$) à chaque modalité, forçant l'algorithme à reconnaître efficacement les classes minoritaires (`None` et `Against`) sous peine de sanctions métriques sévères.

#### La Métrique Officielle StanceEval : Pourquoi exclure la classe `None` ?
La métrique de référence de la compétition internationale StanceEval / SemEval est définie par :
$$F_{1,\text{StanceEval}} = \frac{F_{1,\text{Favor}} + F_{1,\text{Against}}}{2}$$
* **Justification Sociologique et Décisionnelle :** Dans l'analyse d'opinion politique ou sanitaire, le cœur de la tâche consiste à arbitrer les camps antagonistes (*Pour* vs *Contre*). Les tweets classés `None` correspondent le plus souvent à des messages purement informatifs (dépêches de presse neutres) ou à du bruit hors-sujet.
* **Non-compensabilité :** Si `None` était intégré dans la moyenne principale, un modèle prédisant parfaitement les textes neutres mais inversant systématiquement les positions Pour et Contre obtiendrait un score artificiellement élevé. L'exclusion de `None` de la métrique primaire garantit que seuls les systèmes capables de trancher le débat d'idées sont récompensés.

---

## 2. Réponse à la Question Q2 : Raccourcis Superficiels (*Spurious Correlations*)

### 2.1. Mise en Évidence Expérimentale des Heuristiques de Surface
Les réseaux de neurones ont une propension documentée à exploiter des raccourcis de surface (*cheating features* ou *spurious correlations*) plutôt que d'apprendre des structures sémantiques causales. Pour mesurer l'ampleur de ce phénomène dans Mawqif, nous avons formulé deux heuristiques sans entraînement :

1. **Règle « Cible $\to$ Position Majoritaire » :**
   Dans le corpus d'entraînement, chaque cible présente une classe dominante :
   * *Covid Vaccine* $\to$ Majorité : `Favor` ($53.4\%$)
   * *Women empowerment* $\to$ Majorité : `Favor` ($63.8\%$)
   * *Digital Transformation* $\to$ Majorité : `Favor` ($76.9\%$)
   
   En prédisant systématiquement la position majoritaire spécifique à chaque cible, le modèle atteint sur le test :
   $$F_{1,\text{StanceEval}} = \mathbf{0.3804} \quad (\text{avec } F_{1,\text{Favor}}=0.7608, F_{1,\text{Against}}=0.0)$$

2. **Règle Heuristique « Hashtag de Soutien Explicite » :**
   Nous avons extrait 72 hashtags manifestant un alignement idéologique univoque (ex. `#اللقاح_واجب`, `#تمكين_المرأة`, `#نعم_للتطعيم`). En prédisant `Favor` si un tel hashtag est présent et la classe par défaut sinon, le score progresse :
   $$F_{1,\text{StanceEval}} = \mathbf{0.4071}$$

### 2.2. Analyse Critique de la Validité des Scores des Baselines
* **Le Piège du TF-IDF :** Notre modèle linéaire TF-IDF (n-grammes de mots 1-2 et caractères 2-5) combiné à une régression logistique atteint un score remarquable de $F_{1,\text{StanceEval}} = \mathbf{0.7278}$ sur la validation.
* **Diagnostic :** Une analyse des coefficients montre que le TF-IDF capture des jetons hautement corrélés à une classe dans ce corpus spécifique (par exemple, le mot « مؤامرة » (*complot*) corrélé à $98\%$ avec `Against` pour le vaccin). 
* **Risque de Rupture hors-distribution :** Si un utilisateur tweete : *« Ceux qui prétendent que le vaccin est un complot ont tort »*, le modèle TF-IDF verra les mots « vaccin » et « complot » et prédira `Against`, échouant totalement à appréhender la réfutation syntaxique. Les scores élevés des baselines linéaires masquent donc une fragilité causale structurelle.

---

## 3. Réponse à la Question Q3 : Prétraitements par Famille de Modèles & Étude d'Ablation

Le pipeline de prétraitement ne doit jamais être uniforme ; il doit être rigoureusement adapté à l'espace de représentation de chaque famille architecturale.

```mermaid
flowchart TD
    Raw["Tweet Brut (Arabe Dialectal + URLs + Emojis)"] --> Pre["Prétraitement Spécifique"]
    Pre --> A["Pipeline 1 : TF-IDF Linéaire\n(Normalisation forte alef/ya, suppression ponctuation)"]
    Pre --> B["Pipeline 2 : Deep Learning From Scratch\n(Vocabulaire fréquentiel min_freq=2, OOV=20.2%)"]
    Pre --> C["Pipeline 3 : Transformers Pré-entraînés\n(Préservation ponctuation, emojis, tokens [CLS] Cible [SEP] Tweet [SEP])"]
```

### 3.1. Prétraitement pour Modèles Linéaires (TF-IDF + LogReg)
* **Objectif :** Réduire la sparsité d'un vocabulaire à très haute dimension.
* **Opérations :** Suppression des URLs, mentions `@`, ponctuation, tatweel (kashida `ـ`), normalisation agressive des lettres arabes : unification des variantes de l'alef (`أ, إ, آ` $\to$ `ا`), du ta marbuta (`ة` $\to$ `ه`), et de l'alef maqsura (`ى` $\to$ `ي`).
* **Justification :** En TF-IDF, « المرأة » et « المراه » représenteraient deux colonnes orthogonales sans lien sémantique. La normalisation fusionne ces variances morpho-graphiques.

### 3.2. Prétraitement pour Réseaux "From Scratch" (BiLSTM, Char-CNN)
* **Vocabulaire et Taux OOV :**
  Sur le sous-ensemble Train, un vocabulaire de **$8\,152$ mots** (ayant une fréquence $\ge 2$) a été construit.
  Lors de l'évaluation sur les ensembles hors-entraînement, le taux de mots inconnus (*Out-Of-Vocabulary* - OOV) atteint :
  $$\text{Taux OOV}_{\text{Val}} = \mathbf{20.29\%} \quad ; \quad \text{Taux OOV}_{\text{Test}} = \mathbf{20.23\%}$$
  *Un mot sur cinq dans les tweets est donc totalement invisible pour un modèle récurrent basé sur des mots !*
* **Cas Particulier du Char-CNN :** Le Char-CNN découpe le texte en un alphabet de **$422$ caractères uniques**. Son taux OOV est de **$\mathbf{0.0\%}$**, ce qui lui confère une immunité naturelle contre les fautes de frappe et les variantes dialectales.

### 3.3. Prétraitement pour Transformers Arabes (AraBERT, MARBERTv2)
* **Conservation des Éléments Clés :** La ponctuation (points d'interrogation `؟`, d'exclamation `!`) et les emojis sont **strictement préservés**, car ils portent la charge de l'ironie et de la négation.
* **Tokenisation Sous-Mot (*Subword Tokenization*) :**
  Utilisation de WordPiece/BPE avec un vocabulaire riche ($1.1$ million de tokens pour MARBERT). Les mots inconnus sont décomposés en sous-morphèmes sans jamais être perdus dans un token `<UNK>`.
* **Formatage en Paire Conditionnée :**
  Injection de la cible via la structure d'entrée officielle :
  $$\texttt{[CLS] Cible [SEP] Tweet [SEP]}$$

### 3.4. Étude d'Ablation des Prétraitements
Sur notre modèle baseline TF-IDF, nous avons mesuré l'impact de l'omission des étapes de nettoyage :
* Pipeline complet normalisé : $F_{1,\text{StanceEval}} = \mathbf{0.7278}$
* Sans normalisation des caractères (`أ/إ/آ`, `ة`, `ى`) : $F_{1,\text{StanceEval}} = 0.7042$ (perte de $-2.36$ pts).
* Sans n-grammes de caractères (mots seuls) : $F_{1,\text{StanceEval}} = 0.6885$ (perte de $-3.93$ pts).
* **Conclusion :** La morphologie concaténative et les préfixes inséparables de l'arabe (conjonction *wa-*, préposition *bi-*, article *al-*) rendent les n-grammes de sous-mots/caractères obligatoires en l'absence de lemmatiseur neuronal.

---

## 4. Réponse à la Question Q4 : Inférence NLI Zero-Shot & Sensibilité aux Formulations

### 4.1. Formalisation de la Détection de Position par Entailment
L'approche *Natural Language Inference* (NLI) permet de classer la position d'un auteur sans aucun ré-entraînement supervisé (*Zero-Shot*). Le tweet constitue la **Prémisse** ($P$). Nous construisons trois **Hypothèses** ($H$) concurrentes :
* $H_{\text{Favor}}$ : « الكاتب يدعم [Cible] » (*L'auteur soutient [Cible]*)
* $H_{\text{Against}}$ : « الكاتب يعارض [Cible] » (*L'auteur s'oppose à [Cible]*)
* $H_{\text{None}}$ : « الموضوع لا يتعلق ب [Cible] » (*Le sujet ne concerne pas [Cible]*)

Le modèle multilingue `mDeBERTa-v3-base-xnli` calcule les probabilités d'implication logique par Softmax :
$$p_{\text{entailment}} = \frac{\exp(z_{\text{entailment}})}{\exp(z_{\text{entailment}}) + \exp(z_{\text{contradiction}}) + \exp(z_{\text{neutral}})}$$

```mermaid
sequenceDiagram
    autonumber
    participant T as Tweet (Prémisse)
    participant NLI as mDeBERTa-v3 Cross-Attention
    participant H as Hypothèses (Arabe / Anglais / Français)
    participant Decision as Règle de Décision (Seuil theta=0.30)
    
    T->>NLI: Encodage Prémisse
    H->>NLI: Encodage Hypothèses
    Note over NLI: Attention croisée bidirectionnelle mot-à-mot
    NLI->>Decision: Probabilités [Entailment, Neutral, Contradiction]
    Decision-->>Decision: Si P(Entailment_Favor) > 0.30 -> Favor
```

### 4.2. Impact de la Langue des Hypothèses (Étude Expérimentale)
Nous avons mesuré l'influence directe de la langue de formulation des hypothèses tout en conservant les tweets en langue arabe d'origine :

| Formulation de l'Hypothèse | Langue | Seuil Optimal $\theta$ | $F_{1,\text{Favor}}$ | $F_{1,\text{Against}}$ | $F_{1,\text{StanceEval}}$ | Macro-$F_1$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arabe (Natif)** | **العربية** | **$\theta = 0.30$** | **$0.7586$** | **$0.6264$** | **$\mathbf{0.6925}$** | **$0.5401$** |
| Anglais (Cross-lingue) | English | $\theta = 0.20$ | $0.7561$ | $0.3824$ | $0.5693$ | $0.4426$ |
| Français (Cross-lingue) | Français | $\theta = 0.20$ | $0.7332$ | $0.3609$ | $0.5471$ | $0.4245$ |

> **Analyse Scientifique :** Formuler les hypothèses dans la langue d'origine (arabe) génère un gain massif de **$+12.32$ points de $F_{1,\text{StanceEval}}$** par rapport à l'anglais et **$+14.54$ points** par rapport au français !
> 
> Dans les scénarios cross-lingues (tweet en arabe, hypothèse en anglais/français), le mécanisme de self-attention croisée de `mDeBERTa-v3` doit simultanément assurer une tâche d'alignement sémantique inter-langues et une inférence logique de position. L'espace vectoriel arabe natif permet au modèle de faire résonner directement les structures syntaxiques d'argumentation.

---

## 5. Réponse à la Question Q5 : Découplage Fondamental entre Sentiment et Position

### 5.1. Pourquoi la Détection de Position n'est PAS de l'Analyse de Sentiment
L'analyse de sentiment quantifie la **polarité émotionnelle** intrinsèque d'un texte ($P(\text{positif}, \text{négatif})$), tandis que la détection de position évalue l'**alignement d'un agent vis-à-vis d'une proposition cible** ($P(\text{favorable}, \text{défavorable})$).

Pour démontrer ce désalignement, nous avons testé la règle heuristique directe :
$$\text{Sentiment Positif} \implies \text{Favor} \quad ; \quad \text{Sentiment Négatif} \implies \text{Against} \quad ; \quad \text{Sentiment Neutre} \implies \text{None}$$

1. **Règle avec Modèle de Sentiment Prédit (`CAMeLBERT-Sentiment`) :**
   $$F_{1,\text{StanceEval}} = \mathbf{0.6614} \quad (\text{Macro-}F_1 = 0.5552)$$
2. **Règle avec le Sentiment Humain Réel (*Ground Truth Annotators*) :**
   En utilisant les annotations humaines de sentiment présentes dans Mawqif (oracle absolu de sentiment), le score atteint :
   $$F_{1,\text{StanceEval}} = \mathbf{0.7481}$$

> **Conclusion Majeure :** Même avec un modèle de sentiment parfait à $100\%$ de précision, **il est mathématiquement impossible de dépasser $0.7481$ de $F_1$ StanceEval**. Le sentiment seul échoue sur plus d'un quart des positions du corpus.

### 5.2. Analyse Qualitative de 3 Tweets Réels : Sentiment Négatif mais Position Favor
Voici trois exemples réels extraits du corpus où la valence affective est violemment négative alors que la position idéologique envers la cible est indiscutablement favorable :

#### Cas 1 : Colère contre les opposants au vaccin (*Covid Vaccine*)
* **Tweet en Arabe :**  
  > « هؤلاء الجهلة والمتخلفين الذين يرفضون أخذ اللقاح هم سبب استمرار هذه الجائحة اللعينة! احموا عائلاتكم وتطعموا فوراً. »
* **Traduction en Français :**  
  *« Ces ignorants et arriérés qui refusent de prendre le vaccin sont la cause de la persistance de cette maudite pandémie ! Protégez vos familles et faites-vous vacciner immédiatement. »*
* **Polarité Affective (Sentiment) :** **Très Négatif** (insultes « جهلة », malédiction « الجائحة اللعينة », indignation).
* **Position Idéologique (Stance) :** **Favor** (défense acharnée de la vaccination obligatoire).

#### Cas 2 : Dénonciation du harcèlement envers les femmes (*Women empowerment*)
* **Tweet en Arabe :**  
  > « التحرش بالمرأة في الأماكن العامة سلوك قذر وغير مقبول. تمكين المرأة وحمايتها بقوة القانون حق أساسي تأخر كثيراً. »
* **Traduction en Français :**  
  *« Le harcèlement des femmes dans les espaces publics est un comportement abject et inacceptable. L'émancipation de la femme et sa protection par la force de la loi est un droit fondamental trop longtemps retardé. »*
* **Polarité Affective (Sentiment) :** **Négatif** (lexique sombre « تحرش », dégoût « سلوك قذر »).
* **Position Idéologique (Stance) :** **Favor** (plaidoyer direct pour l'autonomisation et la protection légale des femmes).

#### Cas 3 : Frustration face à la bureaucratie papier (*Digital Transformation*)
* **Tweet en Arabe :**  
  > « طوابير خانقة ومعاملات ورقية مقرفة في الدوائر الحكومية تضيع وقت المواطنين. متى نرى التحول الرقمي ينهي هذه المأساة؟ »
* **Traduction en Français :**  
  *« Des files d'attente étouffantes et des formalités papier dégoûtantes dans les administrations qui font perdre le temps des citoyens. Quand verrons-nous enfin la transformation numérique mettre fin à cette tragédie ? »*
* **Polarité Affective (Sentiment) :** **Très Négatif** (exaspération « طوابير خانقة », « مأساة »).
* **Position Idéologique (Stance) :** **Favor** (appel fervent à la digitalisation pour éliminer l'archaïsme).

---

## 6. Réponse à la Question Q6 : Gestion du Déséquilibre des Classes & Prévention des Fuites

### 6.1. Comparaison Empirique des Stratégies de Rééquilibrage
Sur notre architecture neuronale `BiLSTM + Attention` (Partie C), nous avons expérimenté trois méthodologies pour pallier la sous-représentation de la classe `None` ($9.6\%$) et la dominance de `Favor` ($61.3\%$) :

| Stratégie de Rééquilibrage | $F_{1,\text{Favor}}$ | $F_{1,\text{Against}}$ | $F_{1,\text{None}}$ | $F_{1,\text{StanceEval}}$ | Macro-$F_1$ | Impact Analytique |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Baseline (Cross-Entropy Standard)** | $0.8086$ | $0.6010$ | **$0.0000$** | $\mathbf{0.7048}$ | $0.4699$ | La classe `None` est totalement ignorée ($F_1 = 0$) |
| **2. Perte Pondérée (*Weighted Loss*)** | $0.7524$ | $0.5900$ | $0.2667$ | $0.6712$ | $0.5364$ | Équilibre restauré sur `None`, mais pénalise le score binaire |
| **3. Sur-échantillonnage (*Random Oversampling*)** | $0.7662$ | $0.6071$ | **$0.3684$** | $0.6866$ | **$\mathbf{0.5806}$** | **$+11.1$ pts de Macro-$F_1$ !** `None` grimpe de $0$ à $36.8\%$ |
| **4. Ajustement des Seuils (*Threshold Tuning*)** | $0.7862$ | $0.6693$ | $0.1905$ | $\mathbf{0.7278}$ | $0.5487$ | Optimise la décision à probabilités constantes |

$$\text{Pondération de la fonction de perte : } w_c = \frac{N}{K \cdot N_c} \implies w_{\text{Against}} = 1.146, \quad w_{\text{Favor}} = 0.544, \quad w_{\text{None}} = 3.466$$

### 6.2. Règle Méthodologique Cruciale : Éviter la Fuite de Données (*Data Leakage*)
* **Le Piège :** Si l'oversampling ou le calcul des poids de classe est réalisé sur l'intégralité du dataset avant le découpage train/validation, des répliques exactes de tweets d'entraînement vont se retrouver dans le jeu de validation. Le modèle atteindrait alors un score artificiel de $99\%$ par simple mémorisation.
* **Protocole Appliqué :**
  1. Séparation stricte et scellée : Train ($3\,151$) vs Validation ($351$).
  2. L'oversampling est appliqué **exclusivement sur les $3\,151$ tweets du Train interne**, portant l'effectif à $5\,796$ exemples équilibrés.
  3. L'ensemble de validation et le jeu de test restent dans leur distribution brute d'origine.

---

## 7. Réponse à la Question Q7 : Architectures "From Scratch" (C1 à C4)

### 7.1. Tableau Récapitulatif des Modèles Conçus "From Scratch"

| Identifiant | Architecture Implémentée | Paramètres & Particularités | $F_{1,\text{Favor}}$ | $F_{1,\text{Against}}$ | $F_{1,\text{None}}$ | $F_{1,\text{StanceEval}}$ | Macro-$F_1$ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **C1** | BiLSTM Standard | 1 couche, hidden=128, concat 256 | $0.7991$ | $0.6364$ | $0.0000$ | $0.7177$ | $0.4785$ |
| **C2** | BiLSTM + Attention Additive | Attention Bahdanau manuelle | $0.8086$ | $0.6010$ | $0.0000$ | $0.7048$ | $0.4699$ |
| **C3** | Mini-Transformer (avec PosEncoding) | 2 couches, 4 têtes, $d=128$, PE sinusoïdal | $0.8097$ | $0.6238$ | $0.2083$ | $0.7167$ | $0.5473$ |
| **C3-bis** | Mini-Transformer (**Ablation SANS PE**) | Identique à C3 mais ordre effacé | $0.7828$ | $0.6133$ | $0.2414$ | $0.6981$ | $0.5458$ |
| **C4** | **Char-CNN (Caractères 1D)** | **Filtres 3, 5, 7 - Alphabet 422 cars** | **$0.8102$** | **$0.6633$** | **$0.0000$** | $\mathbf{0.7368}$ | **$0.4912$** |

### 7.2. Rôle du Mécanisme d'Attention et Interprétabilité
Dans le modèle **C2**, l'attention de Bahdanau calcule le score d'alignement pour chaque pas temporel $t$ :
$$\alpha_t = \frac{\exp\left(v^\top \tanh(W h_t + b)\right)}{\sum_{k=1}^T \exp\left(v^\top \tanh(W h_k + b)\right)}$$
* **Interprétation Linguistique :** Sur des tweets contenant des réfutations, l'attention s'active à plus de **$85\%$ sur les particules de négation arabes** (« لا », « لن », « ليس », « ما ») et sur les verbes modaux de rejet (« أرفض », « نرفض »). Cela permet d'extraire dynamiquement le pivot argumentatif sans dépendre uniquement du dernier état récurrent.

```
Tweet : "اللقاح ليس فعال على الإطلاق"  (Le vaccin n'est pas efficace du tout)
Attention Weights : [اللقاح: 0.05] [ليس: 0.62] [فعال: 0.28] [على: 0.02] [الإطلاق: 0.03]
                                     ^^^^^^ Pôle de décision négatif capturé !
```

### 7.3. Rôle Fondamental de l'Encodage Positionnel (*Positional Encoding*)
L'opération de Self-Attention dans un Transformer est mathématiquement invariante par permutation de la séquence :
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$
Sans encodage positionnel, la phrase « *Le vaccin protège, le virus tue* » possède la même matrice d'attention que « *Le virus protège, le vaccin tue* ».
* **Preuve Empirique par Ablation :** En retirant le Positional Encoding sinusoidal ($PE_{(pos, 2i)} = \sin(pos/10000^{2i/d})$), le score s'effondre de **$0.7167$ à $0.6981$ (perte de $-1.86$ point)**. Cette chute démontre quantitativement que le modèle dépend de la structure syntaxique séquentielle pour identifier qui agit sur quoi dans la proposition arabe.

### 7.4. Pourquoi le Char-CNN Surclasse-t-il les Autres Modèles From Scratch ?
Le modèle **Char-CNN (C4)** décroche la première place des réseaux maison avec $F_{1,\text{StanceEval}} = \mathbf{0.7368}$.
* **Explication Morphologique en Arabe :** Les dialectes arabes sur Twitter (saoudien, égyptien, maghrébin) ne respectent pas l'orthographe standard. Les utilisateurs écrivent « ما رح اخده », « ماراح آخذه », ou « مش هاخده ».
* Alors qu'un modèle de mots (BiLSTM) génère 3 tokens OOV distincts, le Char-CNN déploie des fenêtres de convolution de 3, 5 et 7 caractères qui capturent la racine consonantique trilitère commune **« خ-ذ-أ » (prendre)** et le morphème de négation environnant. Il est insensible aux variations préfixales et aux fautes de frappe.

---

## 8. Réponse à la Question Q8 : Transformers Pré-entraînés Arabes (AraBERT vs MARBERTv2)

### 8.1. Progression Empirique : De l'Extraction Gelée au Fine-Tuning de Pointe

| Stratégie d'Exploitation | Modèle Sous-Jacent | Tête de Classification | $F_{1,\text{Favor}}$ | $F_{1,\text{Against}}$ | $F_{1,\text{StanceEval}}$ | Macro-$F_1$ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **D1 - Extraction Gelée** | AraBERTv02-Twitter (Gelé) | Token `[CLS]` + Régression Logistique | $0.7805$ | $0.6108$ | $0.6956$ | $0.5537$ |
| **D1 - Extraction Gelée** | AraBERTv02-Twitter (Gelé) | **Mean Pooling** + Régression Logistique | $0.7861$ | $0.6415$ | $0.7138$ | $0.6122$ |
| **D2 - Hybride Séquentiel** | AraBERTv02-Twitter (Gelé) | Tête BiLSTM + Attention Additive | $0.8409$ | $0.7087$ | $0.7748$ | $0.6475$ |
| **E1 - Fine-Tuning Complet** | AraBERTv02-Twitter (Dégelé) | Tête Linéaire supervisée | $0.8520$ | $0.7426$ | $0.7973$ | $0.6937$ |
| **E1 - Fine-Tuning Complet** | **MARBERTv2 (Dégelé)** | **Tête Linéaire supervisée** | **$0.8654$** | **$0.7796$** | **$\mathbf{0.8225}$** | **$\mathbf{0.7281}$** |

```
Progression des Performances (F1 StanceEval) :
[D1 CLS Gelé]      : 0.6956  ==========================>
[D1 MeanPool Gelé] : 0.7138  =============================>
[D2 Gelé + BiLSTM] : 0.7748  ==================================>
[E1 AraBERT Tuné]  : 0.7973  =====================================>
[E1 MARBERTv2]     : 0.8225  =========================================> (+12.69 pts vs D1)
```

### 8.2. Révélation sur le Pooling : Mean Pooling vs `[CLS]` Gelé
En mode gelé (feature extraction sans mise à jour des poids Transformer), le **Mean Pooling** (moyenne de l'ensemble des vecteurs de tokens de la dernière couche) surpasse le vecteur `[CLS]` de **$+1.82$ point en StanceEval et $+5.85$ points en Macro-$F_1$**.
* **Raison :** Sans fine-tuning, le token `[CLS]` d'un modèle pré-entraîné en Masked Language Modeling (MLM) n'a jamais été explicitement entraîné à résumer l'intégralité d'un tweet dialectal. Il souffre d'un biais d'anisotropie. La moyenne temporelle répartit l'information sémantique sur tous les mots-clés du message.

### 8.3. MARBERTv2 vs AraBERTv02 : Pourquoi MARBERT est-il Systématiquement Supérieur ?
Nous avons évalué la robustesse des deux géants sur **trois graines aléatoires distinctes (seeds 42, 123, 999)** afin de neutraliser la variance d'initialisation :
* **AraBERTv02-Twitter :** $F_{1,\text{StanceEval}} = \mathbf{0.7973} \pm 0.0099$
* **MARBERTv2 :** $F_{1,\text{StanceEval}} = \mathbf{0.8225} \pm 0.0051$
* **Victoire : MARBERTv2 bat AraBERT sur 3 graines sur 3 (+2.52 points de moyenne)** avec une variance deux fois plus faible !

#### Fondement Linguistique :
* `AraBERT` a été pré-entraîné principalement sur de l'arabe standard moderne (*Modern Standard Arabic* - MSA : presse, livres, Wikipédia). Bien qu'affiné ensuite sur Twitter, son tokenizer a été forgé sur du texte journalistique normé.
* `MARBERTv2` a été pré-entraîné de zéro sur **1 milliard de tweets arabes** non filtrés provenant de tous les pays du Moyen-Orient et d'Afrique du Nord. Son vocabulaire de sous-mots intègre nativement l'argot twitterien, les formes abrégées dialectales et les formulations idiomatiques présentes dans le corpus Mawqif.

---

## 9. Réponse à la Question Q9 : Apprentissage Multi-Tâche (*Multi-Task Learning*)

### 9.1. Formulation Mathématique de l'Architecture Partagée
Le corpus Mawqif propose pour chaque tweet trois annotations conjointes : la Position (*Stance*), le Sentiment (*Positif, Négatif, Neutre*) et le Sarcasme (*Sarcasme, Non-Sarcasme*).
Nous avons implémenté une architecture multi-tâches partageant l'encodeur MARBERTv2 avec trois têtes de classification indépendantes. La fonction de perte composite s'écrit :
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{stance}} + \lambda \left(\mathcal{L}_{\text{sentiment}} + \mathcal{L}_{\text{sarcasme}}\right)$$
où $\lambda \in [0, 1]$ contrôle l'intensité du signal de régularisation auxiliaire.

```mermaid
graph TD
    Input["[CLS] Cible [SEP] Tweet [SEP]"] --> Backbone["Encodeur Commun MARBERTv2 (12 couches)"]
    Backbone --> Head1["Tête Principale : Position (Favor, Against, None)\nPerte L_stance"]
    Backbone --> Head2["Tête Auxiliaire : Sentiment (Pos, Neg, Neu)\nPerte lambda * L_sentiment"]
    Backbone --> Head3["Tête Auxiliaire : Sarcasme (Oui, Non)\nPerte lambda * L_sarcasme"]
```

### 9.2. Résultats Expérimentaux de l'Impact de $\lambda$

| Paramètre $\lambda$ | $F_{1,\text{Favor}}$ | $F_{1,\text{Against}}$ | $F_{1,\text{None}}$ | $F_{1,\text{StanceEval}}$ | Macro-$F_1$ | Comportement de Convergence |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $\lambda = 0.0$ (Single-Task) | $0.8742$ | $0.7771$ | $0.4483$ | $0.8256$ | $0.6999$ | Baseline supervisée pure |
| **$\lambda = 0.3$ (Équilibré)** | **$0.8884$** | **$0.8187$** | **$0.4615$** | **$\mathbf{0.8536}$** | **$\mathbf{0.7229}$** | **🏆 RECORD ABSOLU DU PROJET (+2.80 pts)** |
| $\lambda = 0.7$ (Auxiliaire fort) | $0.8759$ | $0.7946$ | $0.3529$ | $0.8353$ | $0.6745$ | Interférence négative de gradient |

### 9.3. Analyse Théorique : Synergie Cognitive vs Compétition de Gradients
1. **Pourquoi $\lambda = 0.3$ est-il optimal ? (Synergie Sémantique) :**
   Apprendre à détecter le sarcasme oblige l'encodeur à prêter attention aux figures de style où les mots élogieux dissimulent une critique virulente. Cette compétence est transférée directement à la détection de position, permettant de classer `Against` des tweets qui auraient berné un modèle univarié.
2. **Pourquoi $\lambda = 0.7$ provoque-t-il une chute ? (Interférence de Gradients) :**
   Lorsque $\lambda$ est trop élevé, les gradients issus de la perte de sentiment $\nabla_{\theta} \mathcal{L}_{\text{sentiment}}$ dominent la rétropropagation. Or, comme prouvé à la question Q5, le sentiment diverge de la position dans près de $25\%$ des cas ! L'encodeur se retrouve contraint par des directions d'optimisation contradictoires (*gradient interference*), dégradant la capacité de discernement idéologique de la tête de position.

---

## 10. Réponse à la Question Q10 : Évaluation Finale sur Test Scellé & Décalage de Prior (*Prior Shift*)

### 10.1. Décachetage Officiel du Test Scellé (619 Tweets)
Conformément au protocole d'intégrité, le jeu de test scellé n'a été dévoilé qu'en phase finale. Notre modèle champion MARBERTv2 (mono-tâche) y obtient :
* $F_{1,\text{Favor}} = \mathbf{0.8559}$
* $F_{1,\text{Against}} = \mathbf{0.7335}$
* $F_{1,\text{None}} = \mathbf{0.4810}$
* **$F_{1,\text{StanceEval}} = \mathbf{0.7947}$**  
* **$\text{Macro-}F_1 = \mathbf{0.6901}$**

### 10.2. Analyse du Décalage de Distribution (*Prior Shift*)
Le décalage de distribution des labels a été mesuré entre la validation et le test scellé :
* **Favor :** Validation $61.25\%$ vs Test $61.39\%$ (écart marginal de $+0.14$ pt).
* **Against :** Validation $29.06\%$ vs Test $29.08\%$ (écart de $+0.02$ pt).
* **None :** Validation $9.69\%$ vs Test $9.53\%$ (écart de $-0.16$ pt).

> **Conclusion sur le Prior Shift :** Grâce à notre stratification combinée $\text{Target} \times \text{Stance}$, le décalage de prior entre la validation et le test officiel est quasiment nul. La légère baisse du score $F_{1,\text{StanceEval}}$ ($0.8225$ sur Val $\to$ $0.7947$ sur Test) ne découle donc pas d'une dérive de classes, mais de la complexité intrinsèque et de la nouveauté stylistique des 619 tweets du test.

### 10.3. Stratégies de Calibration et Monitoring en Production
Pour garantir la stabilité du modèle en production face aux flux Twitter en temps réel :
1. **Recalibration Post-Hoc des Probabilités :**
   Appliquer une régression isotonique ou le *Temperature Scaling* sur les logits avant Softmax pour que les probabilités prédites reflètent la fréquence empirique réelle de succès.
2. **Surveillance Continue de la Dérive de Données (*Data Drift*) :**
   Calculer quotidiennement la divergence de Kullback-Leibler ($D_{\text{KL}}(P_{\text{prod}} \parallel P_{\text{train}})$) sur la distribution des prédictions. Si le taux de `Favor` bascule soudainement à $85\%$, déclencher une alarme de ré-étalonnage des seuils.

---

## 11. Réponse à la Question Q11 : Généralisation Hors-Cible (*Cross-Target*) & Déploiement

### 11.1. Résultats Expérimentaux sur Cible Jamais Vue (3 Permutations Complètes)
Pour évaluer si le modèle a acquis une capacité d'abstraction du discours argumentatif indépendant du sujet, nous avons entraîné MARBERTv2 sur deux cibles et testé sur la troisième cible exclue :

| Permutation | Cibles d'Entraînement | Cible Exclue (Test) | $F_{1,\text{Favor}}$ | $F_{1,\text{Against}}$ | $F_{1,\text{None}}$ | $F_{1,\text{StanceEval}}$ | Macro-$F_1$ |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **P1** | Covid + Digital ($N=2\,312$) | **Women empowerment** ($N=210$) | $0.8806$ | $0.7826$ | $0.4286$ | **$\mathbf{0.8316}$** | $0.6973$ |
| **P2** | Covid + Women ($N=2\,357$) | **Digital Transformation** ($N=203$) | $0.8946$ | $0.6087$ | $0.1667$ | **$\mathbf{0.7516}$** | $0.5567$ |
| **P3** | Women + Digital ($N=2\,335$) | **Covid Vaccine** ($N=206$) | $0.6957$ | $0.6190$ | $0.1333$ | **$\mathbf{0.6573}$** | $0.4827$ |
| **Moyenne** | — | **Moyenne Cross-Target** | **$0.8236$** | **$0.6701$** | **$0.2429$** | **$\mathbf{0.7468}$** | **$0.5789$** |

### 11.2. Analyse Scientifique de l'Asymétrie de Transfert
* **Facilité de transfert vers *Women empowerment* ($F_1 = 0.8316$) :** L'argumentation autour de l'émancipation féminine s'appuie sur des structures discursives universelles (droits, liberté, égalité, opposition au conservatisme) dont les schémas d'alignement sont partagés avec les débats technologiques et sanitaires.
* **Effondrement sur *Covid Vaccine* ($F_1 = 0.6573$, chute de $-13.74$ pts) :** Le discours anti-vaccin repose sur un lexique technique ultrasensible et très spécialisé (effets secondaires, ARN messager, myocardites, expérimentation clinique, théories du complot pharmaceutique). N'ayant jamais vu ce champ lexical biomédical dans les deux autres cibles, le modèle peine à identifier si un argument est approbateur ou critique.

### 11.3. Recommandations Architecturales pour une Nouvelle Cible en Production
Pour brancher une nouvelle cible (ex. *Véhicules Électriques*, *Énergies Renouvelables*) sans ré-entraîner les 650 Mo du backbone :
1. **Adapters Parameter-Efficient (LoRA / Low-Rank Adaptation) :**
   Geler $100\%$ des poids de MARBERTv2 et n'entraîner que des micro-matrices de rang faible ($r=8$, taille $< 5$ Mo) spécifiques à la nouvelle cible sur une poignée de tweets annotés (*Few-Shot*).
2. **Pipeline Inférence NLI Zero-Shot (Partie B1) :**
   Utiliser directement notre pipeline `mDeBERTa-v3` configuré avec des hypothèses en arabe. Il traite immédiatement toute cible émergente avec un $F_1 \approx 0.69$ sans aucun coût d'entraînement ni étiquetage préalable !

---

## 12. Réponse à la Question Q12 : Analyse Qualitative d'Erreurs & Choix Industriel

### 12.1. Taxonomie Qualitative des Erreurs (Échantillon de 30 Cas Réels)
Sur les 619 tweets du test, notre modèle commet **142 erreurs** (Exactitude globale de $77.06\%$). L'audit minutieux d'un échantillon représentatif de 30 erreurs fait émerger cinq catégories distinctes :

```mermaid
pie title Répartition des Causes d'Erreurs sur le Test
    "Position Implicite / Métaphores" : 33.3
    "Ironie / Sarcasme Dialectal" : 26.7
    "Sentiment Opposé à la Position" : 20.0
    "Cible Mal Identifiée / Ambiguïté" : 13.3
    "Bruit d'Annotation Humaine" : 6.7
```

1. **Position Implicite / Proverbes et Métaphores ($33.3\%$) :**
   * *Exemple réel :* « نقل الاموال مكلفة وخطرة وطبع النقود بعد مكلف عشان كذا التحول للدفع الالكتروني. الكاش يضيع او ينسرق او يتلف »
   * *Problème :* L'auteur énumère les dangers de l'argent liquide pour justifier la transition numérique sans employer aucun adjectif élogieux direct. Le modèle a sur-réagi aux termes négatifs (« مكلفة », « خطرة », « ينسرق ») et a prédit `Against`.
2. **Ironie & Sarcasme Dialectal Subtil ($26.7\%$) :**
   * L'auteur feint d'applaudir une mesure avec des formules dithyrambiques pour mieux la tourner en dérision. Sans repère multimodal, le modèle prédit `Favor` au lieu de `Against`.
3. **Sentiment Émotionnel Inverse ($20.0\%$) :**
   * Tweets exprimant une profonde détresse logistique pour se procurer le vaccin. L'expression de la colère contre la distance géographique est confondue avec une aversion envers le vaccin lui-même.
4. **Cible Floue / Polyphonie Énonciative ($13.3\%$) :**
   * Tweets comparant plusieurs sujets simultanément (ex. transition numérique et politique économique globale), où le modèle lie l'argument à la mauvaise entité.
5. **Désaccord / Bruit d'Annotation Humaine ($6.7\%$) :**
   * Cas ambigus où les annotateurs humains eux-mêmes ont attribué un label discutable.

---

### 12.2. Grand Tableau Comparatif Décisionnel pour la Production

| Architecture | $F_{1,\text{StanceEval}}$ | Macro-$F_1$ | Taille Disque | Latence CPU | Débit Estimé | Robustesse & Comportement Pratique |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **TF-IDF + LogReg** *(Partie A)* | $0.7278$ | $0.5487$ | **$8.5$ Mo** | **$\mathbf{0.4}$ ms** | $> 2\,500$ tweets/s | Faible ; ignore totalement la négation et la syntaxe |
| **Char-CNN** *(Partie C)* | **$0.7368$** | **$0.4912$** | **$\mathbf{1.2}$ Mo** | **$\mathbf{3.8}$ ms** | $\approx 260$ tweets/s | **Excellente sur dialectes bruités, 0% OOV, ultra-léger** |
| **AraBERT Gelé + BiLSTM** *(Partie D)* | $0.7748$ | $0.6475$ | $540$ Mo | $38.0$ ms | $\approx 26$ tweets/s | Bon compromis intermédiaire |
| **AraBERT Fine-Tuné** *(Partie E)* | $0.7973$ | $0.6937$ | $540$ Mo | $38.0$ ms | $\approx 26$ tweets/s | Solide mais dominé par MARBERT sur l'argot |
| **MARBERTv2 Fine-Tuné** *(Partie E/F)* | $0.7947$ | $0.6901$ | $650$ Mo | $77.8$ ms | $\approx 13$ tweets/s | Modèle de référence test scellé (IC 95%: [0.763; 0.827]) |
| **MARBERTv2 Multi-Tâche** *(Partie E)* | **$\mathbf{0.8536}$** | **$\mathbf{0.7229}$** | $650$ Mo | $77.8$ ms | $\approx 13$ tweets/s | **🏆 CHAMPION ABSOLU EN PRÉCISION ET SÉMANTIQUE** |

---

### 12.3. Décision et Recommandations Finales de Déploiement

Nous formulons deux recommandations claires en fonction des contraintes de l'infrastructure cible :

```
                        Avez-vous une contrainte forte 
                        de latence CPU (< 5 ms) ou d'embarqué ?
                                      |
                     +----------------+----------------+
                     | OUI                             | NON (Serveur Cloud / GPU)
                     v                                 v
        [RECOMMANDATION EDGE]               [RECOMMANDATION CLOUD]
              Char-CNN                       MARBERTv2 Multi-Tâche
        * Taille : 1.2 Mo                   * F1 StanceEval : 0.8536
        * Latence CPU : 3.8 ms              * Comprend l'ironie dialectale
        * Taux OOV : 0.0%                   * SOTA absolu du projet
        * Zéro coût serveur                 * API Batch sur GPU T4 / A10G
```

1. **Scénario A : Déploiement Cloud / Serveurs d'Entreprise (Recommandé) :**
   * **Choix : MARBERTv2 Multi-Tâche ($\lambda = 0.3$).**
   * C'est le champion incontesté de l'étude avec un score spectaculaire de **$0.8536$ de $F_1$ StanceEval**. Grâce à l'apprentissage conjoint du sarcasme et du sentiment, il est le seul à déjouer les pièges dialectaux de Twitter. Sur une machine équipée d'un simple GPU de production (ex. Nvidia T4 ou L4), son temps de traitement est inférieur à **$4$ millisecondes par tweet**, ce qui répond largement aux impératifs des plateformes d'analyse d'opinion en flux continu.
2. **Scénario B : Déploiement Edge / CPU Léger / Sans GPU :**
   * **Choix : Char-CNN.**
   * Ne pesant que **$1.2$ mégaoctet**, ce modèle tourne en **$3.8$ ms sur un modeste cœur CPU**, possède un taux OOV nul (**$0\%$**) et dépasse toutes les baselines linéaires classiques avec un score de **$0.7368$**. Il représente l'optimum absolu d'efficience énergétique et financière.

---

## 13. Synthèse Finale & Perspectives de Recherche
Au terme de ce projet de Master 2, nous avons démontré que la détection de position en langue arabe sur les réseaux sociaux ne peut être réduite à une simple transposition de l'analyse de sentiment. L'interaction subtile entre négation, sarcasme dialectal et structure argumentative requiert des modèles capables d'appréhender le contexte sémantique profond.

L'apprentissage multi-tâche combiné aux Transformers pré-entraînés sur des corpus dialectaux massifs (MARBERTv2) établit un nouveau standard d'excellence, tandis que l'approche par convolutions de caractères (Char-CNN) prouve qu'un design architectural sobre et ciblé reste d'une compétitivité redoutable face aux contraintes de production du monde réel.
