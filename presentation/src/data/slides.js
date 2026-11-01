export const slides = [
  {
    id: 1,
    title: "Détection de Position (Stance Detection) en Langue Arabe sur Twitter",
    subtitle: "Master 2 Intelligence Artificielle & TALN — Corpus Mawqif",
    component: "Slide01_Intro",
    notes: `Bonjour à tous. Aujourd'hui, nous vous présentons notre projet de recherche en Traitement Automatique du Langage Naturel consacré à la détection de position (ou Stance Detection) en langue arabe sur Twitter.

L'objectif est simple : pour un tweet donné et un sujet précis (comme le vaccin Covid, la transition numérique ou l'émancipation des femmes), le modèle doit deviner si l'auteur est Pour, Contre ou Neutre. Sur les réseaux sociaux arabes, c'est particulièrement difficile à cause de la multitude de dialectes, des fautes d'orthographe et de l'argot.

Nous avons testé toute la gamme des approches : d'un classifieur aléatoire qui plafonne à 0.38, jusqu'au modèle de sacs de mots TF-IDF à 0.7278, un modèle à caractères Char-CNN à 0.7368, et le modèle pré-entraîné MARBERTv2 à 0.8225.

Notre grand résultat est l'architecture Multi-Tâche avec le sarcasme et le sentiment, qui établit le record du projet à 0.8536. 

Voyons maintenant une question fondamentale : pourquoi la détection de position ne peut-elle pas être traitée comme une simple analyse de sentiment ?`
  },
  {
    id: 2,
    title: "Pourquoi Stance ≠ Sentiment ?",
    subtitle: "Comprendre l'opinion idéologique au-delà de la valence affective",
    component: "Slide02_StanceVsSentiment",
    notes: `L'intuition la plus courante consiste à croire qu'un tweet positif signifie 'Pour', et qu'un tweet négatif signifie 'Contre'. En réalité, c'est une grave erreur.

Pour le prouver, nous avons d'abord testé un modèle de sentiment automatique : il n'obtient que 0.66 de F1. Pire encore : même si un être humain nous donnait le sentiment parfait à 100%, le score plafonne à 0.7481. Il reste donc 25% d'erreurs incompressibles.

Regardez l'exemple réel affiché : l'auteur insulte violemment ceux qui refusent le vaccin. Le sentiment est extrêmement négatif (colère, agressivité). Pourtant, sur la cible 'Vaccin Covid', la position de l'auteur est totalement POUR le vaccin !

En résumé : 1 tweet sur 4 a une émotion opposée à sa position. Il faut donc repérer la cible de l'opinion, et pas seulement mesurer la colère ou la joie de l'auteur.`
  },
  {
    id: 3,
    title: "Audit des Données & Protocole Expérimental",
    subtitle: "Prévention active du Data Leakage et détection des artefacts",
    component: "Slide03_Data",
    notes: `Avant d'entraîner le moindre modèle, nous avons mené un audit minutieux du corpus Mawqif, composé de 3 502 tweets d'entraînement et 619 tweets de test.

Premier point d'attention : le déséquilibre. 61% des tweets sont Pour, 29% Contre, et seulement 10% Neutres. Il y a donc 6 fois plus de 'Pour' que de 'Neutre'.

Deuxième point méthodologique : nous avons intercepté un bug vicieux de la bibliothèque Pandas, qui transformait automatiquement le mot 'None' en case vide NaN, supprimant 339 tweets sans prévenir. Nous avons également stratifié le découpage 90/10 en 9 strates pour que chaque sous-ensemble contienne exactement la même proportion de sujets et de positions.

Enfin, notre règle d'or d'intégrité : le test officiel de 619 tweets a été enfermé à clé et n'a jamais été vu pendant toute la phase de réglage.`
  },
  {
    id: 4,
    title: "Métriques d'Évaluation : Pourquoi pas l'Accuracy ?",
    subtitle: "Sanctionner le déséquilibre des classes et l'effet majorité",
    component: "Slide04_Metrics",
    notes: `Sur des données déséquilibrées, l'Accuracy classique est une métrique trompeuse qu'il faut bannir.

Regardez notre modèle naïf : s'il se contente de toujours répondre 'Pour' sans même lire le texte, il obtient 61.25% d'Accuracy ! Il donne l'impression de marcher correctement alors qu'il est incapable de repérer le moindre opposant.

C'est pourquoi la compétition internationale utilise la métrique officielle F1_StanceEval, qui est la moyenne stricte entre le score sur les 'Pour' et le score sur les 'Contre'.

Avec cette formule, notre modèle paresseux qui ignore les 'Contre' reçoit immédiatement un 0 sur cette classe, ce qui fait chuter son score réel à 0.3804. C'est une sanction juste et indispensable.`
  },
  {
    id: 5,
    title: "Partie A : Baselines & Raccourcis Superficiels",
    subtitle: "Quand les modèles statistiques exploitent des corrélations fallacieuses",
    component: "Slide05_Baselines",
    notes: `Dans la Partie A, nous avons évalué les baselines statistiques. Les règles basées sur 72 hashtags militants n'atteignent que 0.4071 car la majorité des internautes n'utilisent pas de hashtags explicites.

En revanche, le modèle linéaire TF-IDF combinant les mots et les n-grammes de caractères monte à un score très solide de 0.7278. Les caractères sont essentiels : les retirer fait chuter le score de 4 points car l'arabe colle préfixes et suffixes aux racines.

Cependant, le TF-IDF souffre d'un grave défaut : il apprend des réflexes bêtes. Par exemple, il associe le mot 'complot' à 'Contre'. Si un internaute écrit 'Je ne crois PAS à la théorie du complot', le TF-IDF lit le mot 'complot' et prédit 'Contre' à tort, car il est totalement aveugle à la négation syntaxique.`
  },
  {
    id: 6,
    title: "Partie B : Inférence NLI Zero-Shot Multilingue",
    subtitle: "Classifier sans aucun ré-entraînement grâce aux paires Prémisse-Hypothèse",
    component: "Slide06_NLI",
    notes: `Dans la Partie B, nous avons voulu tester l'approche Zero-Shot avec le modèle multilingue mDeBERTa-v3, qui n'a reçu aucun entraînement sur notre corpus.

Le principe est élégant : le tweet est la Prémisse. On formule une Hypothèse logique comme 'L'auteur soutient le vaccin', et le modèle calcule si le tweet valide logiquement cette hypothèse.

Le résultat dépend énormément de la langue de l'hypothèse : en arabe natif, le modèle atteint un très bon score de 0.6925 sans aucun ré-entraînement.

Mais dès qu'on traduit l'hypothèse en anglais ou en français, le score s'effondre de 12 à 14 points (0.56 et 0.54). Cela démontre que le modèle perd ses repères dialectaux quand la prémisse et l'hypothèse ne sont pas dans la même langue.`
  },
  {
    id: 7,
    title: "Partie C : Deep Learning From Scratch",
    subtitle: "BiLSTM, Attention de Bahdanau, Mini-Transformer et le triomphe du Char-CNN",
    component: "Slide07_FromScratch",
    notes: `Dans la Partie C, nous avons construit nos propres réseaux de neurones sans aucun poids pré-entraîné. Nous nous sommes heurtés à un mur : avec un vocabulaire de 8 152 mots, le taux de mots inconnus (OOV) dépasse 20% sur les tweets de test. Un mot sur cinq était donc invisible !

Malgré cela, le BiLSTM atteint 0.7177 et le Mini-Transformer 0.7167. Fait scientifique notable : si on supprime l'encodage positionnel du Transformer, il perd 1.86 point, ce qui prouve expérimentalement que l'ordre des mots est capital pour comprendre l'opinion.

Mais la grande victoire revient à notre Char-CNN : en lisant les caractères bruts plutôt que les mots entiers, son taux de mots inconnus est de 0.0% ! Il apprend les racines des mots, résiste aux fautes d'orthographe, bat tous les autres modèles à 0.7368, et ne pèse que 1.2 mégaoctet.`
  },
  {
    id: 8,
    title: "Partie D & E : Transformers Arabes Pré-entraînés",
    subtitle: "AraBERT vs MARBERTv2 : L'impact décisif des données de pré-entraînement",
    component: "Slide08_PLMs",
    notes: `Dans les Parties D et E, nous passons aux Transformers pré-entraînés sur des milliards de mots.

En mode gelé, le Mean Pooling surpasse le token CLS de 1.8 point. Mais c'est lors du fine-tuning complet que la différence se fait. Nous avons évalué AraBERT et MARBERTv2 sur 3 tirages aléatoires pour une rigueur parfaite.

AraBERT monte à 0.7973. Mais MARBERTv2 le domine largement à 0.8225, en étant deux fois plus stable.

Pourquoi cette victoire ? AraBERT a été pré-entraîné sur des articles de journaux officiels. MARBERTv2, lui, a lu plus d'un milliard de vrais tweets informels : il comprend naturellement l'argot du web arabe, les contractions et les métaphores des réseaux sociaux.`
  },
  {
    id: 9,
    title: "Partie E2 : Multi-Tâche — Record Absolu du Projet !",
    subtitle: "Régularisation conjointe par le Sentiment et le Sarcasme",
    component: "Slide09_MultiTask",
    notes: `Voici le moment fort du projet : notre modèle Transformer Multi-Tâche.

Sur Twitter, le principal piège est l'ironie : des gens qui écrivent 'Bravo les champions !' pour se moquer d'une politique. Pour déjouer cela, nous avons branché deux têtes secondaires sur MARBERTv2 : une tête pour le Sentiment, et une tête pour détecter si le tweet est Sarcastique.

En faisant varier le coefficient lambda, nous observons un effet spectaculaire : à lambda = 0.3, le score bondit à 0.8536 ! C'est le record absolu du projet, soit un gain net de +2.8 points.

En apprenant à repérer l'ironie, le modèle comprend enfin les tweets au second degré. Attention toutefois : si on augmente trop lambda (0.7), le modèle se concentre trop sur l'émotion et perd en précision.`
  },
  {
    id: 10,
    title: "Partie F : Épreuve du Feu sur le Test Scellé",
    subtitle: "Validation Statistique, Robustesse Cross-Target et Stress-Test d'Ablation",
    component: "Slide10_TestRobustesse",
    notes: `Vient ensuite le décachetage officiel du jeu de test scellé de 619 tweets, évalué en aveugle complet.

MARBERTv2 y confirme sa grande solidité avec un score officiel de 0.7947, avec un intervalle de confiance resserré entre 0.76 et 0.82. Le test statistique de McNemar montre que MARBERT et AraBERT ont un comportement très similaire sur les cas extrêmes (p = 0.625).

Pour prouver que le modèle ne triche pas, nous avons masqué les mots-clés comme 'vaccin' ou 'femme' sur les tweets bien classés : le score reste à 93.23% (chute de seulement 6.7 points). C'est la preuve qu'il analyse la structure argumentative et pas des mots isolés.

Enfin, en test hors-cible, le transfert est excellent sur les droits des femmes (0.8316) mais chute sur le Covid (0.6573) en raison du vocabulaire biomédical spécifique.`
  },
  {
    id: 11,
    title: "Partie F6 : Analyse Qualitative des Erreurs",
    subtitle: "Audit exhaustif des 142 erreurs commises sur les 619 tweets du test",
    component: "Slide11_Errors",
    notes: `Sur les 619 tweets du test, notre modèle commet 142 erreurs (77% d'exactitude). Nous avons audité ces échecs pour comprendre leurs causes réelles.

Première cause (33%) : la position implicite. Dans l'exemple affiché, l'auteur liste les risques de vol du liquide pour encourager le paiement numérique. Le modèle voit les mots 'vol, perte' et prédit 'Contre' à tort.

Deuxième cause (27%) : le sarcasme subtil. L'auteur feint de féliciter les autorités sanitaires avec des bénédictions pour se moquer des rappels vaccinaux. Le modèle prend l'éloge au premier degré.

Au total, 60% des erreurs viennent de ces deux phénomènes. C'est la limite actuelle du texte pur sur les réseaux sociaux, qui nécessiterait du contexte utilisateur ou des données multimodales.`
  },
  {
    id: 12,
    title: "Partie G : Décision de Mise en Production",
    subtitle: "Le compromis qualité, latence, coût et infrastructure pour l'industrie",
    component: "Slide12_Production",
    notes: `Pour conclure, nous proposons deux recommandations opérationnelles claires pour le déploiement en entreprise :

Premier choix : Si on dispose de serveurs Cloud avec cartes graphiques GPU, on déploie MARBERTv2 Multi-Tâche. C'est le champion absolu avec 0.8536 de F1, capable d'absorber les tweets en moins de 4 millisecondes par GPU tout en comprenant le sarcasme.

Deuxième choix : Si on doit tourner sur mobile, sur des objets connectés ou sur un petit serveur CPU sans carte graphique, notre Char-CNN est imbattable. Il ne pèse que 1.2 Mo, tourne en 3.8 millisecondes sur CPU, a 0% de mots inconnus et coûte zéro euro de matériel supplémentaire.

Merci beaucoup pour votre attention, nous sommes ravis de répondre à toutes vos questions.`
  }
];
