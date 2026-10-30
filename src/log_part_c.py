import pandas as pd

results_path = "results/results.csv"
df_res = pd.read_csv(results_path)

new_rows = [
    {
        "Experiment_ID": "C1_BiLSTM_Standard",
        "Section": "Partie C",
        "Model": "BiLSTM Standard (From Scratch)",
        "Target": "All",
        "Seed": 42,
        "Threshold": "argmax",
        "F1_Favor": 0.7991,
        "F1_Against": 0.6364,
        "F1_None": 0.0,
        "F1_StanceEval": 0.7177,
        "Macro_F1": 0.4785,
        "Split_Evaluated": "Val",
        "Details": "Embed dim 128, Hidden dim 128, 1 couche"
    },
    {
        "Experiment_ID": "C2_BiLSTM_Attention",
        "Section": "Partie C",
        "Model": "BiLSTM + Attention Bahdanau (From Scratch)",
        "Target": "All",
        "Seed": 42,
        "Threshold": "argmax",
        "F1_Favor": 0.8086,
        "F1_Against": 0.6010,
        "F1_None": 0.0,
        "F1_StanceEval": 0.7048,
        "Macro_F1": 0.4699,
        "Split_Evaluated": "Val",
        "Details": "Attention additive fait main, dim 64"
    },
    {
        "Experiment_ID": "C3_Transformer_WithPos",
        "Section": "Partie C",
        "Model": "Mini-Transformer (Avec Positional Encoding)",
        "Target": "All",
        "Seed": 42,
        "Threshold": "argmax",
        "F1_Favor": 0.8097,
        "F1_Against": 0.6238,
        "F1_None": 0.2083,
        "F1_StanceEval": 0.7167,
        "Macro_F1": 0.5473,
        "Split_Evaluated": "Val",
        "Details": "2 couches, d_model=128, 4 têtes, avec PE sinusoïdal"
    },
    {
        "Experiment_ID": "C3_Transformer_NoPos_Ablation",
        "Section": "Partie C",
        "Model": "Mini-Transformer (Ablation SANS PosEncoding)",
        "Target": "All",
        "Seed": 42,
        "Threshold": "argmax",
        "F1_Favor": 0.7828,
        "F1_Against": 0.6133,
        "F1_None": 0.2414,
        "F1_StanceEval": 0.6981,
        "Macro_F1": 0.5458,
        "Split_Evaluated": "Val",
        "Details": "Ablation study : perte de l'ordre syntaxique (-1.86 F1_StanceEval)"
    },
    {
        "Experiment_ID": "C4_Char_CNN",
        "Section": "Partie C",
        "Model": "Char-CNN (Convolutions 1D Caractères)",
        "Target": "All",
        "Seed": 42,
        "Threshold": "argmax",
        "F1_Favor": 0.8102,
        "F1_Against": 0.6633,
        "F1_None": 0.0,
        "F1_StanceEval": 0.7368,
        "Macro_F1": 0.4912,
        "Split_Evaluated": "Val",
        "Details": "Filtres 3, 5, 7 - Meilleur modèle From Scratch !"
    },
    {
        "Experiment_ID": "C_Imbalance_WeightedLoss",
        "Section": "Partie C",
        "Model": "C2 BiLSTM+Attention (Perte Pondérée)",
        "Target": "All",
        "Seed": 42,
        "Threshold": "argmax",
        "F1_Favor": 0.7524,
        "F1_Against": 0.5900,
        "F1_None": 0.2667,
        "F1_StanceEval": 0.6712,
        "Macro_F1": 0.5364,
        "Split_Evaluated": "Val",
        "Details": "Poids [Against=1.14, Favor=0.54, None=3.51]"
    },
    {
        "Experiment_ID": "C_Imbalance_Oversampling",
        "Section": "Partie C",
        "Model": "C2 BiLSTM+Attention (Sur-échantillonnage)",
        "Target": "All",
        "Seed": 42,
        "Threshold": "argmax",
        "F1_Favor": 0.7662,
        "F1_Against": 0.6071,
        "F1_None": 0.3684,
        "F1_StanceEval": 0.6866,
        "Macro_F1": 0.5806,
        "Split_Evaluated": "Val",
        "Details": "Oversampling sur Train (5799 ex). Macro-F1 +11 points !"
    }
]

df_updated = pd.concat([df_res, pd.DataFrame(new_rows)], ignore_index=True)
# Supprimer d'éventuels doublons d'expériences si relancé
df_updated = df_updated.drop_duplicates(subset=["Experiment_ID"], keep="last")
df_updated.to_csv(results_path, index=False, encoding="utf-8")
print("results.csv mis à jour avec les résultats de la Partie C !")
