import pandas as pd

results_path = "results/results.csv"
df_res = pd.read_csv(results_path)

new_rows = [
    {
        "Experiment_ID": "B1_NLI_mDeBERTa_AR",
        "Section": "Partie B",
        "Model": "mDeBERTa-v3-base-xnli (Hypothèses Arabe)",
        "Target": "All",
        "Seed": 42,
        "Threshold": "theta=0.30",
        "F1_Favor": 0.7586,
        "F1_Against": 0.6264,
        "F1_None": 0.2353,
        "F1_StanceEval": 0.6925,
        "Macro_F1": 0.5401,
        "Split_Evaluated": "Val",
        "Details": "Hypothèses en arabe (الكاتب يدعم / يعارض)"
    },
    {
        "Experiment_ID": "B1_NLI_mDeBERTa_EN",
        "Section": "Partie B",
        "Model": "mDeBERTa-v3-base-xnli (Hypothèses Anglais)",
        "Target": "All",
        "Seed": 42,
        "Threshold": "theta=0.20",
        "F1_Favor": 0.7561,
        "F1_Against": 0.3824,
        "F1_None": 0.1892,
        "F1_StanceEval": 0.5693,
        "Macro_F1": 0.4426,
        "Split_Evaluated": "Val",
        "Details": "Cross-lingual NLI (The author supports / opposes)"
    },
    {
        "Experiment_ID": "B1_NLI_mDeBERTa_FR",
        "Section": "Partie B",
        "Model": "mDeBERTa-v3-base-xnli (Hypothèses Français)",
        "Target": "All",
        "Seed": 42,
        "Threshold": "theta=0.20",
        "F1_Favor": 0.7332,
        "F1_Against": 0.3609,
        "F1_None": 0.1795,
        "F1_StanceEval": 0.5471,
        "Macro_F1": 0.4245,
        "Split_Evaluated": "Val",
        "Details": "Cross-lingual NLI (L'auteur soutient / s'oppose)"
    },
    {
        "Experiment_ID": "B2_CAMeLBERT_Sentiment",
        "Section": "Partie B",
        "Model": "CAMeLBERT Sentiment -> Règle Stance",
        "Target": "All",
        "Seed": 42,
        "Threshold": "Heuristique directe",
        "F1_Favor": 0.6766,
        "F1_Against": 0.6462,
        "F1_None": 0.3429,
        "F1_StanceEval": 0.6614,
        "Macro_F1": 0.5552,
        "Split_Evaluated": "Val",
        "Details": "positive->Favor, negative->Against, neutral->None"
    },
    {
        "Experiment_ID": "B2_GroundTruth_Sentiment",
        "Section": "Partie B",
        "Model": "Sentiment Réel Annotateurs -> Règle Stance",
        "Target": "All",
        "Seed": 42,
        "Threshold": "Heuristique directe",
        "F1_Favor": 0.7632,
        "F1_Against": 0.7330,
        "F1_None": 0.3158,
        "F1_StanceEval": 0.7481,
        "Macro_F1": 0.6040,
        "Split_Evaluated": "Val",
        "Details": "Plafond théorique de la règle de sentiment"
    }
]

df_updated = pd.concat([df_res, pd.DataFrame(new_rows)], ignore_index=True)
df_updated.to_csv(results_path, index=False, encoding="utf-8")
print("results.csv mis à jour avec succès avec les résultats de la Partie B !")
