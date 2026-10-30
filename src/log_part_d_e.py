import pandas as pd

results_path = "results/results.csv"
df_res = pd.read_csv(results_path)

new_rows = [
    {
        "Experiment_ID": "D1_AraBERT_Frozen_CLS",
        "Section": "Partie D",
        "Model": "AraBERTv02-Twitter Gelé [CLS] + LogReg",
        "Target": "All",
        "Seed": 42,
        "Threshold": "argmax",
        "F1_Favor": 0.7805,
        "F1_Against": 0.6108,
        "F1_None": 0.2697,
        "F1_StanceEval": 0.6956,
        "Macro_F1": 0.5537,
        "Split_Evaluated": "Val",
        "Details": "Feature extraction token [CLS], class_weight=balanced"
    },
    {
        "Experiment_ID": "D1_AraBERT_Frozen_MeanPooling",
        "Section": "Partie D",
        "Model": "AraBERTv02-Twitter Gelé [Mean Pooling] + LogReg",
        "Target": "All",
        "Seed": 42,
        "Threshold": "argmax",
        "F1_Favor": 0.7861,
        "F1_Against": 0.6415,
        "F1_None": 0.4091,
        "F1_StanceEval": 0.7138,
        "Macro_F1": 0.6122,
        "Split_Evaluated": "Val",
        "Details": "Moyenne des tokens > CLS (+1.82 pts StanceEval, +5.85 Macro)"
    },
    {
        "Experiment_ID": "D2_AraBERT_Frozen_BiLSTM_Attn",
        "Section": "Partie D",
        "Model": "AraBERT Gelé + Tête BiLSTM + Attention",
        "Target": "All",
        "Seed": 42,
        "Threshold": "argmax",
        "F1_Favor": 0.8409,
        "F1_Against": 0.7087,
        "F1_None": 0.3929,
        "F1_StanceEval": 0.7748,
        "Macro_F1": 0.6475,
        "Split_Evaluated": "Val",
        "Details": "Hybride BERT gelé + BiLSTM/Attention (+6.1 pts sur D1)"
    },
    {
        "Experiment_ID": "E1_FineTuning_AraBERT_Avg3Seeds",
        "Section": "Partie E",
        "Model": "AraBERTv02-Twitter Fine-Tuné (Moyenne 3 Seeds)",
        "Target": "All",
        "Seed": "42, 123, 999",
        "Threshold": "argmax",
        "F1_Favor": 0.8520,
        "F1_Against": 0.7426,
        "F1_None": 0.4865,
        "F1_StanceEval": 0.7973,
        "Macro_F1": 0.6937,
        "Split_Evaluated": "Val",
        "Details": "lr 2e-5, warmup 10%, early stop. Std = +/- 0.0099"
    },
    {
        "Experiment_ID": "E1_FineTuning_MARBERTv2_Avg3Seeds",
        "Section": "Partie E",
        "Model": "MARBERTv2 Fine-Tuné (Moyenne 3 Seeds)",
        "Target": "All",
        "Seed": "42, 123, 999",
        "Threshold": "argmax",
        "F1_Favor": 0.8654,
        "F1_Against": 0.7796,
        "F1_None": 0.5393,
        "F1_StanceEval": 0.8225,
        "Macro_F1": 0.7281,
        "Split_Evaluated": "Val",
        "Details": "Champion SOTA Single-Task ! Bat AraBERT sur 3/3 seeds (+2.52 pts)"
    },
    {
        "Experiment_ID": "E2_MultiTask_MARBERT_Lambda00",
        "Section": "Partie E",
        "Model": "MARBERTv2 Multi-Tâche (lambda=0.0)",
        "Target": "All",
        "Seed": 42,
        "Threshold": "argmax",
        "F1_Favor": 0.8742,
        "F1_Against": 0.7771,
        "F1_None": 0.4483,
        "F1_StanceEval": 0.8256,
        "Macro_F1": 0.6999,
        "Split_Evaluated": "Val",
        "Details": "Baseline single-task architecture multi-têtes"
    },
    {
        "Experiment_ID": "E2_MultiTask_MARBERT_Lambda03",
        "Section": "Partie E",
        "Model": "MARBERTv2 Multi-Tâche (lambda=0.3)",
        "Target": "All",
        "Seed": 42,
        "Threshold": "argmax",
        "F1_Favor": 0.8884,
        "F1_Against": 0.8187,
        "F1_None": 0.4615,
        "F1_StanceEval": 0.8536,
        "Macro_F1": 0.7229,
        "Split_Evaluated": "Val",
        "Details": "🏆 RECORD ABSOLU DU PROJET : F1 StanceEval = 0.8536 ! (+2.80 pts)"
    },
    {
        "Experiment_ID": "E2_MultiTask_MARBERT_Lambda07",
        "Section": "Partie E",
        "Model": "MARBERTv2 Multi-Tâche (lambda=0.7)",
        "Target": "All",
        "Seed": 42,
        "Threshold": "argmax",
        "F1_Favor": 0.8759,
        "F1_Against": 0.7946,
        "F1_None": 0.3529,
        "F1_StanceEval": 0.8353,
        "Macro_F1": 0.6745,
        "Split_Evaluated": "Val",
        "Details": "Chute liée à la compétition de gradients (task interference)"
    }
]

df_updated = pd.concat([df_res, pd.DataFrame(new_rows)], ignore_index=True)
df_updated = df_updated.drop_duplicates(subset=["Experiment_ID"], keep="last")
df_updated.to_csv(results_path, index=False, encoding="utf-8")
print("results.csv mis à jour avec les résultats des Parties D & E !")
