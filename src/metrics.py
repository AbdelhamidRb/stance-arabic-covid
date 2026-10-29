import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

def compute_stance_metrics(y_true, y_pred, labels=["Favor", "Against", "None"]):
    """
    Calcule :
    - F1_Favor, F1_Against, F1_None
    - F1_StanceEval = (F1_Favor + F1_Against) / 2 (Métrique principale)
    - Macro_F1 = moyenne des 3 classes
    """
    # Calcul par classe
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    
    f1_dict = {f"F1_{label}": round(score, 4) for label, score in zip(labels, f1)}
    
    f1_favor = f1_dict.get("F1_Favor", 0.0)
    f1_against = f1_dict.get("F1_Against", 0.0)
    f1_none = f1_dict.get("F1_None", 0.0)
    
    f1_stanceval = round((f1_favor + f1_against) / 2.0, 4)
    macro_f1 = round((f1_favor + f1_against + f1_none) / 3.0, 4)
    
    return {
        "F1_Favor": f1_favor,
        "F1_Against": f1_against,
        "F1_None": f1_none,
        "F1_StanceEval": f1_stanceval,
        "Macro_F1": macro_f1
    }

def bootstrap_confidence_interval(y_true, y_pred, n_bootstraps=1000, alpha=0.05, seed=42):
    """Calcule l'intervalle de confiance à 95% par bootstrap."""
    rng = np.random.RandomState(seed)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_samples = len(y_true)
    
    scores = []
    for _ in range(n_bootstraps):
        indices = rng.randint(0, n_samples, n_samples)
        m = compute_stance_metrics(y_true[indices], y_pred[indices])
        scores.append(m["F1_StanceEval"])
        
    scores = np.sort(scores)
    lower = np.percentile(scores, 100 * (alpha / 2.0))
    upper = np.percentile(scores, 100 * (1.0 - alpha / 2.0))
    return round(float(lower), 4), round(float(upper), 4)

def mcnemar_test(y_true, y_pred_m1, y_pred_m2):
    """
    Test de McNemar pour comparer deux modèles M1 et M2.
    Table de contingence :
      b = M1 juste et M2 faux
      c = M1 faux et M2 juste
    """
    y_true = np.array(y_true)
    correct_m1 = (np.array(y_pred_m1) == y_true)
    correct_m2 = (np.array(y_pred_m2) == y_true)
    
    b = np.sum(correct_m1 & ~correct_m2)
    c = np.sum(~correct_m1 & correct_m2)
    
    # Correction de continuité de Edwards
    chi2 = ((abs(b - c) - 1.0) ** 2) / (b + c) if (b + c) > 0 else 0.0
    from scipy.stats import chi2 as chi2_dist
    p_value = 1.0 - chi2_dist.cdf(chi2, df=1)
    
    return {
        "b (M1_correct_M2_wrong)": int(b),
        "c (M1_wrong_M2_correct)": int(c),
        "chi2": round(float(chi2), 4),
        "p_value": round(float(p_value), 5),
        "significant": bool(p_value < 0.05)
    }
