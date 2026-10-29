import pandas as pd
import numpy as np
import re
import json
import sys

# Forcer l'encodage utf-8 pour les prints
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

train_df = pd.read_csv("data/Mawqif_AllTargets_Train.csv", keep_default_na=False, encoding="utf-8-sig")
test_df = pd.read_csv("data/Mawqif_AllTargets_Test.csv", keep_default_na=False, encoding="utf-8-sig")

# Normalisation arabe
def normalize_arabic(text):
    text = str(text)
    tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(tashkeel, '', text)
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'[ىي]', 'ي', text)
    text = re.sub(r'ـ', '', text)
    return re.sub(r'\s+', ' ', text).strip()

train_df['text_norm'] = train_df['text'].apply(normalize_arabic)
test_df['text_norm'] = test_df['text'].apply(normalize_arabic)

# Analyse du conflit
conflicts_exact = train_df.groupby(['text', 'target'])['stance'].nunique()
conflicting_groups_exact = conflicts_exact[conflicts_exact > 1]

conflict_details = []
for (text, target) in conflicting_groups_exact.index:
    rows = train_df[(train_df['text'] == text) & (train_df['target'] == target)]
    conflict_details.append({
        'target': target,
        'text': text,
        'stances': rows['stance'].tolist(),
        'confidences': rows['stance:confidence'].tolist(),
        'ids': rows['ID'].tolist() if 'ID' in rows else []
    })

# Analyse des chevauchements train/test
train_pairs = set(zip(train_df['text'], train_df['target']))
test_leakage = test_df[test_df.apply(lambda r: (r['text'], r['target']) in train_pairs, axis=1)]

audit_report = {
    "total_train": len(train_df),
    "total_test": len(test_df),
    "stance_train_dist": train_df['stance'].value_counts().to_dict(),
    "stance_test_dist": test_df['stance'].value_counts().to_dict(),
    "target_train_dist": train_df['target'].value_counts().to_dict(),
    "target_test_dist": test_df['target'].value_counts().to_dict(),
    "contradictory_train_cases": conflict_details,
    "leakage_test_count": len(test_leakage)
}

with open("results/audit_summary.json", "w", encoding="utf-8") as f:
    json.dump(audit_report, f, ensure_ascii=False, indent=2)

print("Audit summary sauvegardé dans results/audit_summary.json avec succès !")
print(json.dumps(conflict_details, ensure_ascii=False, indent=2))
