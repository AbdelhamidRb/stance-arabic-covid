import pandas as pd
import numpy as np
import re
from dataset import load_and_preprocess_mawqif
from metrics import compute_stance_metrics

train_df, val_df, _ = load_and_preprocess_mawqif()
y_val = val_df['stance'].values

# Hashtags de soutien
hashtag_pattern = re.compile(r'#\w+')
favor_hashtags = {}
other_hashtags = {}

for _, row in train_df.iterrows():
    tags = hashtag_pattern.findall(row['text'])
    for t in tags:
        if row['stance'] == 'Favor':
            favor_hashtags[t] = favor_hashtags.get(t, 0) + 1
        else:
            other_hashtags[t] = other_hashtags.get(t, 0) + 1
            
support_tags = set()
for tag, count in favor_hashtags.items():
    other_cnt = other_hashtags.get(tag, 0)
    total = count + other_cnt
    if total >= 2 and (count / total) >= 0.70:
        support_tags.add(tag)

print(f"Nombre support tags: {len(support_tags)}")

# Variante 1: hashtag -> Favor, sinon None
pred_tag_none = [
    "Favor" if len(set(hashtag_pattern.findall(t)).intersection(support_tags)) > 0 else "None"
    for t in val_df['text']
]
print("Hashtag -> Favor, sinon None:", compute_stance_metrics(y_val, pred_tag_none))

# Variante 2: hashtag -> Favor, sinon Against
pred_tag_against = [
    "Favor" if len(set(hashtag_pattern.findall(t)).intersection(support_tags)) > 0 else "Against"
    for t in val_df['text']
]
print("Hashtag -> Favor, sinon Against:", compute_stance_metrics(y_val, pred_tag_against))
