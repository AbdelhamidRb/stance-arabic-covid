import pandas as pd
import numpy as np
import re
import unicodedata

# 1. Chargement avec correction de l'artefact
train_df = pd.read_csv("data/Mawqif_AllTargets_Train.csv", keep_default_na=False, encoding="utf-8-sig")
test_df = pd.read_csv("data/Mawqif_AllTargets_Test.csv", keep_default_na=False, encoding="utf-8-sig")

# Regex pour entités
url_pattern = re.compile(r'https?://\S+|www\.\S+')
mention_pattern = re.compile(r'@\w+')
hashtag_pattern = re.compile(r'#\w+')

# Regex pour Emojis (plages Unicode des emojis et symboles graphiques)
emoji_pattern = re.compile(
    r'[\U00010000-\U0010ffff]'
    r'|[\u2600-\u27bf]'
    r'|[\u2300-\u23ff]'
    r'|[\u2b50-\u2b55]'
    r'|[\u200d]'
)

def count_entities(df):
    stats = []
    for target, group in df.groupby('target'):
        n_tweets = len(group)
        urls = group['text'].apply(lambda x: len(url_pattern.findall(x))).sum()
        mentions = group['text'].apply(lambda x: len(mention_pattern.findall(x))).sum()
        hashtags = group['text'].apply(lambda x: len(hashtag_pattern.findall(x))).sum()
        emojis = group['text'].apply(lambda x: len(emoji_pattern.findall(x))).sum()
        stats.append({
            'Target': target,
            'Tweets': n_tweets,
            'Total_URLs': urls,
            'Avg_URLs': round(urls / n_tweets, 2),
            'Total_Mentions': mentions,
            'Avg_Mentions': round(mentions / n_tweets, 2),
            'Total_Hashtags': hashtags,
            'Avg_Hashtags': round(hashtags / n_tweets, 2),
            'Total_Emojis': emojis,
            'Avg_Emojis': round(emojis / n_tweets, 2),
        })
    return pd.DataFrame(stats)

print("=== 1. COMPTE DES ENTITES PAR CIBLE (TRAIN) ===")
stats_df = count_entities(train_df)
print(stats_df.to_string(index=False))

# 2. Doublons exacts
print("\n=== 2. DOUBLONS EXACTS ===")
# Dans train
exact_dup_train = train_df.duplicated(subset=['text', 'target'], keep=False)
print(f"Doublons exacts (texte + cible) dans train: {exact_dup_train.sum()} lignes concernées")

# Entre train et test
train_texts = set(zip(train_df['text'], train_df['target']))
test_overlap = test_df.apply(lambda row: (row['text'], row['target']) in train_texts, axis=1)
print(f"Chevauchement exact (texte + cible) entre test et train: {test_overlap.sum()} lignes")

# 3. Normalisation du texte arabe
def normalize_arabic(text):
    text = str(text)
    # Suppression tashkeel (diacritiques)
    tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(tashkeel, '', text)
    # Normalisation alef
    text = re.sub(r'[إأآا]', 'ا', text)
    # Normalisation teh marbuta
    text = re.sub(r'ة', 'ه', text)
    # Normalisation yeh
    text = re.sub(r'[ىي]', 'ي', text)
    # Suppression tatweel (kashida)
    text = re.sub(r'ـ', '', text)
    # Nettoyage des espaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

train_df['text_norm'] = train_df['text'].apply(normalize_arabic)
test_df['text_norm'] = test_df['text'].apply(normalize_arabic)

print("\n=== 3. DOUBLONS APRES NORMALISATION ARABE ===")
norm_dup_train = train_df.duplicated(subset=['text_norm', 'target'], keep=False)
print(f"Doublons après normalisation (texte_norm + cible) dans train: {norm_dup_train.sum()} lignes concernées")

train_norm_pairs = set(zip(train_df['text_norm'], train_df['target']))
test_norm_overlap = test_df.apply(lambda row: (row['text_norm'], row['target']) in train_norm_pairs, axis=1)
print(f"Chevauchement après normalisation entre test et train: {test_norm_overlap.sum()} lignes")

# 4. Doublons avec labels contradictoires
print("\n=== 4. DOUBLONS AUX LABELS CONTRADICTOIRES ===")
conflicts_exact = train_df.groupby(['text', 'target'])['stance'].nunique()
conflicting_groups_exact = conflicts_exact[conflicts_exact > 1]
print(f"Groupes (texte exact, cible) ayant des stances contradictoires dans train: {len(conflicting_groups_exact)}")

conflicts_norm = train_df.groupby(['text_norm', 'target'])['stance'].nunique()
conflicting_groups_norm = conflicts_norm[conflicts_norm > 1]
print(f"Groupes (texte normalisé, cible) ayant des stances contradictoires dans train: {len(conflicting_groups_norm)}")

if len(conflicting_groups_norm) > 0:
    print("\nExemple de conflit après normalisation:")
    for group_key in conflicting_groups_norm.index[:3]:
        sample = train_df[(train_df['text_norm'] == group_key[0]) & (train_df['target'] == group_key[1])]
        print(f"Target: {group_key[1]}")
        print(f"Text: {group_key[0][:80]}...")
        print(f"Stances associées: {sample['stance'].tolist()}")
        print(f"Confiances: {sample['stance:confidence'].tolist()}")
        print("-" * 40)
