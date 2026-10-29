import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================================================================
# 1. ARCHITECTURE C1 : BiLSTM STANDARD
# ==============================================================================
class BiLSTM_Classifier(nn.Module):
    """
    Réseau récurrent bidirectionnel (BiLSTM) standard pour la classification de texte.
    
    Principe :
    Le modèle lit le tweet dans les deux sens :
      1. De gauche à droite (sens normal) pour comprendre le contexte passé.
      2. De droite à gauche (sens inverse) pour comprendre le contexte futur.
    Les deux représentations finales sont concaténées pour prendre la décision.
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=3, pretrained_embeddings=None, freeze=False):
        super().__init__()
        
        # 1. COUCHE D'EMBEDDING (Plongement lexical) :
        # Transforme un identifiant de mot (ex: le mot n°45) en un vecteur continu de taille `embed_dim` (ex: 128 nombres).
        # padding_idx=0 signifie que le token de remplissage <PAD> aura toujours un vecteur de zéros.
        if pretrained_embeddings is not None:
            # Si on fournit des vecteurs pré-entraînés (ex: AraVec)
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=freeze)
            embed_dim = pretrained_embeddings.shape[1]
        else:
            # Sinon, initialisation aléatoire (appris de zéro par le réseau)
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            
        # 2. COUCHE LSTM BIDIRECTIONNELLE :
        # - embed_dim : taille du vecteur d'entrée reçu de l'embedding (128)
        # - hidden_dim : taille de l'état caché interne de chaque LSTM (128)
        # - bidirectional=True : active deux LSTMs parallèles (un aller et un retour)
        # - batch_first=True : les tenseurs ont la forme (taille_du_batch, longueur_de_phrase, dimensions)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        
        # 3. RÉGULARISATION (Dropout) :
        # Désactive aléatoirement 30% des neurones pendant l'entraînement pour éviter le surapprentissage (overfitting)
        self.dropout = nn.Dropout(0.3)
        
        # 4. COUCHE DE CLASSIFICATION (Dense / Linéaire) :
        # Comme le LSTM est bidirectionnel, la sortie finale concatène l'aller (128) et le retour (128) = 256.
        # On projette ces 256 valeurs vers les 3 classes cibles (Against=0, Favor=1, None=2).
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x est un tenseur d'indices de mots de forme : (batch_size, seq_len)
        # Exemple : 32 tweets de 100 mots chacun -> forme (32, 100)
        
        # Étape A : Transformation des indices en vecteurs denses
        # emb aura la forme : (batch_size, seq_len, embed_dim) -> ex: (32, 100, 128)
        emb = self.dropout(self.embedding(x))
        
        # Étape B : Passage dans le BiLSTM
        # out : contient tous les états cachés intermédiaires pour chaque mot -> (32, 100, 256)
        # h_n : contient le dernier état caché final pour chaque direction -> (num_directions=2, batch_size=32, hidden_dim=128)
        out, (h_n, c_n) = self.lstm(emb)
        
        # Étape C : Concaténation des deux derniers états cachés
        # h_n[-2] : état final du parcours gauche -> droite (taille 128)
        # h_n[-1] : état final du parcours droite -> gauche (taille 128)
        # h_cat aura la forme : (batch_size, 256)
        h_cat = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        
        # Étape D : Calcul des logits (scores non normalisés pour les 3 classes)
        # logits aura la forme : (batch_size, 3)
        logits = self.fc(self.dropout(h_cat))
        
        return logits


# ==============================================================================
# 2. ARCHITECTURE C2 : MÉCANISME D'ATTENTION (BAHDANAU) FAIT MAIN
# ==============================================================================
class Attention(nn.Module):
    """
    Mécanisme d'attention additive (type Bahdanau) implémenté mathématiquement de zéro.
    
    Problème résolu :
    Un LSTM classique ne garde que le dernier état et a tendance à oublier les mots du début.
    L'Attention calcule un poids d'importance alpha_t pour CHAQUE mot du tweet,
    puis fait la moyenne pondérée de tous les mots pour créer le vecteur de contexte final.
    """
    def __init__(self, hidden_dim, attention_dim=64):
        super().__init__()
        # W projette l'état caché du LSTM (ex: 256) vers un espace intermédiaire (64)
        self.W = nn.Linear(hidden_dim, attention_dim)
        # v projette cet espace vers un score scalaire unique par mot (1 nombre)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, lstm_outputs, mask=None):
        # lstm_outputs : sortie du LSTM pour chaque mot, forme (batch_size, seq_len, 256)
        # mask : tenseur booléen (True si mot réel, False si mot de padding <PAD>)
        
        # Étape 1 : Calcul de la projection non-linéaire u_t = tanh(W * h_t)
        # u aura la forme : (batch_size, seq_len, 64)
        u = torch.tanh(self.W(lstm_outputs))
        
        # Étape 2 : Calcul du score d'alignement brut e_t = v^T * u_t
        # scores aura la forme : (batch_size, seq_len, 1)
        scores = self.v(u)
        
        # Étape 3 : Masquage des tokens de padding <PAD>
        # On remplace les scores des tokens <PAD> par -1 milliard (-1e9)
        # Ainsi, lors du Softmax, exp(-1e9) donnera exactement 0% d'attention aux paddings !
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), -1e9)
            
        # Étape 4 : Normalisation probabiliste (Softmax sur la dimension du temps / mots)
        # alpha_t représente le pourcentage d'attention accordé à chaque mot (somme = 1.0)
        # alpha aura la forme : (batch_size, seq_len, 1)
        alpha = F.softmax(scores, dim=1)
        
        # Étape 5 : Somme pondérée de tous les mots (Vecteur de Contexte)
        # context = somme sur t de (alpha_t * h_t)
        # context aura la forme : (batch_size, 256)
        context = torch.sum(alpha * lstm_outputs, dim=1)
        
        # On retourne le vecteur de contexte et les poids d'attention (pour pouvoir les visualiser !)
        return context, alpha.squeeze(-1)


class BiLSTM_Attention_Classifier(nn.Module):
    """
    C2 : BiLSTM combiné avec notre couche d'Attention fait main.
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=3, pretrained_embeddings=None, freeze=False):
        super().__init__()
        
        # Embedding
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=freeze)
            embed_dim = pretrained_embeddings.shape[1]
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            
        # LSTM bidirectionnel
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        
        # Notre couche d'attention personnalisée (reçoit les 256 dimensions du BiLSTM)
        self.attention = Attention(hidden_dim * 2, attention_dim=64)
        
        self.dropout = nn.Dropout(0.3)
        # Classification finale à partir du vecteur de contexte
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # Création du masque : True pour les vrais mots, False pour le padding (qui a l'indice 0)
        mask = (x != 0)
        
        # 1. Embedding
        emb = self.dropout(self.embedding(x))
        
        # 2. Passage dans le BiLSTM (out contient les représentations de chaque mot)
        out, _ = self.lstm(emb)
        
        # 3. Calcul de l'Attention : sélectionne dynamiquement les mots clés du tweet
        context, alpha = self.attention(out, mask=mask)
        
        # 4. Prédiction des 3 classes
        logits = self.fc(self.dropout(context))
        
        return logits, alpha


# ==============================================================================
# 3. ARCHITECTURE C3 : MINI-TRANSFORMER & ENCODAGE POSITIONNEL
# ==============================================================================
class PositionalEncoding(nn.Module):
    """
    Encodage positionnel sinusoïdal standard (Vaswani et al., Attention Is All You Need).
    
    Pourquoi est-ce indispensable ?
    Contrairement au LSTM qui lit mot après mot, le Transformer traite tous les mots en même temps !
    Sans cet encodage, le réseau ne sait pas qui est le 1er ou le 10e mot.
    L'encodage positionnel additionne une onde sinusoïdale unique à chaque position pour coder l'ordre.
    """
    def __init__(self, d_model, max_len=256):
        super().__init__()
        # Matrice de zéros pour stocker les encodages : (max_len=256, d_model=128)
        pe = torch.zeros(max_len, d_model)
        # Vecteur de positions : [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Facteur d'échelle géométrique pour faire varier la fréquence selon la dimension
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Fonctions sinus sur les indices pairs (0, 2, 4...)
        pe[:, 0::2] = torch.sin(position * div_term)
        # Fonctions cosinus sur les indices impairs (1, 3, 5...)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Ajout d'une dimension de batch fictive (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        # register_buffer informe PyTorch que ce tenseur fait partie du modèle mais n'a pas de gradients (fixe)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x : (batch_size, seq_len, d_model)
        # On additionne simplement l'encodage correspondant à la longueur réelle de la séquence
        return x + self.pe[:, :x.size(1)]


class MiniTransformer_Classifier(nn.Module):
    """
    C3 : Mini-Encodeur Transformer (2 couches, d_model=128, 4 têtes d'attention).
    Comprend une option `use_pos_encoding` pour réaliser l'étude d'ablation demandée !
    """
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, num_classes=3, use_pos_encoding=True):
        super().__init__()
        self.use_pos_encoding = use_pos_encoding
        
        # Couche d'embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Encodage positionnel (actif ou désactivé pour l'ablation study)
        self.pos_encoder = PositionalEncoding(d_model) if use_pos_encoding else None
        
        # Définition d'une couche Transformer standard :
        # - d_model=128 : taille des vecteurs de représentation
        # - nhead=4 : 4 têtes d'attention en parallèle (chacune de taille 128 / 4 = 32)
        # - dim_feedforward=256 : taille de la couche dense interne
        # - batch_first=True : entrée au format (batch, seq, feature)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.2,
            batch_first=True
        )
        # Empilement de 2 couches d'encodeur
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(0.3)
        # Couche de classification finale
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x : (batch_size, seq_len)
        
        # Masque de padding pour le Transformer (True où se trouvent les <PAD>, pour les ignorer dans l'attention)
        pad_mask = (x == 0)
        
        # 1. Embedding multiplié par sqrt(d_model) comme dans le papier Transformer original pour stabiliser la variance
        emb = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        
        # 2. Injection de l'ordre des mots (si activé)
        if self.use_pos_encoding and self.pos_encoder is not None:
            emb = self.pos_encoder(emb)
            
        emb = self.dropout(emb)
        
        # 3. Passage à travers les couches de Self-Attention du Transformer
        # out aura la forme : (batch_size, seq_len, d_model)
        out = self.transformer_encoder(emb, src_key_padding_mask=pad_mask)
        
        # 4. Moyenne globale pondérée (Global Average Pooling) excluant les paddings
        # On transforme la séquence de mots en un seul vecteur résumant le tweet
        mask = (~pad_mask).unsqueeze(-1).float() # (batch_size, seq_len, 1)
        sum_embeddings = torch.sum(out * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask # (batch_size, d_model)
        
        # 5. Logits de classification
        logits = self.fc(self.dropout(pooled))
        return logits


# ==============================================================================
# 4. ARCHITECTURE C4 : Char-CNN (CONVOLUTIONS SUR CARACTÈRES)
# ==============================================================================
class CharCNN_Classifier(nn.Module):
    """
    C4 : Réseau convolutif opérant directement au niveau des lettres/caractères.
    
    Pourquoi est-ce puissant en arabe ?
    Parce qu'il ne découpe pas en mots : il fait glisser des filtres de différentes tailles
    (3, 5, 7 lettres) sur les caractères. Il capte les racines arabes, les clitiques (préfixes/suffixes)
    et reste totalement insensible aux fautes d'orthographe ou au dialecte !
    """
    def __init__(self, num_chars, embed_dim=64, num_filters=64, filter_sizes=[3, 5, 7], num_classes=3):
        super().__init__()
        
        # Embedding de chaque caractère individuel (ex: dimension 64)
        self.embedding = nn.Embedding(num_chars, embed_dim, padding_idx=0)
        
        # Liste de couches de convolutions 1D :
        # - Filtre taille 3 : regarde des fenêtres de 3 lettres consécutives (tri-grammes)
        # - Filtre taille 5 : regarde des fenêtres de 5 lettres
        # - Filtre taille 7 : regarde des fenêtres de 7 lettres
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        self.dropout = nn.Dropout(0.3)
        # Après le Max-Pooling, on concatène les sorties des 3 filtres : 64 * 3 = 192 caractéristiques
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        # x : tenseur d'indices de caractères de forme (batch_size, seq_len_chars)
        
        # 1. Embedding : (batch_size, seq_len, embed_dim)
        # La Conv1D attend la dimension des canaux en 2e position : on transpose vers (batch_size, embed_dim, seq_len)
        emb = self.embedding(x).transpose(1, 2)
        
        pooled_outputs = []
        for conv in self.convs:
            # Application de la convolution + activation ReLU
            # c aura la forme : (batch_size, num_filters, seq_len - kernel_size + 1)
            c = F.relu(conv(emb))
            
            # Max-Pooling sur toute la longueur temporelle (1-Max Pooling)
            # Ne retient que le signal le plus fort détecté par chaque filtre n'importe où dans le texte
            p = F.max_pool1d(c, kernel_size=c.shape[2]).squeeze(2) # (batch_size, num_filters)
            pooled_outputs.append(p)
            
        # Concaténation des caractéristiques des filtres de taille 3, 5 et 7
        # h_pool aura la forme : (batch_size, 64 * 3 = 192)
        h_pool = torch.cat(pooled_outputs, dim=1)
        
        # Classification finale
        logits = self.fc(self.dropout(h_pool))
        return logits
