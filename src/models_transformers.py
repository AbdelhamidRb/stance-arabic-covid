import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

# ==============================================================================
# 1. ARCHITECTURE D2 : AraBERT GELÉ + TÊTE BiLSTM + ATTENTION (de C2)
# ==============================================================================
class Attention(nn.Module):
    """Mécanisme d'attention additive (Bahdanau) pour pondérer les états de BERT"""
    def __init__(self, hidden_dim, attention_dim=64):
        super().__init__()
        self.W = nn.Linear(hidden_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, rnn_outputs, mask=None):
        u = torch.tanh(self.W(rnn_outputs))
        scores = self.v(u)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), -1e9)
        alpha = F.softmax(scores, dim=1)
        context = torch.sum(alpha * rnn_outputs, dim=1)
        return context, alpha.squeeze(-1)


class FrozenBERT_BiLSTMAttention(nn.Module):
    """
    Partie D2 : Modèle hybride associant un PLM gelé (AraBERT)
    et une tête séquentielle BiLSTM + Attention.
    
    Principe :
    - Le corps BERT extrait des représentations riches de 768 dimensions pour chaque token.
    - Ses poids sont 100% gelés (requires_grad = False) pour aller vite.
    - La tête BiLSTM + Attention apprend à arbitrer la position sur ces représentations.
    """
    def __init__(self, pretrained_model_name, hidden_dim=128, num_classes=3):
        super().__init__()
        # 1. Chargement du corps BERT pré-entraîné
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        
        # 2. GEL DES POIDS DE BERT (Mode Frozen)
        # Aucun gradient n'est calculé pour BERT -> gain de mémoire et de temps colossal !
        for param in self.bert.parameters():
            param.requires_grad = False
            
        bert_hidden_size = self.bert.config.hidden_size # 768 pour bert-base
        
        # 3. TÊTE BiLSTM : prend les 768 dimensions de BERT et les compresse en 2x128
        self.lstm = nn.LSTM(
            input_size=bert_hidden_size,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        
        # 4. TÊTE D'ATTENTION (sur les 256 dimensions du BiLSTM)
        self.attention = Attention(hidden_dim * 2, attention_dim=64)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        # 1. Extraction des états cachés par BERT (sans calcul de gradients)
        with torch.no_grad():
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            # sequence_output contient tous les tokens : forme (batch_size, seq_len, 768)
            sequence_output = bert_outputs.last_hidden_state
            
        # 2. Passage dans le BiLSTM entraînable
        lstm_out, _ = self.lstm(sequence_output) # forme : (batch_size, seq_len, 256)
        
        # 3. Calcul de l'Attention (on masque les tokens de padding via attention_mask)
        mask = (attention_mask == 1)
        context, alpha = self.attention(lstm_out, mask=mask)
        
        # 4. Classification finale
        logits = self.fc(self.dropout(context))
        return logits, alpha


# ==============================================================================
# 2. ARCHITECTURE E2 : MODÈLE MULTI-TÂCHE (Multi-Task Learning)
# ==============================================================================
class MultiTaskArabicTransformer(nn.Module):
    """
    Partie E2 : Modèle Transformer Multi-Tâche unifié.
    
    Partage un corps Transformer unique (AraBERT ou MARBERTv2) et possède
    3 têtes de classification indépendantes :
      - Tête 1 : Position / Stance (3 classes : Against=0, Favor=1, None=2)
      - Tête 2 : Sentiment (3 classes : Negative=0, Positive=1, Neutral=2)
      - Tête 3 : Sarcasme (2 classes : No=0, Yes=1)
      
    Fonction de perte totale :
      L_totale = L_stance + lambda1 * L_sentiment + lambda2 * L_sarcasme
    """
    def __init__(self, pretrained_model_name, num_stance=3, num_sentiment=3, num_sarcasm=2, dropout_prob=0.3):
        super().__init__()
        self.config = AutoConfig.from_pretrained(pretrained_model_name)
        self.transformer = AutoModel.from_pretrained(pretrained_model_name)
        
        hidden_size = self.config.hidden_size # 768
        self.dropout = nn.Dropout(dropout_prob)
        
        # Tête 1 : Position (Tâche Principale)
        self.stance_head = nn.Linear(hidden_size, num_stance)
        
        # Tête 2 : Sentiment (Tâche Auxiliaire 1)
        self.sentiment_head = nn.Linear(hidden_size, num_sentiment)
        
        # Tête 3 : Sarcasme (Tâche Auxiliaire 2)
        self.sarcasm_head = nn.Linear(hidden_size, num_sarcasm)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        stance_labels=None,
        sentiment_labels=None,
        sarcasm_labels=None,
        lambda_sentiment=0.3,
        lambda_sarcasm=0.3,
        stance_class_weights=None
    ):
        # 1. Passage dans le corps Transformer partagé
        if token_type_ids is not None:
            outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
            
        # Récupération du vecteur de synthèse [CLS] (indice 0)
        cls_rep = self.dropout(outputs.last_hidden_state[:, 0, :])
        
        # 2. Calcul des prédictions des 3 têtes
        stance_logits = self.stance_head(cls_rep)
        sentiment_logits = self.sentiment_head(cls_rep)
        sarcasm_logits = self.sarcasm_head(cls_rep)
        
        # 3. Calcul éventuel de la perte globale multi-tâche
        total_loss = None
        if stance_labels is not None:
            # Perte Stance (principale)
            loss_fn_stance = nn.CrossEntropyLoss(weight=stance_class_weights)
            loss_stance = loss_fn_stance(stance_logits, stance_labels)
            total_loss = loss_stance
            
            # Perte Sentiment (auxiliaire 1)
            if sentiment_labels is not None and lambda_sentiment > 0.0:
                loss_fn_sent = nn.CrossEntropyLoss()
                loss_sent = loss_fn_sent(sentiment_logits, sentiment_labels)
                total_loss = total_loss + (lambda_sentiment * loss_sent)
                
            # Perte Sarcasme (auxiliaire 2)
            if sarcasm_labels is not None and lambda_sarcasm > 0.0:
                loss_fn_sarc = nn.CrossEntropyLoss()
                loss_sarc = loss_fn_sarc(sarcasm_logits, sarcasm_labels)
                total_loss = total_loss + (lambda_sarcasm * loss_sarc)
                
        return {
            "loss": total_loss,
            "stance_logits": stance_logits,
            "sentiment_logits": sentiment_logits,
            "sarcasm_logits": sarcasm_logits
        }
