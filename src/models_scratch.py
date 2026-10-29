import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BiLSTM_Classifier(nn.Module):
    """C1: BiLSTM Standard"""
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=3, pretrained_embeddings=None, freeze=False):
        super().__init__()
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=freeze)
            embed_dim = pretrained_embeddings.shape[1]
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len)
        emb = self.dropout(self.embedding(x)) # (B, T, D)
        out, (h_n, c_n) = self.lstm(emb)       # h_n: (2, B, hidden_dim)
        # Concatener les deux derniers etats caches (aller et retour)
        h_cat = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1) # (B, 2*hidden_dim)
        logits = self.fc(self.dropout(h_cat))
        return logits


class Attention(nn.Module):
    """Mecanisme d'attention Bahdanau fait main (C2)"""
    def __init__(self, hidden_dim, attention_dim=64):
        super().__init__()
        self.W = nn.Linear(hidden_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, lstm_outputs, mask=None):
        # lstm_outputs: (B, T, 2*hidden_dim)
        # u: (B, T, attention_dim)
        u = torch.tanh(self.W(lstm_outputs))
        # scores: (B, T, 1)
        scores = self.v(u)
        
        if mask is not None:
            # Remplacer les scores du padding par une valeur tres negative (-1e9)
            scores = scores.masked_fill(~mask.unsqueeze(-1), -1e9)
            
        alpha = F.softmax(scores, dim=1) # (B, T, 1)
        context = torch.sum(alpha * lstm_outputs, dim=1) # (B, 2*hidden_dim)
        return context, alpha.squeeze(-1) # alpha: (B, T)


class BiLSTM_Attention_Classifier(nn.Module):
    """C2: BiLSTM + Attention fait main"""
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=3, pretrained_embeddings=None, freeze=False):
        super().__init__()
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=freeze)
            embed_dim = pretrained_embeddings.shape[1]
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.attention = Attention(hidden_dim * 2, attention_dim=64)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        mask = (x != 0) # (B, T)
        emb = self.dropout(self.embedding(x))
        out, _ = self.lstm(emb) # (B, T, 2*hidden_dim)
        context, alpha = self.attention(out, mask=mask) # context: (B, 2*hidden_dim)
        logits = self.fc(self.dropout(context))
        return logits, alpha


class PositionalEncoding(nn.Module):
    """Encodage positionnel sinusoidal standard (Transformer)"""
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, D)
        return x + self.pe[:, :x.size(1)]


class MiniTransformer_Classifier(nn.Module):
    """C3: Mini-Encodeur Transformer (2 couches, d_model=128, 4 tetes)"""
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, num_classes=3, use_pos_encoding=True):
        super().__init__()
        self.use_pos_encoding = use_pos_encoding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model) if use_pos_encoding else None
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.2,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B, T)
        pad_mask = (x == 0) # True for padded positions
        emb = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        if self.use_pos_encoding:
            emb = self.pos_encoder(emb)
        emb = self.dropout(emb)
        
        out = self.transformer_encoder(emb, src_key_padding_mask=pad_mask) # (B, T, D)
        
        # Masked average pooling pour obtenir le vecteur de phrase
        mask = (~pad_mask).unsqueeze(-1).float() # (B, T, 1)
        sum_embeddings = torch.sum(out * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask # (B, D)
        
        logits = self.fc(self.dropout(pooled))
        return logits


class CharCNN_Classifier(nn.Module):
    """C4: Char-CNN (Convolutions 1D sur caracteres)"""
    def __init__(self, num_chars, embed_dim=64, num_filters=64, filter_sizes=[3, 5, 7], num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(num_chars, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        # x: (B, seq_len)
        emb = self.embedding(x).transpose(1, 2) # (B, embed_dim, seq_len)
        pooled_outputs = []
        for conv in self.convs:
            c = F.relu(conv(emb)) # (B, num_filters, L_out)
            p = F.max_pool1d(c, kernel_size=c.shape[2]).squeeze(2) # (B, num_filters)
            pooled_outputs.append(p)
            
        h_pool = torch.cat(pooled_outputs, dim=1) # (B, num_filters * 3)
        logits = self.fc(self.dropout(h_pool))
        return logits
