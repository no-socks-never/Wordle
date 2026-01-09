# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==================== 辅助模块 ====================

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        attn_weights = F.softmax(self.attn(x), dim=1)
        context = torch.sum(attn_weights * x, dim=1)
        return context, attn_weights

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# ==================== 核心模型 ====================

class BaseLSTMModel(nn.Module):
    def __init__(self, vocab_size, num_numeric_features, word_emb_dim=32, hidden_dim=64, output_dim=7):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, word_emb_dim)
        # 即使对 5 个字母取平均，维度依然是 word_emb_dim
        input_dim = word_emb_dim + num_numeric_features
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, word_seq, numeric_seq):
        # word_seq: [B, L, 5] -> emb: [B, L, 5, E]
        emb = self.word_emb(word_seq)
        # 【核心修复】：在字符维度（dim=2）取平均，得到单词嵌入 [B, L, E]
        emb = emb.mean(dim=2) 
        
        # 现在两者都是 3 维，可以拼接
        x = torch.cat([emb, numeric_seq], dim=-1)
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return F.softmax(out, dim=-1)

class BiLSTMAttentionModel(nn.Module):
    def __init__(self, vocab_size, num_numeric_features, word_emb_dim=32, hidden_dim=64, output_dim=7):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, word_emb_dim)
        input_dim = word_emb_dim + num_numeric_features
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = TemporalAttention(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, word_seq, numeric_seq):
        emb = self.word_emb(word_seq).mean(dim=2) # [B, L, E]
        x = torch.cat([emb, numeric_seq], dim=-1)
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        out = self.fc(context)
        return F.softmax(out, dim=-1)

class TransformerTimeModel(nn.Module):
    def __init__(self, vocab_size, num_numeric_features, d_model=64, nhead=4, num_layers=2, word_emb_dim=32, output_dim=7):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, word_emb_dim)
        input_dim = word_emb_dim + num_numeric_features
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, word_seq, numeric_seq):
        emb = self.word_emb(word_seq).mean(dim=2) # [B, L, E]
        x = torch.cat([emb, numeric_seq], dim=-1)
        x = self.input_linear(x)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        out = self.fc(x[:, -1, :])
        return F.softmax(out, dim=-1)