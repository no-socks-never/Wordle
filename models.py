# models.py
"""
模型定义：
- BaseLSTMModel：简单单向 LSTM
- BiLSTMAttentionModel：双向 LSTM + 时间注意力
- TransformerTimeModel：时间序列 Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    WORD_EMB_DIM,
    HIDDEN_DIM_LSTM,
    HIDDEN_DIM_TRANSFORMER,
    NUM_TRANSFORMER_LAYERS,
    NUM_ATTENTION_HEADS,
)


class BaseLSTMModel(nn.Module):
    """
    文本 + 数值特征 -> LSTM -> 全连接 -> 7 维分布
    """

    def __init__(
        self,
        vocab_size: int,
        num_numeric_features: int,
        word_emb_dim: int = WORD_EMB_DIM,
        hidden_dim: int = HIDDEN_DIM_LSTM,
        output_dim: int = 7,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_numeric_features = num_numeric_features

        self.word_emb = nn.Embedding(vocab_size, word_emb_dim)
        input_dim = word_emb_dim + num_numeric_features
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, word_seq, numeric_seq):
        """
        word_seq: [B, L] (int64)
        numeric_seq: [B, L, F]  (F 可以为 0)
        返回：
            dist: [B, 7]，softmax 后的概率分布
        """
        emb = self.word_emb(word_seq)  # [B,L,E]
        if numeric_seq is not None and numeric_seq.size(-1) > 0:
            x = torch.cat([emb, numeric_seq], dim=-1)
        else:
            x = emb

        out, (h_n, c_n) = self.lstm(x)
        last_hidden = out[:, -1, :]  # [B,H]
        logits = self.fc(last_hidden)  # [B,7]
        dist = F.softmax(logits, dim=-1)
        return dist


class TemporalAttention(nn.Module):
    """
    简单时间维度注意力：
    给定 h: [B, L, H]，返回加权和向量 [B, H] 和注意力权重 [B, L]
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        # h: [B,L,H]
        scores = self.attn(h).squeeze(-1)  # [B,L]
        alpha = F.softmax(scores, dim=-1)  # [B,L]
        context = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)  # [B,H]
        return context, alpha


class BiLSTMAttentionModel(nn.Module):
    """
    BiLSTM + Attention：
    - 输入与 BaseLSTM 相同（可加数值特征）
    - 输出 7 维分布，以及注意力权重方便画热力图
    """

    def __init__(
        self,
        vocab_size: int,
        num_numeric_features: int,
        word_emb_dim: int = WORD_EMB_DIM,
        hidden_dim: int = HIDDEN_DIM_LSTM,
        output_dim: int = 7,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_numeric_features = num_numeric_features

        self.word_emb = nn.Embedding(vocab_size, word_emb_dim)
        input_dim = word_emb_dim + num_numeric_features
        self.bilstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = TemporalAttention(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, word_seq, numeric_seq):
        emb = self.word_emb(word_seq)  # [B,L,E]
        if numeric_seq is not None and numeric_seq.size(-1) > 0:
            x = torch.cat([emb, numeric_seq], dim=-1)
        else:
            x = emb

        h, _ = self.bilstm(x)           # [B,L,2H]
        context, alpha = self.attention(h)  # [B,2H], [B,L]
        logits = self.fc(context)       # [B,7]
        dist = F.softmax(logits, dim=-1)
        return dist, alpha  # 方便后续画 attention 热力图


class PositionalEncoding(nn.Module):
    """
    标准正弦位置编码，用作 transformer 输入加法。
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [B, L, d_model]
        """
        L = x.size(1)
        return x + self.pe[:, :L, :]


class TransformerTimeModel(nn.Module):
    """
    时间序列 Transformer：
    - 输入：word embedding + 数值特征，通过线性层投到 d_model
    - 位置编码 + TransformerEncoder
    - 取最后一个时间步表示，映射到 7 维分布
    """

    def __init__(
        self,
        vocab_size: int,
        num_numeric_features: int,
        d_model: int = HIDDEN_DIM_TRANSFORMER,
        nhead: int = NUM_ATTENTION_HEADS,
        num_layers: int = NUM_TRANSFORMER_LAYERS,
        word_emb_dim: int = WORD_EMB_DIM,
        output_dim: int = 7,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_numeric_features = num_numeric_features

        self.word_emb = nn.Embedding(vocab_size, word_emb_dim)
        input_dim = word_emb_dim + num_numeric_features
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, word_seq, numeric_seq):
        emb = self.word_emb(word_seq)  # [B,L,E]
        if numeric_seq is not None and numeric_seq.size(-1) > 0:
            x = torch.cat([emb, numeric_seq], dim=-1)
        else:
            x = emb

        x = self.input_linear(x)          # [B,L,d_model]
        x = self.pos_encoding(x)          # [B,L,d_model]
        h = self.encoder(x)               # [B,L,d_model]
        last_h = h[:, -1, :]              # [B,d_model]
        logits = self.fc(last_h)          # [B,7]
        dist = F.softmax(logits, dim=-1)
        return dist


