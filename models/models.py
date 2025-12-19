import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attention_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        return self.fc_out(out), attention_weights


class MultiInputBidirectionalLSTMAttention(nn.Module):
    """
    Multi-Input Bidirectional LSTM + Attention for BTC Price Prediction

    OPTIMIZED: 20 features across 3 groups
      - price_volume: 5 (OHLCV)
      - technical: 9 (EMA_21, MACD×3, RSI, ATR, ADX, BB_width, OBV)
      - regimes: 6 (hmm_regime_high, hmm_regime_duration, ema_alignment,
                     trend_strong_bull, volume_percentile, volume_trend)

    Output: 3 classes (DOWN=0, HOLD=1, UP=2)
    """

    def __init__(self, input_sizes, hidden_size=128, num_layers=2, num_heads=4, dropout=0.3, num_classes=3):
        super().__init__()

        self.input_sizes = input_sizes
        self.hidden_size = hidden_size
        self.input_branches = list(input_sizes.keys())

        # Input branches
        self.input_projections = nn.ModuleDict()
        self.branch_lstms = nn.ModuleDict()
        self.branch_projections = nn.ModuleDict()

        for branch_name, input_size in input_sizes.items():
            self.input_projections[branch_name] = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

            self.branch_lstms[branch_name] = nn.LSTM(
                hidden_size, hidden_size, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )

            self.branch_projections[branch_name] = nn.Linear(hidden_size * 2, hidden_size)

        # Fusion layer
        total_concat_size = hidden_size * len(input_sizes)
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_concat_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Attention
        self.pos_encoder = PositionalEncoding(hidden_size)
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, inputs):
        branch_outputs = []

        for branch_name in self.input_branches:
            x = inputs[branch_name]
            x = self.input_projections[branch_name](x)
            lstm_out, _ = self.branch_lstms[branch_name](x)
            lstm_out = self.branch_projections[branch_name](lstm_out)
            branch_outputs.append(lstm_out)

        concatenated = torch.cat(branch_outputs, dim=-1)
        fused = self.fusion_layer(concatenated)
        fused = self.pos_encoder(fused)
        fused = self.norm1(fused)

        attn_out, _ = self.attention(fused)
        attn_out = self.norm2(attn_out + fused)

        pooled = attn_out.mean(dim=1)
        pooled = self.dropout(pooled)
        output = self.classifier(pooled)

        return output

    def get_attention_weights(self, inputs):
        with torch.no_grad():
            branch_outputs = []
            for branch_name in self.input_branches:
                x = inputs[branch_name]
                x = self.input_projections[branch_name](x)
                lstm_out, _ = self.branch_lstms[branch_name](x)
                lstm_out = self.branch_projections[branch_name](lstm_out)
                branch_outputs.append(lstm_out)

            concatenated = torch.cat(branch_outputs, dim=-1)
            fused = self.fusion_layer(concatenated)
            fused = self.pos_encoder(fused)
            fused = self.norm1(fused)
            _, attention_weights = self.attention(fused)

        return attention_weights.cpu().numpy()


if __name__ == "__main__":
    batch_size = 32
    seq_len = 48

    # OPTIMIZED: 20 total features (6 regime features only)
    input_sizes = {
        'price_volume': 5,
        'technical': 9,
        'regimes': 6
    }

    dummy_inputs = {
        'price_volume': torch.randn(batch_size, seq_len, 5),
        'technical': torch.randn(batch_size, seq_len, 9),
        'regimes': torch.randn(batch_size, seq_len, 6)
    }
    dummy_targets = torch.randint(0, 3, (batch_size,))

    model = MultiInputBidirectionalLSTMAttention(input_sizes=input_sizes)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params:,}")

    output = model(dummy_inputs)
    print(f"Output shape: {output.shape}")

    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, dummy_targets)
    loss.backward()
    print(f"Loss: {loss.item():.4f}")

    attn_weights = model.get_attention_weights(dummy_inputs)
    print(f"Attention shape: {attn_weights.shape}")

    print("\nAll tests passed")
