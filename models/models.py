import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for sequence position information
    """
    def __init__(self, d_model, max_len=200):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :]


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with masking support
    """
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.fc_out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, hidden_size)
            mask: (batch, seq_len) - 1 for valid, 0 for padded
        Returns:
            out: (batch, seq_len, hidden_size)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections
        Q = self.query(x)  # (batch, seq_len, hidden_size)
        K = self.key(x)
        V = self.value(x)

        # Reshape for multi-head: (batch, seq_len, num_heads, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose: (batch, num_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Attention scores: (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply mask if provided
        if mask is not None:
            # Expand mask: (batch, 1, 1, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention: (batch, num_heads, seq_len, head_dim)
        out = torch.matmul(attention_weights, V)

        # Concatenate heads: (batch, seq_len, hidden_size)
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.hidden_size)

        # Final linear
        out = self.fc_out(out)

        return out, attention_weights


class LSTMAttentionEnhanced(nn.Module):
    """
    Enhanced LSTM + Multi-Head Attention for BTC prediction

    Features:
    - Bidirectional LSTM
    - Multi-head attention with masking
    - Positional encoding
    - Layer normalization
    - Residual connections
    """
    def __init__(
        self,
        input_size,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        dropout=0.3,
        num_classes=3,
        bidirectional=True
    ):
        super(LSTMAttentionEnhanced, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size, max_len=200)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        lstm_output_size = hidden_size * self.num_directions

        # Project LSTM output if bidirectional
        if bidirectional:
            self.lstm_proj = nn.Linear(lstm_output_size, hidden_size)
        else:
            self.lstm_proj = nn.Identity()

        # Multi-head attention
        self.attention = MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Classifier with residual
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, input_size)
            mask: (batch, seq_len) - 1 for valid, 0 for padded
        Returns:
            output: (batch, num_classes)
        """
        # Input projection
        x = self.input_proj(x)  # (batch, seq_len, hidden_size)

        # Add positional encoding
        x = self.pos_encoder(x)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size * num_directions)
        lstm_out = self.lstm_proj(lstm_out)  # (batch, seq_len, hidden_size)

        # Residual connection + Layer norm
        lstm_out = self.norm1(lstm_out + x)

        # Multi-head attention with mask
        attn_out, attention_weights = self.attention(lstm_out, mask=mask)

        # Residual connection + Layer norm
        attn_out = self.norm2(attn_out + lstm_out)

        # Global average pooling (consider mask)
        if mask is not None:
            # Weighted average by mask
            mask_expanded = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            pooled = (attn_out * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            pooled = attn_out.mean(dim=1)  # (batch, hidden_size)

        # Dropout
        pooled = self.dropout(pooled)

        # Classification
        output = self.fc(pooled)

        return output

    def get_attention_weights(self, x, mask=None):
        """Get attention weights for visualization"""
        with torch.no_grad():
            x = self.input_proj(x)
            x = self.pos_encoder(x)
            lstm_out, _ = self.lstm(x)
            lstm_out = self.lstm_proj(lstm_out)
            lstm_out = self.norm1(lstm_out + x)
            _, attention_weights = self.attention(lstm_out, mask=mask)
        return attention_weights.cpu().numpy()


class LSTMAttention(nn.Module):
    """
    Standard LSTM + Attention (with masking support)
    Simpler version for comparison
    """
    def __init__(
        self,
        input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        num_classes=3,
        bidirectional=True
    ):
        super(LSTMAttention, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        lstm_output_size = hidden_size * self.num_directions

        # Attention
        self.attention = nn.Linear(lstm_output_size, 1, bias=False)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, input_size)
            mask: (batch, seq_len) - 1 for valid, 0 for padded
        """
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size * num_directions)

        # Attention scores
        attention_scores = self.attention(lstm_out)  # (batch, seq_len, 1)

        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            attention_scores = attention_scores.masked_fill(mask_expanded == 0, float('-inf'))

        # Softmax
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, seq_len, 1)

        # Weighted sum
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden_size * num_directions)

        # Dropout
        context = self.dropout(context)

        # Classification
        output = self.fc(context)

        return output

    def get_attention_weights(self, x, mask=None):
        """Get attention weights for visualization"""
        with torch.no_grad():
            lstm_out, _ = self.lstm(x)
            attention_scores = self.attention(lstm_out)

            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                attention_scores = attention_scores.masked_fill(mask_expanded == 0, float('-inf'))

            attention_weights = F.softmax(attention_scores, dim=1)
        return attention_weights.squeeze(-1).cpu().numpy()


class SimpleLSTM(nn.Module):
    """
    Baseline LSTM without attention (for comparison)
    """
    def __init__(
        self,
        input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        num_classes=3
    ):
        super(SimpleLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x, mask=None):
        # LSTM
        lstm_out, (hidden, _) = self.lstm(x)

        # Use last hidden state
        output = self.dropout(hidden[-1])

        # Classification
        output = self.fc(output)

        return output


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    """
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing for better generalization
    """
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        """
        Args:
            pred: (batch, num_classes) - logits
            target: (batch,) - class labels
        """
        pred = pred.log_softmax(dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


# ==================== MODEL TESTING ====================
if __name__ == "__main__":
    print("Testing Enhanced LSTM + Attention Models...\n")

    # Config
    batch_size = 32
    seq_len = 96
    input_size = 50
    num_classes = 3

    # Dummy data
    x = torch.randn(batch_size, seq_len, input_size)
    y = torch.randint(0, num_classes, (batch_size,))
    mask = torch.ones(batch_size, seq_len)
    mask[:, -10:] = 0  # Mask last 10 timesteps

    print(f"Input shape:  {x.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Mask shape:   {mask.shape}\n")

    # Test Enhanced LSTM + Attention
    print("=" * 60)
    print("1. Enhanced LSTM + Multi-Head Attention")
    print("=" * 60)
    model1 = LSTMAttentionEnhanced(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        dropout=0.3,
        bidirectional=True
    )

    output1 = model1(x, mask=mask)
    print(f"Output shape: {output1.shape}")
    print(f"Parameters:   {sum(p.numel() for p in model1.parameters()):,}")

    attn1 = model1.get_attention_weights(x[:1], mask=mask[:1])
    print(f"Attention shape: {attn1.shape}")

    # Test Standard LSTM + Attention
    print("\n" + "=" * 60)
    print("2. Standard LSTM + Attention (with mask)")
    print("=" * 60)
    model2 = LSTMAttention(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        bidirectional=True
    )

    output2 = model2(x, mask=mask)
    print(f"Output shape: {output2.shape}")
    print(f"Parameters:   {sum(p.numel() for p in model2.parameters()):,}")

    attn2 = model2.get_attention_weights(x[:1], mask=mask[:1])
    print(f"Attention shape: {attn2.shape}")

    # Test Losses
    print("\n" + "=" * 60)
    print("3. Loss Functions")
    print("=" * 60)

    # Focal Loss
    alpha = torch.tensor([1.5, 1.0, 1.5])
    criterion_focal = FocalLoss(alpha=alpha, gamma=2.0)
    loss_focal = criterion_focal(output1, y)
    print(f"Focal Loss: {loss_focal.item():.4f}")

    # Label Smoothing
    criterion_smooth = LabelSmoothingLoss(num_classes=3, smoothing=0.1)
    loss_smooth = criterion_smooth(output1, y)
    print(f"Label Smoothing Loss: {loss_smooth.item():.4f}")

    # CrossEntropy
    criterion_ce = nn.CrossEntropyLoss(weight=alpha)
    loss_ce = criterion_ce(output1, y)
    print(f"CrossEntropy Loss: {loss_ce.item():.4f}")

    print("\n✅ All models working correctly!")
