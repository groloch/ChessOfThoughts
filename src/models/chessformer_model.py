from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ChessFormerConfig


def init_xavier(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias


class SmolgenModule(nn.Module):
    """
    Smolgen: Dynamic positional attention enhancement
    Generates supplemental attention logits based on current position
    """
    def __init__(self, hidden_size: int, num_heads: int, compress_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.compress_size = compress_size
        
        self.position_compress = nn.Sequential(
            nn.Linear(64 * 32, compress_size),
            nn.ReLU(),
            nn.LayerNorm(compress_size)
        )
        
        self.head_projections = nn.ModuleList([
            nn.Linear(compress_size, compress_size) for _ in range(num_heads)
        ])
        
        self.attention_logits_proj = nn.Linear(compress_size, 64 * 64)  # 64x64 attention matrix
        
        self.token_compress = nn.Linear(hidden_size, 32, bias=False)

        self.apply(init_xavier)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, 64, hidden_size] - token representations
        Returns:
            supplemental_logits: [batch_size, num_heads, 64, 64] - additional attention logits
        """
        batch_size = x.size(0)
        
        compressed_tokens = self.token_compress(x)  # [batch_size, 64, 32]
        flattened = compressed_tokens.view(batch_size, -1)  # [batch_size, 64*32]
        position_vector = self.position_compress(flattened)  # [batch_size, compress_size]
        
        supplemental_logits = []
        for head_proj in self.head_projections:
            head_vector = head_proj(position_vector)  # [batch_size, compress_size]
            logits = self.attention_logits_proj(head_vector)  # [batch_size, 64*64]
            logits = logits.view(batch_size, 64, 64)  # [batch_size, 64, 64]
            supplemental_logits.append(logits)
        
        return torch.stack(supplemental_logits, dim=1)  # [batch_size, num_heads, 64, 64]


class ChessMultiHeadAttention(nn.Module):
    """
    Multi-head attention with chess-specific enhancements:
    1. Trainable positional bias
    2. Smolgen dynamic attention
    3. No bias in QKV projections (for efficiency)
    """
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int = 32, use_smolgen: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.use_smolgen = use_smolgen
        
        self.query = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.key = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.value = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_size)
        
        self.positional_bias = nn.Parameter(torch.zeros(num_heads, 64, 64))
        
        if use_smolgen:
            self.smolgen = SmolgenModule(hidden_size, num_heads)
        
        self.scale = (head_dim) ** -0.5

        self.apply(init_xavier)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        scores = scores + self.positional_bias.unsqueeze(0)
        
        if self.use_smolgen:
            smolgen_logits = self.smolgen(x)  # [batch_size, num_heads, 64, 64]
            scores = scores + smolgen_logits
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn_weights, v)
        
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.out_proj(out)
        
        return out


class ChessFFN(nn.Module):
    """
    Feed-forward network optimized for chess:
    - Uses Mish activation
    - Smaller expansion ratio (1x to 1.5x instead of 4x)
    """
    def __init__(self, hidden_size: int, ffn_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ffn_size)
        self.fc2 = nn.Linear(ffn_size, hidden_size)

        self.apply(init_xavier)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(nn.functional.mish(self.fc1(x)))


class ChessTransformerLayer(nn.Module):
    """
    Transformer layer with Post-LayerNorm architecture
    """
    def __init__(self, hidden_size: int, num_heads: int, ffn_size: int, head_dim: int = 32, use_smolgen: bool = True):
        super().__init__()
        self.attention = ChessMultiHeadAttention(hidden_size, num_heads, head_dim, use_smolgen)
        self.ffn = ChessFFN(hidden_size, ffn_size)
        self.norm1 = LayerNorm(hidden_size)
        self.norm2 = LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.norm1(self.attention(x, mask))
        x = x + self.norm2(self.ffn(x))
        return x


class ChessEmbedding(nn.Module):
    """
    Enhanced embedding layer that encodes full board state:
    1. Processes piece information for current and last 7 plies
    2. Includes castling, en passant, rule 50, repetition info
    3. Adds learnable positional embeddings
    4. Uses enhanced preprocessing layer
    """
    def __init__(self, hidden_size: int, enhanced_channels: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.board_preprocess = nn.Linear(12, enhanced_channels)  # 12 piece types to C channels
        
        input_size = (12 + 1 + 4 + 1)  # 12 piece types + en passant + 4 castling sides + 50 rule count
        self.piece_embedding = nn.Linear(input_size + enhanced_channels, hidden_size)
        
        self.pos_embedding = nn.Parameter(torch.randn(64, hidden_size))
        
        self.post_embedding_ffn = ChessFFN(hidden_size, hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, 64, input_features] where input_features includes:
               - A one-hot vectors of (12 planes, current position for each piece)
               - En passant info (1 plane, one hot of en passant square)
               - Castling info (4 planes, one for each castling side)
               - Rule 50 count (number)
        """
        batch_size, seq_len, _ = x.shape
        
        current_board = x[:, :, :12]  # [batch_size, 64, 12]
        enhanced_features = self.board_preprocess(current_board)  # [batch_size, 64, enhanced_channels]
        
        combined_input = torch.cat([x, enhanced_features], dim=-1)
        embedded = self.piece_embedding(combined_input)
        
        embedded = embedded + self.pos_embedding.unsqueeze(0)
        
        embedded = self.post_embedding_ffn(embedded)
        
        return embedded


class ChessPolicyHead(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.query = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.key = nn.Linear(hidden_size, num_heads * head_dim, bias=False)

        self.promo_head = nn.Linear(head_dim, 4, bias=False)

        self.scale = (head_dim) ** -0.5

        self.apply(init_xavier)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores.mean(dim=1)

        promotion_keys = k.mean(dim=1)
        promotion_logits = self.promo_head(promotion_keys)
        q_promo_logits = promotion_logits[:, :, 0]
        promotion_logits = promotion_logits[:, :, 1:]
        promotion_logits[:, :, 0] += q_promo_logits
        promotion_logits[:, :, 1] += q_promo_logits
        promotion_logits[:, :, 2] += q_promo_logits
        promotion_logits = promotion_logits * self.scale

        policy_logits = torch.cat([scores, promotion_logits], dim=-1)
        
        return policy_logits


class ChessValueHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.l1 = nn.Linear(hidden_size, 32)
        self.l2 = nn.Linear(64*32, 128)
        self.l3 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        x = self.l1(x)
        x = nn.functional.mish(x)
        x = x.view(batch_size, -1)
        x = self.l2(x)
        x = nn.functional.mish(x)
        x = self.l3(x)
        x = x.squeeze(-1)

        return x


class ChessTransformer(nn.Module):
    """
    Complete Lc0-like Chess Transformer architecture
    """
    def __init__(
        self,
        num_layers: int = 15,
        hidden_size: int = 768,
        num_heads: int = 24,
        ffn_size: int = 1024,
        head_dim: int = 32,
        use_smolgen: bool = True,
        enhanced_embedding: bool = True,
        num_value_outputs: int = 1
    ):
        super().__init__()
        
        input_features = 8 + 1 + 4 + 1
        
        if enhanced_embedding:
            self.embedding = ChessEmbedding(hidden_size)
        else:
            self.embedding = nn.Sequential(
                nn.Linear(input_features, hidden_size),
                nn.LayerNorm(hidden_size)
            )
        
        self.layers = nn.ModuleList([
            ChessTransformerLayer(hidden_size, num_heads, ffn_size, head_dim, use_smolgen)
            for _ in range(num_layers)
        ])

        self.policy_head = ChessPolicyHead(hidden_size, num_heads, head_dim)

        self.value_head = ChessValueHead(hidden_size)
    
    def embed(self, x: torch.Tensor) -> torch.Tensor:        
        x = self.embedding(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x_pooled = x.mean(dim=1)

        return x_pooled
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, 64, input_features] - board representation
        Returns:
            policy_logits: [batch_size, num_policy_outputs]
            value_logits: [batch_size, num_value_outputs]
        """
        batch_size = x.size(0)

        x = self.embedding(x)
        
        for layer in self.layers:
            x = layer(x)
        
        policy_logits = self.policy_head(x)
        value_logit = self.value_head(x)
        
        return policy_logits, value_logit


def create_chessformer_model(config: ChessFormerConfig):
    return ChessTransformer(
        num_layers=config.num_layers,
        hidden_size=config.hidden_size,
        num_heads=config.num_heads,
        ffn_size=config.ffn_size,
        head_dim=config.head_dim,
        use_smolgen=config.use_smolgen,
        enhanced_embedding=config.enhanced_embedding
    )


if __name__ == '__main__':
    model = ChessTransformer(
        num_layers = 8,
        hidden_size = 384,
        num_heads = 24,
        ffn_size = 512,
        head_dim = 32,
        use_smolgen = True,
        enhanced_embedding = True
    )
    model.eval()

    pos = torch.randn(4, 64, 18)

    with torch.inference_mode():
        k = model(pos)
