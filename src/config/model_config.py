from dataclasses import dataclass


@dataclass
class ChessFormerConfig:
    num_layers: int
    hidden_size: int
    num_heads: int
    ffn_size: int
    head_dim: int
    use_smolgen: bool
    enhanced_embedding: bool
