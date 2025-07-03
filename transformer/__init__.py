"""
Transformer架构实现

这个包实现了完整的Transformer架构，包括：
- 多头注意力机制
- 位置编码
- 前馈网络
- 层归一化
- 完整的Encoder和Decoder
"""

from .attention import (
    MultiHeadAttention, 
    scaled_dot_product_attention, 
    RoPEMultiHeadAttention, 
    RoPESelfAttention,
    GroupedQueryAttention,
    GQASelfAttention,
    RoPEGroupedQueryAttention,
    RoGQASelfAttention
)
from .positional_encoding import PositionalEncoding, get_positional_encoding, RoPE
from .transformer import (
    TransformerEncoder,
    TransformerDecoder,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    Transformer
)
from .utils import create_padding_mask, create_look_ahead_mask

__version__ = "1.0.0"
__author__ = "Transformer Implementation"

__all__ = [
    "MultiHeadAttention",
    "scaled_dot_product_attention",
    "RoPEMultiHeadAttention",
    "RoPESelfAttention",
    "GroupedQueryAttention",
    "GQASelfAttention",
    "RoPEGroupedQueryAttention",
    "RoGQASelfAttention",
    "PositionalEncoding", 
    "get_positional_encoding",
    "RoPE",
    "TransformerEncoder",
    "TransformerDecoder", 
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
    "Transformer",
    "create_padding_mask",
    "create_look_ahead_mask"
] 