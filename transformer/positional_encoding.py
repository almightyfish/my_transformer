"""
位置编码实现

实现了Transformer中的位置编码：
- 正弦余弦位置编码
- 可学习的位置编码
"""

import torch
import torch.nn as nn
import math


def get_positional_encoding(seq_len, d_model):
    """
    生成正弦余弦位置编码
    
    Args:
        seq_len: 序列长度
        d_model: 模型维度
        
    Returns:
        pos_encoding: 位置编码 [seq_len, d_model]
    """
    pos_encoding = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    
    # 计算分母项
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                        (-math.log(10000.0) / d_model))
    
    # 应用sin到偶数索引
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    
    # 应用cos到奇数索引
    pos_encoding[:, 1::2] = torch.cos(position * div_term)
    
    return pos_encoding


class PositionalEncoding(nn.Module):
    """
    位置编码层
    
    Args:
        d_model: 模型维度
        max_seq_len: 最大序列长度
        dropout: dropout概率
    """
    
    def __init__(self, d_model, max_seq_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 预计算位置编码
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_seq_len, 1, d_model]
        
        # 注册为缓冲区，不会被视为模型参数
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 [seq_len, batch_size, d_model]
            
        Returns:
            output: 添加位置编码后的输出 [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    可学习的位置编码
    
    Args:
        d_model: 模型维度
        max_seq_len: 最大序列长度
        dropout: dropout概率
    """
    
    def __init__(self, d_model, max_seq_len=5000, dropout=0.1):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(torch.randn(max_seq_len, 1, d_model))
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 [seq_len, batch_size, d_model]
            
        Returns:
            output: 添加位置编码后的输出 [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class PositionalEmbedding(nn.Module):
    """
    基于嵌入的位置编码（类似词嵌入）
    
    Args:
        d_model: 模型维度
        max_seq_len: 最大序列长度
    """
    
    def __init__(self, d_model, max_seq_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            
        Returns:
            output: 位置嵌入 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.size()
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        return self.embedding(positions)


class RoPE(nn.Module):
    """
    旋转位置编码 (Rotary Position Embedding)
    
    RoPE通过旋转变换将位置信息编码到注意力机制中，
    相比传统位置编码具有更好的外推能力和相对位置建模能力。
    
    Args:
        d_model: 模型维度（必须是偶数）
        max_seq_len: 最大序列长度
        base: 角频率基数，默认为10000
    """
    
    def __init__(self, d_model, max_seq_len=2048, base=10000):
        super(RoPE, self).__init__()
        assert d_model % 2 == 0, "d_model必须是偶数才能使用RoPE"
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 预计算角频率
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # 预计算位置编码
        self._build_cache(max_seq_len)
    
    def _build_cache(self, max_seq_len):
        """预计算并缓存位置编码"""
        seq_len = max_seq_len
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        
        # 计算cos和sin值
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])
    
    def _rotate_half(self, x):
        """旋转向量的一半维度"""
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, q, k, seq_len=None):
        """
        应用RoPE位置编码
        
        Args:
            q: 查询张量 [batch_size, num_heads, seq_len, head_dim]
            k: 键张量 [batch_size, num_heads, seq_len, head_dim]
            seq_len: 序列长度（可选，默认使用q的序列长度）
            
        Returns:
            q_rot: 应用RoPE后的查询张量
            k_rot: 应用RoPE后的键张量
        """
        if seq_len is None:
            seq_len = q.shape[-2]
        
        # 如果序列长度超过缓存，重新构建
        if seq_len > self.cos_cached.shape[2]:
            self._build_cache(seq_len)
        
        # 获取对应长度的cos和sin
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        
        # 应用旋转变换
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        
        return q_rot, k_rot
    
    def apply_rotary_pos_emb(self, x, position_ids=None):
        """
        对单个张量应用旋转位置编码
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model] 或 [batch_size, num_heads, seq_len, head_dim]
            position_ids: 位置ID [batch_size, seq_len]（可选）
            
        Returns:
            x_rot: 应用RoPE后的张量
        """
        if position_ids is None:
            seq_len = x.shape[-2]
            cos = self.cos_cached[:, :, :seq_len, :]
            sin = self.sin_cached[:, :, :seq_len, :]
        else:
            cos = self.cos_cached[:, :, position_ids, :]
            sin = self.sin_cached[:, :, position_ids, :]
        
        return x * cos + self._rotate_half(x) * sin


class RelativePositionalEncoding(nn.Module):
    """
    传统相对位置编码（保留用于兼容性）
    
    Args:
        d_model: 模型维度
        max_relative_position: 最大相对位置
    """
    
    def __init__(self, d_model, max_relative_position=128):
        super(RelativePositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        
        vocab_size = max_relative_position * 2 + 1
        self.embeddings_table = nn.Parameter(torch.randn(vocab_size, d_model))
        
    def forward(self, length):
        """
        生成相对位置编码
        
        Args:
            length: 序列长度
            
        Returns:
            relative_positions: 相对位置编码 [length, length, d_model]
        """
        range_vec = torch.arange(length)
        range_mat = range_vec.unsqueeze(0).expand(length, length)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        
        distance_mat_clipped = torch.clamp(
            distance_mat, 
            -self.max_relative_position, 
            self.max_relative_position
        )
        
        final_mat = distance_mat_clipped + self.max_relative_position
        embeddings = self.embeddings_table[final_mat]
        
        return embeddings 