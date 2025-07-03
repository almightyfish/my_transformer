"""
注意力机制实现

实现了Transformer中的核心注意力机制：
- Scaled Dot-Product Attention
- Multi-Head Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """
    计算缩放点积注意力
    
    Args:
        query: 查询矩阵 [batch_size, num_heads, seq_len, d_k]
        key: 键矩阵 [batch_size, num_heads, seq_len, d_k]
        value: 值矩阵 [batch_size, num_heads, seq_len, d_v]
        mask: 注意力掩码 [batch_size, 1, seq_len, seq_len] 或 [batch_size, num_heads, seq_len, seq_len]
        dropout: dropout层
        
    Returns:
        output: 注意力输出 [batch_size, num_heads, seq_len, d_v]
        attention_weights: 注意力权重 [batch_size, num_heads, seq_len, seq_len]
    """
    d_k = query.size(-1)
    
    # 计算注意力分数
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 应用掩码
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 计算注意力权重
    attention_weights = F.softmax(scores, dim=-1)
    
    # 应用dropout
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    
    # 计算加权输出
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    
    Args:
        d_model: 模型维度
        num_heads: 注意力头数
        dropout: dropout概率
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性投影层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        前向传播
        
        Args:
            query: 查询 [batch_size, seq_len, d_model]
            key: 键 [batch_size, seq_len, d_model]
            value: 值 [batch_size, seq_len, d_model]
            mask: 掩码 [batch_size, 1, seq_len, seq_len]
            
        Returns:
            output: 输出 [batch_size, seq_len, d_model]
            attention_weights: 注意力权重 [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size = query.size(0)
        
        # 线性投影并重塑为多头形式
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        attention_output, attention_weights = scaled_dot_product_attention(
            Q, K, V, mask, self.dropout
        )
        
        # 合并多头
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # 最终的线性投影
        output = self.w_o(attention_output)
        
        return output, attention_weights


class SelfAttention(MultiHeadAttention):
    """
    自注意力机制（特殊的多头注意力，其中query、key、value相同）
    """
    
    def forward(self, x, mask=None):
        """
        前向传播
        
        Args:
            x: 输入 [batch_size, seq_len, d_model]
            mask: 掩码 [batch_size, 1, seq_len, seq_len]
            
        Returns:
            output: 输出 [batch_size, seq_len, d_model]
            attention_weights: 注意力权重 [batch_size, num_heads, seq_len, seq_len]
        """
        return super().forward(x, x, x, mask)


class CrossAttention(MultiHeadAttention):
    """
    交叉注意力机制（用于Decoder中，query来自decoder，key和value来自encoder）
    """
    
    def forward(self, decoder_input, encoder_output, mask=None):
        """
        前向传播
        
        Args:
            decoder_input: 解码器输入 [batch_size, seq_len_dec, d_model]
            encoder_output: 编码器输出 [batch_size, seq_len_enc, d_model]
            mask: 掩码 [batch_size, 1, seq_len_dec, seq_len_enc]
            
        Returns:
            output: 输出 [batch_size, seq_len_dec, d_model]
            attention_weights: 注意力权重 [batch_size, num_heads, seq_len_dec, seq_len_enc]
        """
        return super().forward(decoder_input, encoder_output, encoder_output, mask)


class RoPEMultiHeadAttention(nn.Module):
    """
    支持RoPE的多头注意力机制
    
    Args:
        d_model: 模型维度
        num_heads: 注意力头数
        dropout: dropout概率
        max_seq_len: 最大序列长度
        rope_base: RoPE的角频率基数
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1, max_seq_len=2048, rope_base=10000):
        super(RoPEMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性投影层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # RoPE位置编码
        from .positional_encoding import RoPE
        self.rope = RoPE(self.d_k, max_seq_len, rope_base)
        
    def forward(self, query, key, value, mask=None):
        """
        前向传播
        
        Args:
            query: 查询 [batch_size, seq_len, d_model]
            key: 键 [batch_size, seq_len, d_model]
            value: 值 [batch_size, seq_len, d_model]
            mask: 掩码 [batch_size, 1, seq_len, seq_len]
            
        Returns:
            output: 输出 [batch_size, seq_len, d_model]
            attention_weights: 注意力权重 [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size = query.size(0)
        
        # 线性投影并重塑为多头形式
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 应用RoPE位置编码
        Q, K = self.rope(Q, K)
        
        # 计算注意力
        attention_output, attention_weights = scaled_dot_product_attention(
            Q, K, V, mask, self.dropout
        )
        
        # 合并多头
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # 最终的线性投影
        output = self.w_o(attention_output)
        
        return output, attention_weights


class RoPESelfAttention(RoPEMultiHeadAttention):
    """
    支持RoPE的自注意力机制
    """
    
    def forward(self, x, mask=None):
        """
        前向传播
        
        Args:
            x: 输入 [batch_size, seq_len, d_model]
            mask: 掩码 [batch_size, 1, seq_len, seq_len]
            
        Returns:
            output: 输出 [batch_size, seq_len, d_model]
            attention_weights: 注意力权重 [batch_size, num_heads, seq_len, seq_len]
        """
        return super().forward(x, x, x, mask)


class GroupedQueryAttention(nn.Module):
    """
    分组查询注意力 (Grouped Query Attention, GQA)
    
    GQA是介于多头注意力(MHA)和多查询注意力(MQA)之间的方法。
    它使用较少的key-value头，每个KV头被多个query头共享，
    从而减少内存使用和计算成本，同时保持较好的性能。
    
    Args:
        d_model: 模型维度
        num_heads: 查询头数
        num_kv_heads: 键值头数（通常是num_heads的因子）
        dropout: dropout概率
    """
    
    def __init__(self, d_model, num_heads, num_kv_heads, dropout=0.1):
        super(GroupedQueryAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        assert d_model % num_kv_heads == 0, "d_model必须能被num_kv_heads整除"
        assert num_heads % num_kv_heads == 0, "num_heads必须能被num_kv_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.d_k = d_model // num_heads
        self.d_kv = d_model // num_kv_heads
        
        # 线性投影层
        self.w_q = nn.Linear(d_model, num_heads * self.d_k)  # Query保持全头数
        self.w_k = nn.Linear(d_model, num_kv_heads * self.d_k)  # Key使用较少头数
        self.w_v = nn.Linear(d_model, num_kv_heads * self.d_k)  # Value使用较少头数
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def _repeat_kv(self, x):
        """
        重复key-value张量以匹配query头数
        
        Args:
            x: 输入张量 [batch_size, num_kv_heads, seq_len, d_k]
            
        Returns:
            重复后的张量 [batch_size, num_heads, seq_len, d_k]
        """
        batch_size, num_kv_heads, seq_len, d_k = x.shape
        if self.num_queries_per_kv == 1:
            return x
        
        # 扩展维度并重复
        x = x.unsqueeze(2)  # [batch, num_kv_heads, 1, seq_len, d_k]
        x = x.expand(-1, -1, self.num_queries_per_kv, -1, -1)  # [batch, num_kv_heads, num_queries_per_kv, seq_len, d_k]
        x = x.reshape(batch_size, num_kv_heads * self.num_queries_per_kv, seq_len, d_k)  # [batch, num_heads, seq_len, d_k]
        
        return x
        
    def forward(self, query, key, value, mask=None):
        """
        前向传播
        
        Args:
            query: 查询 [batch_size, seq_len, d_model]
            key: 键 [batch_size, seq_len, d_model]
            value: 值 [batch_size, seq_len, d_model]
            mask: 掩码 [batch_size, 1, seq_len, seq_len]
            
        Returns:
            output: 输出 [batch_size, seq_len, d_model]
            attention_weights: 注意力权重 [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # 线性投影
        Q = self.w_q(query)  # [batch_size, seq_len, num_heads * d_k]
        K = self.w_k(key)    # [batch_size, seq_len, num_kv_heads * d_k]
        V = self.w_v(value)  # [batch_size, seq_len, num_kv_heads * d_k]
        
        # 重塑为多头形式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)
        
        # 重复K和V以匹配Q的头数
        K = self._repeat_kv(K)  # [batch_size, num_heads, seq_len, d_k]
        V = self._repeat_kv(V)  # [batch_size, num_heads, seq_len, d_k]
        
        # 计算注意力
        attention_output, attention_weights = scaled_dot_product_attention(
            Q, K, V, mask, self.dropout
        )
        
        # 合并多头
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # 最终的线性投影
        output = self.w_o(attention_output)
        
        return output, attention_weights


class GQASelfAttention(GroupedQueryAttention):
    """
    分组查询自注意力机制
    """
    
    def forward(self, x, mask=None):
        """
        前向传播
        
        Args:
            x: 输入 [batch_size, seq_len, d_model]
            mask: 掩码 [batch_size, 1, seq_len, seq_len]
            
        Returns:
            output: 输出 [batch_size, seq_len, d_model]
            attention_weights: 注意力权重 [batch_size, num_heads, seq_len, seq_len]
        """
        return super().forward(x, x, x, mask)


class RoPEGroupedQueryAttention(nn.Module):
    """
    支持RoPE的分组查询注意力机制
    
    结合了RoPE位置编码和GQA的优势：
    - RoPE提供更好的位置建模能力
    - GQA提供更高的计算效率
    
    Args:
        d_model: 模型维度
        num_heads: 查询头数
        num_kv_heads: 键值头数
        dropout: dropout概率
        max_seq_len: 最大序列长度
        rope_base: RoPE的角频率基数
    """
    
    def __init__(self, d_model, num_heads, num_kv_heads, dropout=0.1, max_seq_len=2048, rope_base=10000):
        super(RoPEGroupedQueryAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        assert d_model % num_kv_heads == 0, "d_model必须能被num_kv_heads整除"
        assert num_heads % num_kv_heads == 0, "num_heads必须能被num_kv_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.d_k = d_model // num_heads
        
        # 线性投影层
        self.w_q = nn.Linear(d_model, num_heads * self.d_k)
        self.w_k = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.w_v = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # RoPE位置编码
        from .positional_encoding import RoPE
        self.rope = RoPE(self.d_k, max_seq_len, rope_base)
        
    def _repeat_kv(self, x):
        """重复key-value张量以匹配query头数"""
        batch_size, num_kv_heads, seq_len, d_k = x.shape
        if self.num_queries_per_kv == 1:
            return x
        
        x = x.unsqueeze(2)
        x = x.expand(-1, -1, self.num_queries_per_kv, -1, -1)
        x = x.reshape(batch_size, num_kv_heads * self.num_queries_per_kv, seq_len, d_k)
        
        return x
        
    def forward(self, query, key, value, mask=None):
        """
        前向传播
        
        Args:
            query: 查询 [batch_size, seq_len, d_model]
            key: 键 [batch_size, seq_len, d_model]
            value: 值 [batch_size, seq_len, d_model]
            mask: 掩码 [batch_size, 1, seq_len, seq_len]
            
        Returns:
            output: 输出 [batch_size, seq_len, d_model]
            attention_weights: 注意力权重 [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # 线性投影
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        
        # 重塑为多头形式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)
        
        # 重复K和V以匹配Q的头数（为了应用RoPE）
        K_expanded = self._repeat_kv(K)
        
        # 应用RoPE位置编码（只对Q和K）
        Q, K_rope = self.rope(Q, K_expanded)
        
        # 计算注意力
        V_expanded = self._repeat_kv(V)
        attention_output, attention_weights = scaled_dot_product_attention(
            Q, K_rope, V_expanded, mask, self.dropout
        )
        
        # 合并多头
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # 最终的线性投影
        output = self.w_o(attention_output)
        
        return output, attention_weights


class RoGQASelfAttention(RoPEGroupedQueryAttention):
    """
    支持RoPE的分组查询自注意力机制
    """
    
    def forward(self, x, mask=None):
        """
        前向传播
        
        Args:
            x: 输入 [batch_size, seq_len, d_model]
            mask: 掩码 [batch_size, 1, seq_len, seq_len]
            
        Returns:
            output: 输出 [batch_size, seq_len, d_model]
            attention_weights: 注意力权重 [batch_size, num_heads, seq_len, seq_len]
        """
        return super().forward(x, x, x, mask) 