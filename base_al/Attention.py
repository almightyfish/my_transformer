import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (MHA) 实现
    标准的多头注意力机制，每个头都有独立的 query, key, value 投影
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性投影层
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, query, key, value, mask=None):
        """计算缩放点积注意力"""
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, value)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)
        
        # 1. 线性投影并重塑为多头格式
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. 计算注意力
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 3. 重塑并连接所有头
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # 4. 最终线性投影
        output = self.w_o(attention_output)
        
        return output, attention_weights


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) 实现
    通过将 key 和 value 头分组来减少参数量和计算量
    """
    
    def __init__(self, d_model, num_query_heads, num_kv_heads, dropout=0.1):
        super(GroupedQueryAttention, self).__init__()
        assert d_model % num_query_heads == 0
        assert num_query_heads % num_kv_heads == 0
        
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_query_heads // num_kv_heads
        self.d_k = d_model // num_query_heads
        
        # 线性投影层
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.w_v = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, query, key, value, mask=None):
        """计算缩放点积注意力"""
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, value)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)
        
        # 1. 线性投影
        Q = self.w_q(query).view(batch_size, seq_len, self.num_query_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_kv_heads, self.d_k).transpose(1, 2)
        
        # 2. 重复 K 和 V 以匹配查询头的数量
        # 每个 KV 头对应 num_groups 个查询头
        K = K.repeat_interleave(self.num_groups, dim=1)  # [batch, num_query_heads, seq_len, d_k]
        V = V.repeat_interleave(self.num_groups, dim=1)  # [batch, num_query_heads, seq_len, d_k]
        
        # 3. 计算注意力
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 4. 重塑并连接所有头
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # 5. 最终线性投影
        output = self.w_o(attention_output)
        
        return output, attention_weights


def compare_attention_mechanisms():
    """比较 MHA 和 GQA 的参数量和计算复杂度"""
    d_model = 512
    num_heads = 8
    num_kv_heads = 2  # GQA 中的 KV 头数量
    
    # 创建模型
    mha = MultiHeadAttention(d_model, num_heads)
    gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads)
    
    # 计算参数量
    mha_params = sum(p.numel() for p in mha.parameters())
    gqa_params = sum(p.numel() for p in gqa.parameters())
    
    print(f"MHA 参数量: {mha_params:,}")
    print(f"GQA 参数量: {gqa_params:,}")
    print(f"参数减少比例: {(1 - gqa_params/mha_params)*100:.1f}%")
    
    # 测试前向传播
    batch_size, seq_len = 2, 100
    x = torch.randn(batch_size, seq_len, d_model)
    
    # MHA 前向传播
    mha_output, mha_weights = mha(x, x, x)
    print(f"\nMHA 输出形状: {mha_output.shape}")
    
    # GQA 前向传播
    gqa_output, gqa_weights = gqa(x, x, x)
    print(f"GQA 输出形状: {gqa_output.shape}")


if __name__ == "__main__":
    # 运行比较
    compare_attention_mechanisms()
    
    # 详细测试示例
    print("\n" + "="*50)
    print("详细测试示例")
    print("="*50)
    
    # 参数设置
    d_model = 256
    num_heads = 8
    num_kv_heads = 2
    batch_size = 1
    seq_len = 10
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"输入形状: {x.shape}")
    
    # 测试 MHA
    print(f"\n--- Multi-Head Attention ---")
    mha = MultiHeadAttention(d_model, num_heads)
    mha_out, mha_attn = mha(x, x, x)
    print(f"输出形状: {mha_out.shape}")
    print(f"注意力权重形状: {mha_attn.shape}")
    
    # 测试 GQA
    print(f"\n--- Grouped Query Attention ---")
    gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads)
    gqa_out, gqa_attn = gqa(x, x, x)
    print(f"输出形状: {gqa_out.shape}")
    print(f"注意力权重形状: {gqa_attn.shape}")
    
    # 比较计算效率
    print(f"\n--- 效率比较 ---")
    print(f"MHA Query头数: {num_heads}, Key/Value头数: {num_heads}")
    print(f"GQA Query头数: {num_heads}, Key/Value头数: {num_kv_heads}")
    print(f"GQA 每个KV头对应的Query头数: {num_heads // num_kv_heads}")