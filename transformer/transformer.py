"""
Transformer架构实现

实现了完整的Transformer模型：
- TransformerEncoderLayer: 编码器层
- TransformerDecoderLayer: 解码器层
- TransformerEncoder: 编码器
- TransformerDecoder: 解码器
- Transformer: 完整模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from .attention import MultiHeadAttention
from .positional_encoding import PositionalEncoding
from .utils import get_clones


class FeedForward(nn.Module):
    """
    前馈神经网络
    
    Args:
        d_model: 模型维度
        d_ff: 前馈网络的隐藏层维度
        dropout: dropout概率
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入 [batch_size, seq_len, d_model]
            
        Returns:
            output: 输出 [batch_size, seq_len, d_model]
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器层
    
    Args:
        d_model: 模型维度
        num_heads: 注意力头数
        d_ff: 前馈网络的隐藏层维度
        dropout: dropout概率
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        前向传播
        
        Args:
            x: 输入 [batch_size, seq_len, d_model]
            mask: 注意力掩码 [batch_size, 1, seq_len, seq_len]
            
        Returns:
            output: 输出 [batch_size, seq_len, d_model]
        """
        # 自注意力 + 残差连接 + 层归一化
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class TransformerDecoderLayer(nn.Module):
    """
    Transformer解码器层
    
    Args:
        d_model: 模型维度
        num_heads: 注意力头数
        d_ff: 前馈网络的隐藏层维度
        dropout: dropout概率
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        """
        前向传播
        
        Args:
            x: 解码器输入 [batch_size, tgt_seq_len, d_model]
            encoder_output: 编码器输出 [batch_size, src_seq_len, d_model]
            tgt_mask: 目标序列掩码 [batch_size, 1, tgt_seq_len, tgt_seq_len]
            src_mask: 源序列掩码 [batch_size, 1, 1, src_seq_len]
            
        Returns:
            output: 输出 [batch_size, tgt_seq_len, d_model]
        """
        # 掩码自注意力 + 残差连接 + 层归一化
        self_attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_output))
        
        # 交叉注意力 + 残差连接 + 层归一化
        cross_attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer编码器
    
    Args:
        encoder_layer: 编码器层
        num_layers: 层数
    """
    
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        
    def forward(self, x, mask=None):
        """
        前向传播
        
        Args:
            x: 输入 [batch_size, seq_len, d_model]
            mask: 注意力掩码 [batch_size, 1, seq_len, seq_len]
            
        Returns:
            output: 输出 [batch_size, seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerDecoder(nn.Module):
    """
    Transformer解码器
    
    Args:
        decoder_layer: 解码器层
        num_layers: 层数
    """
    
    def __init__(self, decoder_layer, num_layers):
        super(TransformerDecoder, self).__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        
    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        """
        前向传播
        
        Args:
            x: 解码器输入 [batch_size, tgt_seq_len, d_model]
            encoder_output: 编码器输出 [batch_size, src_seq_len, d_model]
            tgt_mask: 目标序列掩码 [batch_size, 1, tgt_seq_len, tgt_seq_len]
            src_mask: 源序列掩码 [batch_size, 1, 1, src_seq_len]
            
        Returns:
            output: 输出 [batch_size, tgt_seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
        return x


class Transformer(nn.Module):
    """
    完整的Transformer模型
    
    Args:
        src_vocab_size: 源词汇表大小
        tgt_vocab_size: 目标词汇表大小
        d_model: 模型维度
        num_heads: 注意力头数
        num_encoder_layers: 编码器层数
        num_decoder_layers: 解码器层数
        d_ff: 前馈网络的隐藏层维度
        max_seq_len: 最大序列长度
        dropout: dropout概率
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, 
                 max_seq_len=5000, dropout=0.1):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        
        # 嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # 编码器
        encoder_layer = TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # 解码器
        decoder_layer = TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # 输出投影层
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # 初始化权重
        self.init_weights()
        
    def init_weights(self):
        """初始化模型权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(self, src, src_mask=None):
        """
        编码源序列
        
        Args:
            src: 源序列 [batch_size, src_seq_len]
            src_mask: 源序列掩码 [batch_size, 1, src_seq_len, src_seq_len]
            
        Returns:
            encoder_output: 编码器输出 [batch_size, src_seq_len, d_model]
        """
        # 嵌入 + 位置编码
        src_emb = self.src_embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        src_emb = src_emb.transpose(0, 1)  # [src_seq_len, batch_size, d_model]
        src_emb = self.pos_encoding(src_emb)
        src_emb = src_emb.transpose(0, 1)  # [batch_size, src_seq_len, d_model]
        
        # 编码
        encoder_output = self.encoder(src_emb, src_mask)
        return encoder_output
    
    def decode(self, tgt, encoder_output, tgt_mask=None, src_mask=None):
        """
        解码目标序列
        
        Args:
            tgt: 目标序列 [batch_size, tgt_seq_len]
            encoder_output: 编码器输出 [batch_size, src_seq_len, d_model]
            tgt_mask: 目标序列掩码 [batch_size, 1, tgt_seq_len, tgt_seq_len]
            src_mask: 源序列掩码 [batch_size, 1, 1, src_seq_len]
            
        Returns:
            decoder_output: 解码器输出 [batch_size, tgt_seq_len, d_model]
        """
        # 嵌入 + 位置编码
        tgt_emb = self.tgt_embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        tgt_emb = tgt_emb.transpose(0, 1)  # [tgt_seq_len, batch_size, d_model]
        tgt_emb = self.pos_encoding(tgt_emb)
        tgt_emb = tgt_emb.transpose(0, 1)  # [batch_size, tgt_seq_len, d_model]
        
        # 解码
        decoder_output = self.decoder(tgt_emb, encoder_output, tgt_mask, src_mask)
        return decoder_output
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        前向传播
        
        Args:
            src: 源序列 [batch_size, src_seq_len]
            tgt: 目标序列 [batch_size, tgt_seq_len]
            src_mask: 源序列掩码 [batch_size, 1, src_seq_len, src_seq_len]
            tgt_mask: 目标序列掩码 [batch_size, 1, tgt_seq_len, tgt_seq_len]
            
        Returns:
            output: 输出 [batch_size, tgt_seq_len, tgt_vocab_size]
        """
        # 编码
        encoder_output = self.encode(src, src_mask)
        
        # 解码
        decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_mask)
        
        # 输出投影
        output = self.output_projection(decoder_output)
        
        return output
    
    def generate(self, src, src_mask=None, max_length=100, start_token=1, end_token=2):
        """
        生成序列（贪心解码）
        
        Args:
            src: 源序列 [batch_size, src_seq_len]
            src_mask: 源序列掩码 [batch_size, 1, src_seq_len, src_seq_len]
            max_length: 最大生成长度
            start_token: 开始标记
            end_token: 结束标记
            
        Returns:
            generated: 生成的序列 [batch_size, gen_seq_len]
        """
        batch_size = src.size(0)
        device = src.device
        
        # 编码源序列
        encoder_output = self.encode(src, src_mask)
        
        # 初始化目标序列
        tgt = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
        
        for _ in range(max_length):
            # 创建目标掩码
            tgt_size = tgt.size(1)
            tgt_mask = torch.triu(torch.ones(tgt_size, tgt_size, device=device), diagonal=1)
            tgt_mask = (tgt_mask == 0).unsqueeze(0).unsqueeze(1)
            
            # 解码
            decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_mask)
            
            # 获取下一个标记
            next_token_logits = self.output_projection(decoder_output[:, -1, :])
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # 添加到目标序列
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # 检查是否所有序列都结束
            if (next_token == end_token).all():
                break
        
        return tgt 