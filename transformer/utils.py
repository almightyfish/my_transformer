"""
工具函数

包含Transformer模型中常用的工具函数：
- 掩码生成
- 模型初始化
- 其他辅助函数
"""

import torch
import torch.nn as nn
import math


def create_padding_mask(seq, pad_idx=0):
    """
    创建填充掩码，用于忽略填充标记
    
    Args:
        seq: 输入序列 [batch_size, seq_len]
        pad_idx: 填充标记的索引
        
    Returns:
        mask: 填充掩码 [batch_size, 1, 1, seq_len]
    """
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask.float()


def create_look_ahead_mask(size):
    """
    创建前瞻掩码，用于防止解码器看到未来的标记
    
    Args:
        size: 序列长度
        
    Returns:
        mask: 前瞻掩码 [size, size]
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0


def create_combined_mask(seq, pad_idx=0):
    """
    创建组合掩码（填充掩码 + 前瞻掩码），用于解码器
    
    Args:
        seq: 输入序列 [batch_size, seq_len]
        pad_idx: 填充标记的索引
        
    Returns:
        mask: 组合掩码 [batch_size, 1, seq_len, seq_len]
    """
    seq_len = seq.size(1)
    look_ahead_mask = create_look_ahead_mask(seq_len).to(seq.device)
    padding_mask = create_padding_mask(seq, pad_idx)
    
    # 广播并组合两个掩码
    combined_mask = torch.minimum(look_ahead_mask.unsqueeze(0).unsqueeze(1), padding_mask)
    return combined_mask


def get_clones(module, n):
    """
    创建模块的n个副本
    
    Args:
        module: 要克隆的模块
        n: 克隆数量
        
    Returns:
        cloned_modules: 克隆的模块列表
    """
    return nn.ModuleList([module for _ in range(n)])


def init_weights(module):
    """
    初始化模型权重
    
    Args:
        module: 要初始化的模块
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Embedding):
        nn.init.xavier_uniform_(module.weight)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)


def count_parameters(model):
    """
    计算模型的参数数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        total_params: 总参数数量
        trainable_params: 可训练参数数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_device():
    """
    获取可用的设备（GPU或CPU）
    
    Returns:
        device: 设备对象
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    保存模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前轮数
        loss: 当前损失
        filepath: 保存路径
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """
    加载模型检查点
    
    Args:
        filepath: 检查点文件路径
        model: 模型
        optimizer: 优化器（可选）
        
    Returns:
        epoch: 轮数
        loss: 损失
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss']


def create_sinusoidal_embeddings(n_pos, dim, out):
    """
    创建正弦位置嵌入
    
    Args:
        n_pos: 位置数量
        dim: 维度
        out: 输出张量
    """
    position_enc = torch.zeros(n_pos, dim)
    position = torch.arange(0, n_pos, dtype=torch.float).unsqueeze(1)
    
    div_term = torch.exp(torch.arange(0, dim, 2).float() * 
                        -(math.log(10000.0) / dim))
    
    position_enc[:, 0::2] = torch.sin(position * div_term)
    position_enc[:, 1::2] = torch.cos(position * div_term)
    
    out.detach_()
    out.requires_grad = False
    out[:, :] = position_enc


def attention_visualization(attention_weights, input_tokens, output_tokens=None):
    """
    可视化注意力权重的辅助函数
    
    Args:
        attention_weights: 注意力权重 [num_heads, seq_len, seq_len]
        input_tokens: 输入标记列表
        output_tokens: 输出标记列表（可选）
        
    Returns:
        visualization_data: 可视化数据字典
    """
    num_heads, seq_len_q, seq_len_k = attention_weights.shape
    
    # 平均所有头的注意力权重
    avg_attention = attention_weights.mean(dim=0)
    
    visualization_data = {
        'attention_weights': avg_attention.detach().cpu().numpy(),
        'input_tokens': input_tokens,
        'output_tokens': output_tokens if output_tokens else input_tokens,
        'num_heads': num_heads
    }
    
    return visualization_data


def calculate_bleu_score(predictions, targets):
    """
    计算BLEU分数的简单实现
    
    Args:
        predictions: 预测序列列表
        targets: 目标序列列表
        
    Returns:
        bleu_score: BLEU分数
    """
    # 这是一个简化的实现，实际使用中建议使用专门的库如nltk或sacrebleu
    from collections import Counter
    
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    total_score = 0
    for pred, target in zip(predictions, targets):
        score = 0
        for n in range(1, 5):  # 1-gram到4-gram
            pred_ngrams = Counter(get_ngrams(pred, n))
            target_ngrams = Counter(get_ngrams(target, n))
            
            overlap = sum((pred_ngrams & target_ngrams).values())
            total_pred = sum(pred_ngrams.values())
            
            if total_pred > 0:
                score += overlap / total_pred
        
        total_score += score / 4  # 平均1到4-gram的分数
    
    return total_score / len(predictions) 