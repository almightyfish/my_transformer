#!/usr/bin/env python3
"""
RoPE (Rotary Position Embedding) 使用示例

这个文件展示了如何使用RoPE位置编码：
1. 基本的RoPE使用
2. 支持RoPE的多头注意力
3. RoPE vs 传统位置编码的比较
4. 在Transformer中集成RoPE
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 如果在包内运行，使用相对导入
try:
    from .positional_encoding import RoPE, PositionalEncoding
    from .attention import RoPEMultiHeadAttention, MultiHeadAttention
    from .utils import create_padding_mask
except ImportError:
    # 如果直接运行该文件，使用绝对导入
    from transformer.positional_encoding import RoPE, PositionalEncoding
    from transformer.attention import RoPEMultiHeadAttention, MultiHeadAttention
    from transformer.utils import create_padding_mask


def basic_rope_example():
    """基本RoPE使用示例"""
    print("=" * 50)
    print("基本RoPE使用示例")
    print("=" * 50)
    
    # 设置参数
    d_model = 128
    seq_len = 10
    batch_size = 2
    num_heads = 8
    head_dim = d_model // num_heads
    
    # 创建RoPE实例
    rope = RoPE(head_dim, max_seq_len=100)
    print(f"RoPE创建完成, head_dim={head_dim}")
    
    # 创建示例张量
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    print(f"原始查询张量形状: {q.shape}")
    print(f"原始键张量形状: {k.shape}")
    
    # 应用RoPE
    q_rope, k_rope = rope(q, k)
    
    print(f"应用RoPE后查询张量形状: {q_rope.shape}")
    print(f"应用RoPE后键张量形状: {k_rope.shape}")
    
    # 验证RoPE的旋转不变性
    # 对于相对位置为0的情况，内积应该保持不变
    original_self_attn = torch.sum(q * k, dim=-1)  # [batch, heads, seq_len]
    rope_self_attn = torch.sum(q_rope * k_rope, dim=-1)
    
    print(f"原始自注意力分数 (前3个位置): {original_self_attn[0, 0, :3]}")
    print(f"RoPE自注意力分数 (前3个位置): {rope_self_attn[0, 0, :3]}")
    print(f"差异: {torch.mean(torch.abs(original_self_attn - rope_self_attn)):.6f}")


def rope_attention_comparison():
    """RoPE注意力 vs 传统注意力比较"""
    print("\n" + "=" * 50)
    print("RoPE注意力 vs 传统注意力比较")
    print("=" * 50)
    
    # 参数设置
    d_model = 256
    num_heads = 8
    seq_len = 32
    batch_size = 4
    
    # 创建输入数据
    x = torch.randn(batch_size, seq_len, d_model)
    mask = create_padding_mask(torch.ones(batch_size, seq_len), pad_idx=0)
    
    # 传统多头注意力
    traditional_attn = MultiHeadAttention(d_model, num_heads)
    rope_attn = RoPEMultiHeadAttention(d_model, num_heads)
    
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    with torch.no_grad():
        traditional_output, traditional_weights = traditional_attn(x, x, x, mask)
        rope_output, rope_weights = rope_attn(x, x, x, mask)
    
    print(f"传统注意力输出形状: {traditional_output.shape}")
    print(f"RoPE注意力输出形状: {rope_output.shape}")
    
    # 比较注意力权重分布
    print(f"传统注意力权重平均值: {traditional_weights.mean():.6f}")
    print(f"RoPE注意力权重平均值: {rope_weights.mean():.6f}")
    
    # 比较输出的差异
    output_diff = torch.mean(torch.abs(traditional_output - rope_output))
    print(f"输出差异: {output_diff:.6f}")
    
    # 分析注意力模式
    print("\n注意力模式分析:")
    print(f"传统注意力权重方差: {traditional_weights.var():.6f}")
    print(f"RoPE注意力权重方差: {rope_weights.var():.6f}")


def rope_extrapolation_test():
    """RoPE外推能力测试"""
    print("\n" + "=" * 50)
    print("RoPE外推能力测试")
    print("=" * 50)
    
    d_model = 64
    max_train_len = 50
    test_len = 100  # 超出训练长度
    
    # 创建RoPE实例
    rope = RoPE(d_model, max_seq_len=max_train_len)
    
    print(f"训练最大长度: {max_train_len}")
    print(f"测试长度: {test_len}")
    
    # 创建测试数据
    q_test = torch.randn(1, 1, test_len, d_model)
    k_test = torch.randn(1, 1, test_len, d_model)
    
    # 测试外推能力
    try:
        q_rope, k_rope = rope(q_test, k_test, seq_len=test_len)
        print("✅ RoPE成功处理超出训练长度的序列")
        print(f"外推后张量形状: {q_rope.shape}")
        
        # 验证位置编码的连续性
        # 计算相邻位置的相似度
        similarities = []
        for i in range(test_len - 1):
            sim = torch.cosine_similarity(q_rope[0, 0, i], q_rope[0, 0, i+1], dim=0)
            similarities.append(sim.item())
        
        similarities = np.array(similarities)
        print(f"相邻位置平均相似度: {similarities.mean():.4f}")
        print(f"相似度标准差: {similarities.std():.4f}")
        
    except Exception as e:
        print(f"❌ RoPE外推失败: {e}")


def visualize_rope_patterns():
    """可视化RoPE的位置编码模式"""
    print("\n" + "=" * 50)
    print("RoPE位置编码模式可视化")
    print("=" * 50)
    
    d_model = 64
    seq_len = 50
    
    # 创建RoPE
    rope = RoPE(d_model, max_seq_len=seq_len)
    
    # 获取不同位置的编码
    positions = torch.arange(seq_len).unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1, 1, seq_len, 1]
    dummy_tensor = torch.ones(1, 1, seq_len, d_model)
    
    # 应用位置编码
    encoded = rope.apply_rotary_pos_emb(dummy_tensor)
    
    # 计算位置间的相似度矩阵
    encoded_2d = encoded.squeeze().squeeze()  # [seq_len, d_model]
    similarity_matrix = torch.mm(encoded_2d, encoded_2d.t())
    
    print(f"位置编码矩阵形状: {encoded_2d.shape}")
    print(f"位置相似度矩阵形状: {similarity_matrix.shape}")
    
    # 分析对角线模式（相对位置模式）
    diagonals = []
    for offset in range(-10, 11):
        if offset == 0:
            continue
        diagonal = torch.diagonal(similarity_matrix, offset=offset)
        diagonals.append(diagonal.mean().item())
    
    print(f"不同相对位置的平均相似度范围: {min(diagonals):.4f} - {max(diagonals):.4f}")
    
    # 尝试保存可视化（如果matplotlib可用）
    try:
        plt.figure(figsize=(10, 8))
        plt.imshow(similarity_matrix.detach().numpy(), cmap='viridis')
        plt.title('RoPE位置编码相似度矩阵')
        plt.xlabel('位置')
        plt.ylabel('位置')
        plt.colorbar()
        plt.savefig('rope_similarity_matrix.png', dpi=150, bbox_inches='tight')
        print("✅ 相似度矩阵已保存为 rope_similarity_matrix.png")
    except Exception as e:
        print(f"⚠️  无法保存可视化图片: {e}")


def rope_vs_sinusoidal_comparison():
    """RoPE vs 正弦余弦位置编码比较"""
    print("\n" + "=" * 50)
    print("RoPE vs 正弦余弦位置编码比较")
    print("=" * 50)
    
    d_model = 128
    seq_len = 64
    batch_size = 2
    
    # 创建测试数据
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 创建位置编码实例
    sinusoidal_pe = PositionalEncoding(d_model, max_seq_len=seq_len)
    rope = RoPE(d_model, max_seq_len=seq_len)
    
    print(f"输入形状: {x.shape}")
    
    # 应用正弦余弦位置编码
    x_sin = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
    x_sin_encoded = sinusoidal_pe(x_sin)
    x_sin_encoded = x_sin_encoded.transpose(0, 1)  # 转回 [batch_size, seq_len, d_model]
    
    # 应用RoPE（注意RoPE通常在注意力计算中应用）
    # 这里我们直接应用到输入上作为演示
    x_rope_encoded = rope.apply_rotary_pos_emb(x)
    
    print(f"正弦余弦编码后形状: {x_sin_encoded.shape}")
    print(f"RoPE编码后形状: {x_rope_encoded.shape}")
    
    # 比较编码效果
    # 计算每种编码方式对原始信号的保持程度
    sin_preservation = torch.cosine_similarity(x.flatten(), x_sin_encoded.flatten(), dim=0)
    rope_preservation = torch.cosine_similarity(x.flatten(), x_rope_encoded.flatten(), dim=0)
    
    print(f"正弦余弦编码原始信号保持度: {sin_preservation:.4f}")
    print(f"RoPE编码原始信号保持度: {rope_preservation:.4f}")
    
    # 分析位置敏感性
    # 交换两个位置，看编码的变化
    x_swapped = x.clone()
    x_swapped[:, [0, seq_len//2]] = x_swapped[:, [seq_len//2, 0]]
    
    x_sin_swapped = x_swapped.transpose(0, 1)
    x_sin_swapped = sinusoidal_pe(x_sin_swapped).transpose(0, 1)
    x_rope_swapped = rope.apply_rotary_pos_emb(x_swapped)
    
    sin_sensitivity = torch.norm(x_sin_encoded - x_sin_swapped) / torch.norm(x_sin_encoded)
    rope_sensitivity = torch.norm(x_rope_encoded - x_rope_swapped) / torch.norm(x_rope_encoded)
    
    print(f"正弦余弦编码位置敏感性: {sin_sensitivity:.4f}")
    print(f"RoPE编码位置敏感性: {rope_sensitivity:.4f}")


def performance_comparison():
    """性能比较"""
    print("\n" + "=" * 50)
    print("性能比较")
    print("=" * 50)
    
    import time
    
    d_model = 512
    num_heads = 8
    seq_len = 128
    batch_size = 32
    num_iterations = 100
    
    # 创建测试数据
    x = torch.randn(batch_size, seq_len, d_model)
    mask = create_padding_mask(torch.ones(batch_size, seq_len), pad_idx=0)
    
    # 创建模型
    traditional_attn = MultiHeadAttention(d_model, num_heads)
    rope_attn = RoPEMultiHeadAttention(d_model, num_heads)
    
    # 预热
    for _ in range(10):
        with torch.no_grad():
            _ = traditional_attn(x, x, x, mask)
            _ = rope_attn(x, x, x, mask)
    
    # 测试传统注意力
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = traditional_attn(x, x, x, mask)
    traditional_time = time.time() - start_time
    
    # 测试RoPE注意力
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = rope_attn(x, x, x, mask)
    rope_time = time.time() - start_time
    
    print(f"传统注意力时间: {traditional_time:.4f}s ({num_iterations} 次迭代)")
    print(f"RoPE注意力时间: {rope_time:.4f}s ({num_iterations} 次迭代)")
    print(f"性能比率 (RoPE/传统): {rope_time/traditional_time:.2f}x")
    
    # 计算参数量
    traditional_params = sum(p.numel() for p in traditional_attn.parameters())
    rope_params = sum(p.numel() for p in rope_attn.parameters())
    
    print(f"传统注意力参数量: {traditional_params:,}")
    print(f"RoPE注意力参数量: {rope_params:,}")
    print(f"参数量差异: {rope_params - traditional_params:,}")


def main():
    """主函数"""
    print("RoPE (Rotary Position Embedding) 演示")
    print("展示RoPE的优势和使用方法")
    
    # 1. 基本使用
    basic_rope_example()
    
    # 2. 注意力比较
    rope_attention_comparison()
    
    # 3. 外推能力测试
    rope_extrapolation_test()
    
    # 4. 位置编码比较
    rope_vs_sinusoidal_comparison()
    
    # 5. 可视化模式
    visualize_rope_patterns()
    
    # 6. 性能比较
    performance_comparison()
    
    print("\n" + "=" * 50)
    print("RoPE演示完成!")
    print("=" * 50)
    
    print("\n📝 总结:")
    print("✅ RoPE的优势:")
    print("  - 更好的外推能力（可以处理比训练时更长的序列）")
    print("  - 直接编码相对位置信息")
    print("  - 不需要额外的位置嵌入参数")
    print("  - 在长序列上表现更好")
    print("\n🔧 使用建议:")
    print("  - 适合需要处理可变长度序列的任务")
    print("  - 在生成任务中表现优异")
    print("  - 可以与传统位置编码混合使用")


if __name__ == "__main__":
    # 设置随机种子以获得可重现的结果
    torch.manual_seed(42)
    np.random.seed(42)
    
    main() 