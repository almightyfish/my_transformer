"""
分组查询注意力 (Grouped Query Attention, GQA) 示例

GQA是一种高效的注意力机制，介于多头注意力(MHA)和多查询注意力(MQA)之间。
通过减少key-value头的数量来降低内存使用和计算成本。

主要优势：
1. 减少内存占用 - key和value使用更少的头数
2. 提高计算效率 - 减少KV-cache的大小
3. 保持性能 - 比MQA性能更好，比MHA效率更高
4. 灵活配置 - 可以根据需求调整KV头数

适用场景：
- 长序列处理
- 资源受限环境
- 推理优化
- 大型语言模型
"""

import torch
import torch.nn as nn
from .attention import (
    MultiHeadAttention, 
    GroupedQueryAttention, 
    GQASelfAttention,
    RoPEGroupedQueryAttention,
    RoGQASelfAttention
)


def compare_attention_mechanisms():
    """比较不同注意力机制的内存使用和计算效率"""
    
    print("=== 注意力机制比较 ===\n")
    
    # 配置参数
    batch_size = 2
    seq_len = 128
    d_model = 512
    num_heads = 8
    num_kv_heads = 2  # GQA使用更少的KV头
    
    # 创建测试数据
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 1. 多头注意力 (MHA)
    print("1. 多头注意力 (MHA)")
    mha = MultiHeadAttention(d_model, num_heads)
    mha_params = sum(p.numel() for p in mha.parameters())
    print(f"   参数量: {mha_params:,}")
    
    with torch.no_grad():
        mha_output, mha_weights = mha(x, x, x)
        print(f"   输出形状: {mha_output.shape}")
        print(f"   注意力权重形状: {mha_weights.shape}")
    
    # 2. 分组查询注意力 (GQA)
    print("\n2. 分组查询注意力 (GQA)")
    gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads)
    gqa_params = sum(p.numel() for p in gqa.parameters())
    print(f"   参数量: {gqa_params:,}")
    print(f"   参数减少: {((mha_params - gqa_params) / mha_params * 100):.1f}%")
    print(f"   查询头数: {num_heads}, KV头数: {num_kv_heads}")
    print(f"   每个KV头对应: {num_heads // num_kv_heads} 个查询头")
    
    with torch.no_grad():
        gqa_output, gqa_weights = gqa(x, x, x)
        print(f"   输出形状: {gqa_output.shape}")
        print(f"   注意力权重形状: {gqa_weights.shape}")
    
    # 3. GQA自注意力
    print("\n3. GQA自注意力")
    gqa_self = GQASelfAttention(d_model, num_heads, num_kv_heads)
    
    with torch.no_grad():
        gqa_self_output, _ = gqa_self(x)
        print(f"   输出形状: {gqa_self_output.shape}")
    
    print()


def demonstrate_rope_gqa():
    """演示支持RoPE的GQA"""
    
    print("=== RoPE + GQA 组合示例 ===\n")
    
    # 配置参数
    batch_size = 2
    seq_len = 64
    d_model = 256
    num_heads = 8
    num_kv_heads = 4
    max_seq_len = 512
    
    # 创建测试数据
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 1. RoPE + GQA
    print("1. RoPE分组查询注意力")
    rope_gqa = RoPEGroupedQueryAttention(
        d_model=d_model,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        max_seq_len=max_seq_len,
        rope_base=10000
    )
    
    with torch.no_grad():
        output, weights = rope_gqa(x, x, x)
        print(f"   输出形状: {output.shape}")
        print(f"   具有位置编码的注意力权重形状: {weights.shape}")
    
    # 2. RoPE + GQA 自注意力
    print("\n2. RoPE GQA自注意力")
    rope_gqa_self = RoGQASelfAttention(
        d_model=d_model,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        max_seq_len=max_seq_len
    )
    
    with torch.no_grad():
        self_output, _ = rope_gqa_self(x)
        print(f"   自注意力输出形状: {self_output.shape}")
    
    print()


def efficiency_analysis():
    """分析不同配置的效率"""
    
    print("=== GQA 效率分析 ===\n")
    
    d_model = 768
    seq_len = 256
    batch_size = 4
    
    configurations = [
        {"name": "标准MHA", "num_heads": 12, "num_kv_heads": 12},
        {"name": "GQA-4", "num_heads": 12, "num_kv_heads": 4},
        {"name": "GQA-2", "num_heads": 12, "num_kv_heads": 2},
        {"name": "MQA", "num_heads": 12, "num_kv_heads": 1},
    ]
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"{'配置':<10} {'参数量':<15} {'参数比例':<10} {'KV缓存比例':<12} {'描述'}")
    print("-" * 70)
    
    base_params = None
    
    for config in configurations:
        if config["num_kv_heads"] == config["num_heads"]:
            # 标准MHA
            model = MultiHeadAttention(d_model, config["num_heads"])
        else:
            # GQA或MQA
            model = GroupedQueryAttention(d_model, config["num_heads"], config["num_kv_heads"])
        
        params = sum(p.numel() for p in model.parameters())
        
        if base_params is None:
            base_params = params
            param_ratio = "100%"
        else:
            param_ratio = f"{params/base_params*100:.0f}%"
        
        # KV缓存大小比例（主要是K和V的参数）
        kv_ratio = f"{config['num_kv_heads']/12*100:.0f}%"
        
        # 描述
        if config["num_kv_heads"] == config["num_heads"]:
            desc = "标准多头注意力"
        elif config["num_kv_heads"] == 1:
            desc = "多查询注意力"
        else:
            desc = f"每{config['num_heads']//config['num_kv_heads']}个Q共享1个KV"
        
        print(f"{config['name']:<10} {params:<15,} {param_ratio:<10} {kv_ratio:<12} {desc}")
    
    print()


def practical_usage_example():
    """实际使用场景示例"""
    
    print("=== 实际使用场景 ===\n")
    
    # 场景1：长序列处理
    print("场景1: 长序列文档处理")
    print("- 序列长度: 2048")
    print("- 模型维度: 768") 
    print("- 使用GQA-4来平衡性能和效率")
    
    long_seq_model = GQASelfAttention(
        d_model=768,
        num_heads=12,
        num_kv_heads=4,  # 1/3的KV头数
        dropout=0.1
    )
    
    # 模拟长序列
    long_x = torch.randn(1, 2048, 768)
    with torch.no_grad():
        long_output, _ = long_seq_model(long_x)
        print(f"   处理结果形状: {long_output.shape}")
    
    print()
    
    # 场景2：移动设备部署
    print("场景2: 移动设备/边缘计算")
    print("- 模型维度: 384")
    print("- 使用GQA-1(MQA)以最大化效率")
    
    mobile_model = RoGQASelfAttention(
        d_model=384,
        num_heads=6,
        num_kv_heads=1,  # 极致的参数压缩
        max_seq_len=512
    )
    
    mobile_params = sum(p.numel() for p in mobile_model.parameters())
    print(f"   模型参数量: {mobile_params:,}")
    
    # 模拟移动设备输入
    mobile_x = torch.randn(1, 128, 384)
    with torch.no_grad():
        mobile_output, _ = mobile_model(mobile_x)
        print(f"   处理结果形状: {mobile_output.shape}")
    
    print()


def performance_optimization_tips():
    """性能优化建议"""
    
    print("=== GQA 性能优化建议 ===\n")
    
    tips = [
        "1. KV头数选择：",
        "   - 通常选择查询头数的1/2到1/4",
        "   - 例如：12个查询头 -> 2-4个KV头",
        "   - 保证num_heads能被num_kv_heads整除",
        "",
        "2. 内存优化：",
        "   - GQA-4相比MHA减少约33%的KV缓存",
        "   - MQA相比MHA减少约92%的KV缓存",
        "   - 适合长序列和批处理",
        "",
        "3. 计算效率：",
        "   - 训练时效果明显，推理时效果更佳",
        "   - 与RoPE结合使用效果更好",
        "   - 适合Transformer大模型",
        "",
        "4. 质量平衡：",
        "   - GQA-4: 95-98%的MHA性能",
        "   - GQA-2: 90-95%的MHA性能", 
        "   - MQA: 85-90%的MHA性能",
        "",
        "5. 使用场景：",
        "   - 推理服务：优先选择GQA或MQA",
        "   - 训练阶段：可以使用GQA加速",
        "   - 资源受限：使用MQA",
        "   - 高质量要求：使用GQA-4"
    ]
    
    for tip in tips:
        print(tip)
    
    print()


def main():
    """主函数：运行所有示例"""
    
    print("🚀 分组查询注意力 (GQA) 完整示例\n")
    print("=" * 50)
    
    # 设置随机种子以保证结果可重现
    torch.manual_seed(42)
    
    # 运行各个示例
    compare_attention_mechanisms()
    demonstrate_rope_gqa()
    efficiency_analysis()
    practical_usage_example()
    performance_optimization_tips()
    
    print("=" * 50)
    print("✅ 所有GQA示例运行完成！")
    print("\n💡 提示：")
    print("- GQA是现代Transformer的重要优化技术")
    print("- 建议在实际项目中根据需求选择合适的KV头数")
    print("- 可以与RoPE等其他技术组合使用以获得更好效果")


if __name__ == "__main__":
    main() 