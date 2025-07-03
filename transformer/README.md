# Transformer架构实现

这是一个完整的Transformer架构实现，基于论文《Attention Is All You Need》。

## 功能特性

- ✅ 完整的Transformer模型（编码器-解码器架构）
- ✅ 多头注意力机制（Multi-Head Attention）
- ✅ **分组查询注意力（Grouped Query Attention, GQA）- 新增！**
  - 减少内存使用和计算成本
  - 介于MHA和MQA之间的高效选择
  - 支持与RoPE组合使用
- ✅ 位置编码（Positional Encoding）
  - 正弦余弦位置编码
  - 可学习位置编码
  - **RoPE（旋转位置编码）- 新增！**
- ✅ 前馈神经网络（Feed Forward Networks）
- ✅ 层归一化（Layer Normalization）
- ✅ 残差连接（Residual Connections）
- ✅ 掩码机制（Masking）
- ✅ 自注意力和交叉注意力
- ✅ **支持RoPE的注意力机制 - 新增！**
- ✅ **高效的RoPE+GQA组合 - 新增！**
- ✅ 可配置的模型参数
- ✅ 序列生成功能

## 文件结构

```
transformer/
├── __init__.py           # 包初始化文件
├── attention.py          # 注意力机制实现（包含GQA）
├── positional_encoding.py # 位置编码实现（包含RoPE）
├── transformer.py        # 主要的Transformer模型
├── utils.py             # 工具函数
├── example.py           # 使用示例
├── rope_example.py      # RoPE使用示例
├── gqa_example.py       # GQA使用示例
└── README.md            # 本文档
```

## 安装依赖

```bash
pip install torch torchvision
```

## 快速开始

### 1. 基本使用

```python
import torch
from transformer import Transformer

# 创建模型
model = Transformer(
    src_vocab_size=10000,    # 源词汇表大小
    tgt_vocab_size=10000,    # 目标词汇表大小
    d_model=512,             # 模型维度
    num_heads=8,             # 注意力头数
    num_encoder_layers=6,    # 编码器层数
    num_decoder_layers=6,    # 解码器层数
    d_ff=2048,              # 前馈网络维度
    max_seq_len=5000,       # 最大序列长度
    dropout=0.1             # dropout概率
)

# 准备输入数据
batch_size, src_seq_len, tgt_seq_len = 32, 50, 40
src = torch.randint(0, 10000, (batch_size, src_seq_len))
tgt = torch.randint(0, 10000, (batch_size, tgt_seq_len))

# 前向传播
output = model(src, tgt)
print(f"输出形状: {output.shape}")  # [32, 40, 10000]
```

### 2. 使用掩码

```python
from transformer.utils import create_padding_mask, create_look_ahead_mask

# 创建源序列掩码（忽略padding）
src_mask = create_padding_mask(src, pad_idx=0)

# 创建目标序列掩码（忽略padding + 防止看到未来）
tgt_mask = create_combined_mask(tgt, pad_idx=0)

# 使用掩码进行前向传播
output = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
```

### 3. 序列生成

```python
# 编码源序列
encoder_output = model.encode(src, src_mask)

# 生成目标序列
generated = model.generate(
    src=src,
    src_mask=src_mask,
    max_length=100,
    start_token=1,
    end_token=2
)
print(f"生成序列形状: {generated.shape}")
```

## 核心组件

### 1. 多头注意力 (MultiHeadAttention)

```python
from transformer.attention import MultiHeadAttention

attention = MultiHeadAttention(d_model=512, num_heads=8)
query = key = value = torch.randn(32, 50, 512)
output, attention_weights = attention(query, key, value)
```

### 2. 位置编码 (PositionalEncoding)

```python
from transformer.positional_encoding import PositionalEncoding

pos_encoding = PositionalEncoding(d_model=512, max_seq_len=5000)
x = torch.randn(50, 32, 512)  # [seq_len, batch_size, d_model]
encoded = pos_encoding(x)
```

### 3. 编码器层 (TransformerEncoderLayer)

```python
from transformer.transformer import TransformerEncoderLayer

encoder_layer = TransformerEncoderLayer(
    d_model=512,
    num_heads=8,
    d_ff=2048,
    dropout=0.1
)

x = torch.randn(32, 50, 512)
output = encoder_layer(x)
```

### 4. 解码器层 (TransformerDecoderLayer)

```python
from transformer.transformer import TransformerDecoderLayer

decoder_layer = TransformerDecoderLayer(
    d_model=512,
    num_heads=8,
    d_ff=2048,
    dropout=0.1
)

decoder_input = torch.randn(32, 40, 512)
encoder_output = torch.randn(32, 50, 512)
output = decoder_layer(decoder_input, encoder_output)
```

### 5. RoPE位置编码 (RoPE) - 新特性！

```python
from transformer.positional_encoding import RoPE

# 创建RoPE实例
rope = RoPE(d_model=64, max_seq_len=2048)

# 应用到查询和键向量
q = torch.randn(32, 8, 50, 64)  # [batch, heads, seq_len, head_dim]
k = torch.randn(32, 8, 50, 64)
q_rope, k_rope = rope(q, k)
```

### 6. 支持RoPE的多头注意力

```python
from transformer.attention import RoPEMultiHeadAttention

# 创建支持RoPE的注意力层
rope_attention = RoPEMultiHeadAttention(
    d_model=512,
    num_heads=8,
    max_seq_len=2048,
    rope_base=10000
)

x = torch.randn(32, 50, 512)
output, attention_weights = rope_attention(x, x, x)
```

### 7. 分组查询注意力 (GQA) - 新增！

```python
from transformer.attention import GroupedQueryAttention, GQASelfAttention

# 创建GQA层（8个查询头，2个KV头）
gqa = GroupedQueryAttention(
    d_model=512,
    num_heads=8,      # 查询头数
    num_kv_heads=2,   # KV头数（减少内存使用）
    dropout=0.1
)

x = torch.randn(32, 50, 512)
output, attention_weights = gqa(x, x, x)
print(f"参数减少: {1 - 2/8:.1%}")  # 约25%的参数减少

# GQA自注意力
gqa_self = GQASelfAttention(d_model=512, num_heads=8, num_kv_heads=2)
self_output, _ = gqa_self(x)
```

### 8. RoPE + GQA 组合 - 新增！

```python
from transformer.attention import RoPEGroupedQueryAttention, RoGQASelfAttention

# 结合RoPE和GQA的优势
rope_gqa = RoPEGroupedQueryAttention(
    d_model=512,
    num_heads=8,
    num_kv_heads=2,
    max_seq_len=2048,
    rope_base=10000
)

# 自注意力版本
rope_gqa_self = RoGQASelfAttention(
    d_model=512,
    num_heads=8,
    num_kv_heads=2,
    max_seq_len=2048
)

x = torch.randn(32, 50, 512)
output, attention_weights = rope_gqa_self(x)
```

## 模型配置

### 标准配置

```python
# Base模型配置
base_config = {
    'src_vocab_size': 10000,
    'tgt_vocab_size': 10000,
    'd_model': 512,
    'num_heads': 8,
    'num_encoder_layers': 6,
    'num_decoder_layers': 6,
    'd_ff': 2048,
    'dropout': 0.1
}

# Small模型配置（用于实验）
small_config = {
    'src_vocab_size': 5000,
    'tgt_vocab_size': 5000,
    'd_model': 256,
    'num_heads': 4,
    'num_encoder_layers': 3,
    'num_decoder_layers': 3,
    'd_ff': 1024,
    'dropout': 0.1
}

# Large模型配置
large_config = {
    'src_vocab_size': 50000,
    'tgt_vocab_size': 50000,
    'd_model': 1024,
    'num_heads': 16,
    'num_encoder_layers': 12,
    'num_decoder_layers': 12,
    'd_ff': 4096,
    'dropout': 0.1
}
```

## 训练示例

```python
import torch
import torch.nn as nn
import torch.optim as optim
from transformer import Transformer
from transformer.utils import create_padding_mask, create_combined_mask

# 创建模型
model = Transformer(**base_config)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# 训练循环
model.train()
for epoch in range(num_epochs):
    for batch_idx, (src, tgt) in enumerate(dataloader):
        # 准备目标序列（input和label）
        tgt_input = tgt[:, :-1]
        tgt_label = tgt[:, 1:]
        
        # 创建掩码
        src_mask = create_padding_mask(src, pad_idx=0)
        tgt_mask = create_combined_mask(tgt_input, pad_idx=0)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(src, tgt_input, src_mask, tgt_mask)
        
        # 计算损失
        loss = criterion(output.reshape(-1, output.size(-1)), tgt_label.reshape(-1))
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
```

## 工具函数

### 掩码生成

```python
from transformer.utils import (
    create_padding_mask,
    create_look_ahead_mask,
    create_combined_mask
)

# 填充掩码
padding_mask = create_padding_mask(sequence, pad_idx=0)

# 前瞻掩码
look_ahead_mask = create_look_ahead_mask(seq_len)

# 组合掩码
combined_mask = create_combined_mask(sequence, pad_idx=0)
```

### 模型工具

```python
from transformer.utils import count_parameters, init_weights

# 计算参数数量
total_params, trainable_params = count_parameters(model)
print(f"总参数: {total_params:,}, 可训练参数: {trainable_params:,}")

# 初始化权重
model.apply(init_weights)
```

## 性能优化建议

1. **批处理大小**: 根据GPU内存调整批处理大小
2. **梯度累积**: 对于大模型，使用梯度累积来模拟大批处理
3. **混合精度**: 使用`torch.cuda.amp`进行混合精度训练
4. **学习率调度**: 使用warmup和学习率衰减
5. **梯度裁剪**: 防止梯度爆炸

```python
# 混合精度训练示例
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(src, tgt_input, src_mask, tgt_mask)
    loss = criterion(output.reshape(-1, output.size(-1)), tgt_label.reshape(-1))

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## RoPE详细说明

### 什么是RoPE？

RoPE（Rotary Position Embedding，旋转位置编码）是一种先进的位置编码方法，通过旋转变换将位置信息直接编码到注意力计算中。

### RoPE的优势

1. **更好的外推能力**：可以处理比训练时更长的序列
2. **直接的相对位置建模**：自然地编码相对位置信息
3. **参数效率**：不需要额外的位置嵌入参数
4. **长序列性能**：在长序列任务上表现优异

### RoPE使用示例

```python
# 运行RoPE演示
cd transformer
python rope_example.py
```

该示例包含：
- 基本RoPE使用方法
- RoPE vs 传统位置编码比较
- 外推能力测试
- 性能分析
- 可视化展示

### 何时使用RoPE？

- ✅ 需要处理可变长度序列的任务
- ✅ 生成任务（文本生成、对话系统）
- ✅ 长文档处理
- ✅ 需要更好相对位置建模的任务

## GQA详细说明

### 什么是GQA？

GQA（Grouped Query Attention，分组查询注意力）是一种介于多头注意力（MHA）和多查询注意力（MQA）之间的高效注意力机制。它通过减少key-value头的数量来降低内存使用和计算成本。

### GQA的优势

1. **内存效率**：相比MHA减少25-75%的KV缓存大小
2. **计算优化**：减少key-value计算量，特别适合推理阶段
3. **性能平衡**：比MQA保持更好的模型质量
4. **灵活配置**：可根据资源限制调整KV头数

### GQA配置建议

| 配置 | 查询头:KV头 | 内存减少 | 性能保持 | 适用场景 |
|------|-------------|----------|----------|----------|
| GQA-4 | 8:2 或 12:4 | ~50% | 95-98% | 平衡性能和效率 |
| GQA-2 | 8:1 或 12:2 | ~75% | 90-95% | 高效推理 |
| MQA | 8:1 或 12:1 | ~90% | 85-90% | 资源受限环境 |

### GQA使用示例

```python
# 运行GQA演示
cd transformer
python gqa_example.py
```

该示例包含：
- 不同注意力机制的性能对比
- 内存使用分析
- RoPE+GQA组合演示
- 实际应用场景示例
- 性能优化建议

### 何时使用GQA？

- ✅ 大型语言模型推理优化
- ✅ 长序列处理任务
- ✅ 资源受限的部署环境
- ✅ 需要平衡性能和效率的场景
- ✅ 与RoPE配合使用以获得最佳效果

## 参考文献

- Vaswani, A., et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).
- Su, J., et al. "RoFormer: Enhanced transformer with rotary position embedding." arXiv preprint arXiv:2104.09864 (2021).
- Ainslie, J., et al. "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." arXiv preprint arXiv:2305.13245 (2023).
- Shazeer, N. "Fast transformer decoding: One write-head is all you need." arXiv preprint arXiv:1911.02150 (2019).
- The Annotated Transformer: http://nlp.seas.harvard.edu/2018/04/03/attention.html

## 许可证

MIT License 