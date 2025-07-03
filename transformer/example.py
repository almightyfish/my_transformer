#!/usr/bin/env python3
"""
Transformer使用示例

这个文件展示了如何使用transformer包进行：
1. 模型创建和配置
2. 数据准备
3. 训练循环
4. 推理和生成
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple

# 如果在包内运行，使用相对导入
try:
    from . import Transformer
    from .utils import create_padding_mask, create_combined_mask, count_parameters
except ImportError:
    # 如果直接运行该文件，使用绝对导入
    from transformer import Transformer
    from transformer.utils import create_padding_mask, create_combined_mask, count_parameters


class SimpleDataset:
    """
    简单的序列到序列数据集
    用于演示目的，生成随机的序列对
    """
    
    def __init__(self, vocab_size=1000, seq_len=50, num_samples=10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 生成随机源序列
        src = torch.randint(1, self.vocab_size-1, (self.seq_len,))
        
        # 生成目标序列（简单的反转操作作为示例）
        tgt = torch.flip(src, [0])
        
        # 添加开始和结束标记
        tgt = torch.cat([torch.tensor([1]), tgt, torch.tensor([2])])  # 1=start, 2=end
        
        return src, tgt


def create_dataloader(dataset, batch_size=32, shuffle=True):
    """创建数据加载器"""
    def collate_fn(batch):
        src_batch, tgt_batch = zip(*batch)
        
        # 填充到相同长度
        src_batch = torch.nn.utils.rnn.pad_sequence(
            src_batch, batch_first=True, padding_value=0
        )
        tgt_batch = torch.nn.utils.rnn.pad_sequence(
            tgt_batch, batch_first=True, padding_value=0
        )
        
        return src_batch, tgt_batch
    
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )


def train_model():
    """训练模型示例"""
    print("=" * 50)
    print("开始训练Transformer模型")
    print("=" * 50)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 模型配置
    config = {
        'src_vocab_size': 1000,
        'tgt_vocab_size': 1000,
        'd_model': 256,
        'num_heads': 8,
        'num_encoder_layers': 3,
        'num_decoder_layers': 3,
        'd_ff': 1024,
        'max_seq_len': 100,
        'dropout': 0.1
    }
    
    # 创建模型
    model = Transformer(**config).to(device)
    print(f"模型创建完成")
    
    # 计算参数数量
    total_params, trainable_params = count_parameters(model)
    print(f"总参数: {total_params:,}, 可训练参数: {trainable_params:,}")
    
    # 创建数据集和数据加载器
    dataset = SimpleDataset(vocab_size=config['src_vocab_size'], seq_len=50, num_samples=1000)
    dataloader = create_dataloader(dataset, batch_size=8, shuffle=True)
    print(f"数据集大小: {len(dataset)}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    
    # 训练循环
    model.train()
    num_epochs = 5
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (src, tgt) in enumerate(dataloader):
            src, tgt = src.to(device), tgt.to(device)
            
            # 准备目标序列
            tgt_input = tgt[:, :-1]  # 去掉最后一个标记作为输入
            tgt_label = tgt[:, 1:]   # 去掉第一个标记作为标签
            
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 20 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, '
                      f'Batch: {batch_idx+1}/{len(dataloader)}, '
                      f'Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch+1} 完成, 平均损失: {avg_loss:.4f}')
        print("-" * 30)
    
    return model, device


def inference_example(model, device):
    """推理示例"""
    print("\n" + "=" * 50)
    print("推理示例")
    print("=" * 50)
    
    model.eval()
    
    # 创建示例输入
    src = torch.randint(1, 999, (1, 10)).to(device)  # 批处理大小=1，序列长度=10
    print(f"源序列: {src.squeeze().tolist()}")
    
    # 创建源掩码
    src_mask = create_padding_mask(src, pad_idx=0)
    
    with torch.no_grad():
        # 方法1: 使用模型的generate方法
        generated = model.generate(
            src=src,
            src_mask=src_mask,
            max_length=15,
            start_token=1,
            end_token=2
        )
        print(f"生成序列: {generated.squeeze().tolist()}")
        
        # 方法2: 手动逐步生成
        print("\n手动逐步生成:")
        encoder_output = model.encode(src, src_mask)
        
        # 初始化目标序列
        tgt = torch.tensor([[1]], device=device)  # 开始标记
        
        for step in range(15):
            # 创建目标掩码
            tgt_size = tgt.size(1)
            tgt_mask = torch.triu(torch.ones(tgt_size, tgt_size, device=device), diagonal=1)
            tgt_mask = (tgt_mask == 0).unsqueeze(0).unsqueeze(1)
            
            # 解码
            decoder_output = model.decode(tgt, encoder_output, tgt_mask, src_mask)
            
            # 获取下一个标记的概率分布
            next_token_logits = model.output_projection(decoder_output[:, -1, :])
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            
            # 贪心解码
            next_token = torch.argmax(next_token_probs, dim=-1, keepdim=True)
            
            print(f"步骤 {step+1}: 预测标记 {next_token.item()}, "
                  f"概率 {next_token_probs.max().item():.4f}")
            
            # 添加到序列
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # 如果生成结束标记，停止
            if next_token.item() == 2:
                break
        
        print(f"手动生成序列: {tgt.squeeze().tolist()}")


def attention_analysis_example(model, device):
    """注意力分析示例"""
    print("\n" + "=" * 50)
    print("注意力分析示例")
    print("=" * 50)
    
    # 为了分析注意力，我们需要修改模型以返回注意力权重
    # 这里展示如何获取单层的注意力权重
    
    model.eval()
    src = torch.randint(1, 999, (1, 8)).to(device)
    tgt = torch.randint(1, 999, (1, 6)).to(device)
    
    # 创建掩码
    src_mask = create_padding_mask(src, pad_idx=0)
    tgt_mask = create_combined_mask(tgt, pad_idx=0)
    
    with torch.no_grad():
        # 获取嵌入
        src_emb = model.src_embedding(src) * torch.sqrt(torch.tensor(model.d_model, dtype=torch.float32))
        src_emb = src_emb.transpose(0, 1)
        src_emb = model.pos_encoding(src_emb)
        src_emb = src_emb.transpose(0, 1)
        
        # 通过第一个编码器层并获取注意力权重
        encoder_layer = model.encoder.layers[0]
        attn_output, attention_weights = encoder_layer.self_attention(src_emb, src_emb, src_emb, src_mask)
        
        print(f"注意力权重形状: {attention_weights.shape}")
        print(f"平均注意力权重前3x3:")
        avg_attention = attention_weights.mean(dim=1)  # 平均所有头
        print(avg_attention[0, :3, :3].cpu().numpy())


def model_comparison():
    """不同配置模型的比较"""
    print("\n" + "=" * 50)
    print("模型配置比较")
    print("=" * 50)
    
    configs = {
        'Small': {
            'src_vocab_size': 1000, 'tgt_vocab_size': 1000,
            'd_model': 128, 'num_heads': 4,
            'num_encoder_layers': 2, 'num_decoder_layers': 2,
            'd_ff': 512, 'dropout': 0.1
        },
        'Medium': {
            'src_vocab_size': 1000, 'tgt_vocab_size': 1000,
            'd_model': 256, 'num_heads': 8,
            'num_encoder_layers': 4, 'num_decoder_layers': 4,
            'd_ff': 1024, 'dropout': 0.1
        },
        'Large': {
            'src_vocab_size': 1000, 'tgt_vocab_size': 1000,
            'd_model': 512, 'num_heads': 8,
            'num_encoder_layers': 6, 'num_decoder_layers': 6,
            'd_ff': 2048, 'dropout': 0.1
        }
    }
    
    for name, config in configs.items():
        model = Transformer(**config)
        total_params, trainable_params = count_parameters(model)
        print(f"{name} 模型:")
        print(f"  - d_model: {config['d_model']}")
        print(f"  - 层数: {config['num_encoder_layers']}E + {config['num_decoder_layers']}D")
        print(f"  - 参数数量: {total_params:,}")
        print(f"  - 内存占用估算: {total_params * 4 / 1024**2:.1f} MB")
        print()


def main():
    """主函数"""
    print("Transformer架构演示")
    print("这个示例展示了如何使用自定义的Transformer实现")
    
    # 1. 模型比较
    model_comparison()
    
    # 2. 训练示例
    trained_model, device = train_model()
    
    # 3. 推理示例
    inference_example(trained_model, device)
    
    # 4. 注意力分析
    attention_analysis_example(trained_model, device)
    
    print("\n" + "=" * 50)
    print("演示完成!")
    print("=" * 50)


if __name__ == "__main__":
    # 设置随机种子以获得可重现的结果
    torch.manual_seed(42)
    np.random.seed(42)
    
    main() 