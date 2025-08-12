import torch
import numpy as np

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    计算广义优势估计 (Generalized Advantage Estimation, GAE)
    
    GAE是一种在强化学习中用于计算优势函数的方法，它平衡了偏差和方差的权衡。
    优势函数 A(s,a) = Q(s,a) - V(s) 表示在状态s下选择动作a相比于平均水平的优势。
    
    参数:
        rewards: 1D Tensor or np.ndarray, shape = (T,)    # 每步获得的奖励
        values: 1D Tensor or np.ndarray, shape = (T+1,)   # 状态值函数，包含最后一个状态的值
        dones: 1D Tensor or np.ndarray, shape = (T,)      # 每步的终止标记（1表示结束，0表示继续）
        gamma: float, 折扣因子 (默认0.99)                  # 未来奖励的折扣率
        lam: float, GAE参数 (默认0.95)                     # 控制偏差-方差权衡的参数
    
    返回值:
        advantages: shape = (T,) - 每步的优势估计
        returns: shape = (T,) - 每步的回报估计 (用于训练价值函数)
    """
    T = len(rewards)  # 时间步数
    advantages = torch.zeros(T, dtype=torch.float32)  # 初始化优势数组
    last_gae_lam = 0  # 上一步的GAE值，用于递推计算

    # 从最后一步开始向前计算，这样可以利用递推关系
    for t in reversed(range(T)):
        # 判断当前步骤是否为终止步骤
        if dones[t]:
            # 如果是终止步骤，下一个状态不存在
            next_non_terminal = 0.0  # 非终止标记为0
            next_value = 0.0        # 下一状态值为0
        else:
            # 如果不是终止步骤，正常计算
            next_non_terminal = 1.0     # 非终止标记为1
            next_value = values[t + 1]  # 下一状态的值
        
        # 计算TD误差: δ_t = r_t + γ * V(s_{t+1}) * (1-done) - V(s_t)
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        
        # 计算GAE: A_t = δ_t + γ * λ * (1-done) * A_{t+1}
        # 这是GAE的核心递推公式，结合了当前TD误差和未来优势的折扣值
        advantages[t] = delta + gamma * lam * next_non_terminal * last_gae_lam
        
        # 更新last_gae_lam为当前计算的优势值，供下一步（实际上是前一步）使用
        last_gae_lam = advantages[t]
    
    # 计算回报: R_t = A_t + V(s_t)
    # 这用于训练价值函数，使V(s_t)趋近于真实的回报
    returns = advantages + values[:-1]  # values[:-1]去掉最后一个元素，保持形状一致
    
    return advantages, returns

# ==================== 示例用法 ====================
if __name__ == "__main__":
    # 模拟一个包含5个时间步的episode
    print("GAE计算示例：")
    print("=" * 50)
    
    # 每步获得的奖励 (5步)
    rewards = torch.tensor([1.0, 0.5, 0.0, 0.2, 1.0])
    print(f"奖励序列: {rewards.tolist()}")
    
    # 状态值函数 (6个值：包含初始状态和所有状态转换后的状态值)
    values = torch.tensor([0.2, 0.4, 0.6, 0.3, 0.1, 0.0])
    print(f"状态值序列: {values.tolist()}")
    
    # 终止标记 (5步：0表示继续，1表示episode结束)
    dones = torch.tensor([0, 0, 0, 0, 1])
    print(f"终止标记: {dones.tolist()}")
    print()
    
    # 计算GAE
    advantages, returns = compute_gae(rewards, values, dones)
    
    # 输出结果
    print("计算结果：")
    print(f"优势估计 (Advantages): {advantages.tolist()}")
    print(f"回报估计 (Returns): {returns.tolist()}")
    print()
    
    # 解释结果
    print("结果解释：")
    print("- 优势估计反映了每个动作相比于平均水平的好坏")
    print("- 正值表示该动作比平均水平好，负值表示比平均水平差")
    print("- 回报估计用于训练价值函数，使其更准确地预测未来回报")