import numpy as np

def cross_entropy_loss(y_true, y_pred, epsilon=1e-12):
    """
    计算交叉熵损失。

    参数:
    y_true: shape=(N, C)，真实标签的 one-hot 编码
    y_pred: shape=(N, C)，模型输出的概率分布（已 softmax）
    epsilon: 防止 log(0)
    """
    # 保证概率不为 0 或 1，避免 log(0)
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    # 交叉熵计算
    loss = -np.sum(y_true * np.log(y_pred), axis=1)
    # 返回所有样本的平均损失
    return np.mean(loss)

# 示例数据
# 假设有两个样本，三类分类
y_true = np.array([[1, 0, 0],
                   [0, 1, 0]])

y_pred = np.array([[0.7, 0.2, 0.1],
                   [0.2, 0.7, 0.1]])

loss = cross_entropy_loss(y_true, y_pred)
print("交叉熵损失:", loss)