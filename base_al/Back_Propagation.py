import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        初始化神经网络
        input_size: 输入层大小
        hidden_size: 隐藏层大小  
        output_size: 输出层大小
        learning_rate: 学习率
        """
        self.learning_rate = learning_rate
        
        # 初始化权重和偏置（使用小的随机值）
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
        
        # 存储前向传播的中间结果，用于反向传播
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None
    
    def sigmoid(self, x):
        """Sigmoid激活函数"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))  # 防止溢出
    
    def sigmoid_derivative(self, x):
        """Sigmoid函数的导数"""
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward(self, X):
        """
        前向传播
        X: 输入数据 (batch_size, input_size)
        """
        # 隐藏层
        self.z1 = np.dot(X, self.W1) + self.b1  # 线性变换
        self.a1 = self.sigmoid(self.z1)         # 激活函数
        
        # 输出层
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # 线性变换
        self.a2 = self.z2  # 输出层不使用激活函数（回归任务）
        
        return self.a2
    
    def compute_loss(self, y_pred, y_true):
        """计算均方误差损失"""
        m = y_true.shape[0]
        loss = (1/(2*m)) * np.sum((y_pred - y_true)**2)
        return loss
    
    def backward(self, X, y_true, y_pred):
        """
        反向传播算法
        X: 输入数据
        y_true: 真实标签
        y_pred: 预测值
        """
        m = X.shape[0]  # 批次大小
        
        print("=== 反向传播详细过程 ===")
        
        # 步骤1：计算输出层的误差
        dL_dz2 = (y_pred - y_true) / m  # 损失对输出层线性输出的梯度
        print(f"输出层误差 dL/dz2:\n{dL_dz2}")
        
        # 步骤2：计算输出层参数的梯度
        dL_dW2 = np.dot(self.a1.T, dL_dz2)  # 损失对W2的梯度
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)  # 损失对b2的梯度
        print(f"输出层权重梯度 dL/dW2:\n{dL_dW2}")
        print(f"输出层偏置梯度 dL/db2:\n{dL_db2}")
        
        # 步骤3：将误差传播到隐藏层
        dL_da1 = np.dot(dL_dz2, self.W2.T)  # 误差传播到隐藏层激活值
        print(f"传播到隐藏层的误差 dL/da1:\n{dL_da1}")
        
        # 步骤4：计算隐藏层的误差（考虑激活函数导数）
        dL_dz1 = dL_da1 * self.sigmoid_derivative(self.z1)  # 链式法则
        print(f"隐藏层误差 dL/dz1:\n{dL_dz1}")
        
        # 步骤5：计算隐藏层参数的梯度
        dL_dW1 = np.dot(X.T, dL_dz1)  # 损失对W1的梯度
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)  # 损失对b1的梯度
        print(f"隐藏层权重梯度 dL/dW1:\n{dL_dW1}")
        print(f"隐藏层偏置梯度 dL/db1:\n{dL_db1}")
        
        return dL_dW1, dL_db1, dL_dW2, dL_db2
    
    def update_parameters(self, dL_dW1, dL_db1, dL_dW2, dL_db2):
        """使用梯度下降更新参数"""
        self.W1 -= self.learning_rate * dL_dW1
        self.b1 -= self.learning_rate * dL_db1
        self.W2 -= self.learning_rate * dL_dW2
        self.b2 -= self.learning_rate * dL_db2
    
    def train(self, X, y, epochs=1000, print_every=100):
        """训练神经网络"""
        losses = []
        
        for epoch in range(epochs):
            # 前向传播
            y_pred = self.forward(X)
            
            # 计算损失
            loss = self.compute_loss(y_pred, y)
            losses.append(loss)
            
            # 反向传播
            if epoch == 0:  # 只在第一次迭代时打印详细过程
                dL_dW1, dL_db1, dL_dW2, dL_db2 = self.backward(X, y, y_pred)
            else:
                dL_dW1, dL_db1, dL_dW2, dL_db2 = self.backward_simple(X, y, y_pred)
            
            # 更新参数
            self.update_parameters(dL_dW1, dL_db1, dL_dW2, dL_db2)
            
            # 打印训练进度
            if epoch % print_every == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
        
        return losses
    
    def backward_simple(self, X, y_true, y_pred):
        """简化版反向传播（不打印详细信息）"""
        m = X.shape[0]
        
        # 输出层梯度
        dL_dz2 = (y_pred - y_true) / m
        dL_dW2 = np.dot(self.a1.T, dL_dz2)
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)
        
        # 隐藏层梯度
        dL_da1 = np.dot(dL_dz2, self.W2.T)
        dL_dz1 = dL_da1 * self.sigmoid_derivative(self.z1)
        dL_dW1 = np.dot(X.T, dL_dz1)
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)
        
        return dL_dW1, dL_db1, dL_dW2, dL_db2
    
    def predict(self, X):
        """预测"""
        return self.forward(X)

# 使用示例
if __name__ == "__main__":
    # 生成简单的训练数据（XOR问题的变种）
    np.random.seed(42)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)  # XOR逻辑
    
    print("训练数据:")
    print("输入 X:")
    print(X)
    print("目标 y:")
    print(y)
    print()
    
    # 创建神经网络
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=1.0)
    
    print("初始参数:")
    print(f"W1:\n{nn.W1}")
    print(f"b1:\n{nn.b1}")
    print(f"W2:\n{nn.W2}")
    print(f"b2:\n{nn.b2}")
    print()
    
    # 训练网络
    losses = nn.train(X, y, epochs=1000, print_every=200)
    
    print("\n训练后的参数:")
    print(f"W1:\n{nn.W1}")
    print(f"b1:\n{nn.b1}")
    print(f"W2:\n{nn.W2}")
    print(f"b2:\n{nn.b2}")
    
    # 测试预测
    print("\n最终预测结果:")
    predictions = nn.predict(X)
    for i in range(len(X)):
        print(f"输入: {X[i]}, 目标: {y[i][0]:.3f}, 预测: {predictions[i][0]:.3f}")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('训练损失曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')
    plt.grid(True)
    plt.show()