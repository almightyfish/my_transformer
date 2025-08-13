import torch.nn as nn
import torch
import math

class LayerNorm(nn.Module):
	def __init__(self, dim):
		super().__init__()
		self.eps = 1e-5
		self.scale = nn.Parameter(torch.ones(dim))
		self.bias = nn.Parameter(torch.zeros(dim))

	def forward(self, x):
		u = x.mean(dim = -1, keepdim = True)
		std = x.std(dim = -1, keepdim=True)
		y = (x - u)/torch.sqrt(std + self.eps)
		return y*self.scale + self.bias

class RMSNorm(nn.Module):
	def __init__(self, dim):
		super().__init__()
		self.eps = 1e-5
		self.gamma = nn.Parameter(torch.ones(dim))
		self.bias = nn.Parameter(torch.zeros(dim))

	def forward(self, x):
		m = torch.mean(x ** 2, dim = -1, keepdim = True)
		y = x/torch.sqrt(m + self.eps)
		return y*self.gamma + self.bias

class Simple(nn.Module):
	def __init__(self, input_dim, output_dim):
		super().__init__()
		self.layernorm = RMSNorm(input_dim)
		self.linear = nn.Linear(input_dim, output_dim)
	def forward(self, x):
		x = self.layernorm(x)
		return self.linear(x)

def train(model, dataloader, criterion, optimizer, num_epochs):
	model.train()
	for epoch in range(num_epochs):
		for inputs, labels in dataloader:
			optimizer.zero_grad()
			outputs= model(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

		print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

input_dim = 512
output_dim = 2048
num_sample = 100

x = torch.randn(num_sample, input_dim)
y = torch.randn(num_sample, output_dim)

dataset = torch.utils.data.TensorDataset(x, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

model = Simple(input_dim, output_dim)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train(model, dataloader, criterion, optimizer, num_epochs = 100)
