from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear_Network(nn.Module):
	def __init__(self):
		#super() function makes class inheritance more manageable and extensible
		super(Linear_Network, self).__init__()
		self.linear_1 = nn.Linear(28*28,256)
		self.linear_2 = nn.Linear(256, 128)
		self.linear_3 = nn.Linear(128, 10)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.relu(x)
		x = self.linear_1(x)
		x = self.relu(x)
		x = self.linear_2(x)
		x = self.relu(x)
		x = self.linear_3(x)
		output = F.log_softmax(x, dim = 1)
		return output
