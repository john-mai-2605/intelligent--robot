from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

class Custom_Network(nn.Module):
	def __init__(self):
		#super() function makes class inheritance more manageable and extensible
		super(Custom_Network, self).__init__()
		pass

		""" 
		fill in init section to create Convolutional Neural Network

		"""
		self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 1, padding = 1, padding_mode = 'zero')
		self.max1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
		self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, padding_mode = 'zero')
		self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1, padding_mode = 'zero')
		self.linear1 = nn.Linear(4*4*128,32)
		self.relu = nn.ReLU()
		self.linear2 = nn.Linear(32, 2)		
		self.dropout = nn.Dropout(0.4)
		self.dropoutc = nn.Dropout(0.2)
		self.bn = nn.BatchNorm1d(32)
		self.bn2 = nn.BatchNorm1d(2)

	def forward(self, x):

		""" 
		fill in forward section to create Convolutional Neural Network

		"""

		"""
		3-layered Conv NN
		"""
		x = self.conv1(x)
		x = self.relu(x)
		x = self.max1(x)
		x = self.conv2(x)
		x = self.relu(x)
		x = self.max1(x)
		x = self.conv3(x)
		x = self.relu(x)
		x = self.max1(x)
		#x = self.bn2(x)
		x = self.dropoutc(x)
		x = torch.flatten(x,1)
		
		"""
		linear layers
		"""
		x = self.dropout(x)
		x = self.relu(x)
		x = self.linear1(x)
		x = self.dropout(x)
		x = self.relu(x)
		x = self.bn(x)
		x = self.linear2(x)
		output = F.log_softmax(x, dim = 1)
		return output
