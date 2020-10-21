from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from model import Conv_Network
import torchvision
import torchvision.datasets
import torchvision.transforms 
from datetime import datetime

def train(model, device, train_loader, optimizer, epoch):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()

		if batch_idx % 10 ==0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
	model.eval()
	test_loss = 0
	correct = 0
	# no_grad() prevents codes from tracking records or using memories 
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			test_loss += F.nll_loss(output, target, reduction = 'sum').item()
			pred = output.argmax(dim = 1, keepdim = True) # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()
	test_loss/=len(test_loader.dataset)
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))
def log():
	# read file into string
	with open('model.py', 'r') as inputfile:
		textstr = inputfile.read()
		fn = datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + ".txt"
		with open("logs/"+fn, 'w') as outputfile:
			outputfile.write(textstr)


def main():
	log()

	transform_train = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])
	transform_test = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
	train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

	testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
	test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = Conv_Network().to(device)
	optimizer = optim.Adam(model.parameters(), lr = 0.001)
	scheduler = StepLR(optimizer, step_size = 1, gamma = 0.8)


	for epoch in range(500):
		train(model, device, train_loader, optimizer, epoch)
		test(model, device, test_loader)


if __name__ == "__main__":
	main()