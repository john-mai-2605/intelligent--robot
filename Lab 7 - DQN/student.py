from simulator import move_forward, turn_left, turn_right, reset_map, set_speed, show_animation, test
# DO NOT MODIFY LINE 1
# You may import any libraries you want. But you may not import simulator_hidden
import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import math

show_animation(False)
set_speed(4000000)          # This line is only meaningful if animations are enabled.
 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

grid_size = (8, 8)
goal = (6, 6)
orientation = ['N','E','S','W']
actions = [move_forward, turn_left, turn_right]
num_epochs = 5000

def reward(coor, done):
	"""
	Task 6 (optional) - design your own reward function
	"""

	x, y = coor
	if coor == goal:
		return 2000
	# if coor[0] in [0,7] or coor[1] in [0,7]:
	# 	return -100
	elif done:
		return -100
	return -1


class ReplayMemory(object):
	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, transition):

		"""
		Task 3 - 
		push input: "transition" into replay meory
		"""
		if len(memory) < self.capacity:
			self.memory.append(transition)
		else:
			self.memory.pop(0)
			self.memory.append(transition)
		return 

	def sample(self, batch_size):
		"""
		Task 3 - 
		give a batch size, pull out batch_sized samples from the memory
		"""
		return random.sample(self.memory, batch_size)
		

	def __len__(self):
		return len(self.memory)

class DQN(nn.Module):
	def __init__(self):
		super(DQN, self).__init__()
		"""
		Task 1 -
		generate your own deep neural network
		"""
		# self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 1, padding = 1, padding_mode = 'zero')
		# self.max1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
		# self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, padding_mode = 'zero')
		# self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1, padding_mode = 'zero')
		self.linear1 = nn.Linear(4,64)
		self.relu = nn.ReLU()
		self.linear2 = nn.Linear(64, 16)		
		self.linear3 = nn.Linear(16, 8)		
		self.linear4 = nn.Linear(8, 3)		
		self.dropout = nn.Dropout()
	def forward(self, x):
		"""
		Task 1 - 
		generate your own deep neural network
		"""
		# x = self.conv1(x)
		# x = self.relu(x)
		# x = self.max1(x)
		# x = self.conv2(x)
		# x = self.relu(x)
		# x = self.max1(x)
		# x = self.conv3(x)
		# x = self.relu(x)
		# x = self.max1(x)
		# x = torch.flatten(x,1)
		
		"""
		linear layers
		"""
		x = self.dropout(x)
		# x = self.relu(x)
		x = self.linear1(x)
		x = self.relu(x)
		x = self.linear2(x)
		x = self.relu(x)
		x = self.linear3(x)		
		x = self.relu(x)
		x = self.linear4(x)
		output = F.log_softmax(x, dim = 0)
		return output
def is_move(x):
	if x == 0: 
		return 1
	else:
		return 0
def optimize_model():
	if len(memory) < BATCH_SIZE:
		return
	transition = memory.sample(BATCH_SIZE)
	"""
	Task 4: optimize model
	"""
	inp = torch.from_numpy(np.asarray([state[0] for state in transition], dtype = np.float32))
	oup = torch.from_numpy(np.asarray([state[2] for state in transition if state[2] is not None], dtype = np.float32))
	mask = torch.from_numpy(np.asarray([(state[2] is not None) for state in transition]))
	move_re = torch.from_numpy(np.asarray([is_move(state[1]) for state in transition]))
	actions = torch.from_numpy(np.asarray([state[1] for state in transition]))
	state_action_values = policy_net(inp).gather(1, actions.unsqueeze(1).long()).squeeze(1)
	state_values = torch.zeros(BATCH_SIZE)	
	state_values[mask] = target_net(oup).max(1)[0].detach()
	rewards = torch.from_numpy(np.asarray([state[3] for state in transition], dtype = np.float32))
	expected_state_action_values = GAMMA*state_values + rewards 
	optimizer.zero_grad()
	loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
	loss.backward()
	optimizer.step()

	if i %TARGET_UPDATE == 0:
		target_net.load_state_dict(policy_net.state_dict())

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(1000)



def select_action(state):
	"""
	Task 2: select action
	"""
	inp = torch.from_numpy(np.asarray(state, dtype = np.float32))
	out = policy_net.forward(inp)
	if random.uniform(0, 1) < epsilon:
		action = torch.tensor(random.randrange(3))
	else:
		with torch.no_grad():
			action = torch.argmax(out)
	return action
	

TARGET_UPDATE = 1
BATCH_SIZE = 32
GAMMA = 0.99
epsilon = 0.1
num_epochs = 2000
for i in range(num_epochs):
	print(i)
	(x, y), ori, sensor, done = reset_map()
	while not done:
		o_i = orientation.index(ori)

		cur_state = [o_i] + sensor #create your own state

		idx_action = select_action(cur_state).item()
		action = actions[idx_action]
		(new_x, new_y), new_ori, new_sensor, done = action()
		new_o_i = orientation.index(new_ori)

		reward_val = reward((new_x,new_y),done)

		new_state = [new_o_i] + new_sensor
		transition = [cur_state, idx_action, new_state, reward_val] # generate your own transition form


		memory.push(transition)
		(x, y), ori, sensor = (new_x, new_y), new_ori, new_sensor
		optimize_model()


"""
Task 5 - save your policy net
"""
torch.save(policy_net.state_dict(), 'save_net')

show_animation(True)
set_speed(500)   

def test_network():
	"""
	Task 5: test your network
	"""
	set_speed(3)
	test()
	(x, y), ori, sensor, done = reset_map()
	
	policy_net.load_state_dict(torch.load('save_net'))# load policy net

	policy_net.eval()

	while True:
		o_i = orientation.index(ori)
		"""
		fill this section to test your network
		"""
		cur_state = [o_i] + sensor #create your own state
		# inp = torch.from_numpy(np.asarray(cur_state, dtype = np.float32))
		# out = policy_net.forward(inp)
		# idx_action = torch.argmax(out)		
		epsilon = 0.02
		idx_action = select_action(cur_state)		
		action = actions[idx_action]
		(new_x, new_y), new_ori, new_sensor, done = action()
		new_o_i = orientation.index(new_ori)

		reward_val = reward((new_x,new_y),done)

		new_state = [new_o_i] + new_sensor
		transition = [cur_state, idx_action, new_state, reward_val] # generate your own transition form
		(x, y), ori, sensor = (new_x, new_y), new_ori, new_sensor
		if done:
			break
test_network()

###############################

#### If you want to try moving around the map with your keyboard, uncomment the below lines 
# import pygame
# set_speed(5)
# show_animation(True)
# while True:
# 	for event in pygame.event.get():
# 		if event.type == pygame.QUIT:
# 			exit("Closing...")
# 		if event.type == pygame.KEYDOWN:
# 			if event.key == pygame.K_LEFT: print(turn_left())
# 			if event.key == pygame.K_RIGHT: print(turn_right())
# 			if event.key == pygame.K_UP: print(move_forward())
# 			if event.key == pygame.K_t: test()
# 			if event.key == pygame.K_r: print(reset_map())
# 			if event.key == pygame.K_q: exit("Closing...")