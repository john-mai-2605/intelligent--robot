from simulator import move_forward, turn_left, turn_right, reset_map, set_speed, show_animation, test
# DO NOT MODIFY LINE 1
# You may import any libraries you want. But you may not import simulator_hidden

import numpy as np
import time
import random
show_animation(True)
set_speed(100000000000)          # This line is only meaningful if animations are enabled.
 
#####################################
#### Implement steps 1 to 3 here ####
#####################################
grid_size   = (8, 8)             # Size of the map
goal        = (6, 6)             # Coordinates of the goal
orientation = ['N','E','S','W']  # List of orientations

# Hyperparameters: Feel free to change all of these!
actions = [move_forward, turn_left, turn_right]
num_epochs = 35000
alpha = 0.1
gamma = 0.8
epsilon = 0.3
q_table = np.zeros([8*8,16,4])

def epsilon_greedy(state, surrounding):
    # Choose action randomly with probability epsilon: exploration
    if random.uniform(0, 1) < epsilon:
        action = random.randrange(4)
    else:
        action = np.random.choice(np.flatnonzero(q_table[state, surrounding] == q_table[state, surrounding].max()))
    return action
def increase(idx):
    if idx < 3:
        return idx + 1
    else:
        return 0
def encode_sensor(ori_id, sensor):
    if ori_id == 0:
        code = (sensor[1], sensor[2], 0, sensor[0])
    if ori_id == 1:
        code = (sensor[0], sensor[1], sensor[2], 0)
    if ori_id == 2:
        code = (0, sensor[0], sensor[1], sensor[2])
    if ori_id == 3:
        code = (sensor[2], 0, sensor[0], sensor[1])
    return code[0] + code[1]*2 + code[2]*4 + code[3]*8
# Define your reward function
def reward(x, y):
    if x in [0, 7] or y in [0, 7]:
        return -10
    elif (x, y) == (6,6): 
        return 10        
    else:
        return 0

for i in range(num_epochs):
    if (i%500 == 0):
        print("Epoch: ", i)
        np.save("q_table_epoch" + str(i), q_table)
    if i > 3*num_epochs/4:
        epsilon = 0.1
    (x, y), ori, sensor, done = reset_map()
    ori_id = orientation.index(ori)
    surrounding = encode_sensor(ori_id, sensor)     
    while not done:   
        new_ori_id = epsilon_greedy(x*8 + y, surrounding)
        while(ori_id != new_ori_id):
            turn_right()
            ori_id = increase(ori_id)
        (nx, ny), ori, sensor, done = move_forward()
        ori_id = orientation.index(ori)
        new_surrounding = encode_sensor(ori_id, sensor)    
        r = reward(nx, ny)
        if done and r == 0:
            r -= 1
        q_table[x*8+y, surrounding, new_ori_id] += alpha*(r + gamma*np.max(q_table[nx*8+ny, new_surrounding]) - q_table[x*8+y, surrounding, new_ori_id])
        x, y, surrounding = nx, ny, new_surrounding


#####################################

np.save("q_table", q_table)

set_speed(3)
test()
(x, y), ori, sensor, done = reset_map()

###############################
#### Implement step 4 here ####
###############################
while not done:
    ori_id = orientation.index(ori)
    surrounding = encode_sensor(ori_id, sensor)     
    new_ori_id = np.random.choice(np.flatnonzero(q_table[x*8+y, surrounding] == q_table[x*8+y, surrounding].max()))
    while(ori_id != new_ori_id):
        turn_right()
        ori_id = increase(ori_id)
    (x, y), ori, sensor, done = move_forward()
###############################

#### If you want to try moving around the map with your keyboard, uncomment the below lines 
# import pygame
# set_speed(5)
# show_animation(True)
# while True:
#   for event in pygame.event.get():
#       if event.type == pygame.QUIT:
#           exit("Closing...")
#       if event.type == pygame.KEYDOWN:
#           if event.key == pygame.K_LEFT: print(turn_left())
#           if event.key == pygame.K_RIGHT: print(turn_right())
#           if event.key == pygame.K_UP: print(move_forward())
#           if event.key == pygame.K_t: test()
#           if event.key == pygame.K_r: print(reset_map())
#           if event.key == pygame.K_q: exit("Closing...")