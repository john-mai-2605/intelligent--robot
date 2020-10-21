from simulator import get_sensors, move_forward, move_backward, turn_left, turn_right, submit, set_map
# DO NOT MODIFY LINE 1
# You may import any libraries you want. But you may not import simulator_hidden
import math
import random
# Colors
black = (0,0,0)
white = (255,255,255)
gray = (100,100,100)
blue = (0,180,255)
x, y, a = 0, 0, 0
xm, ym = 0, 0
visited = []
target = []
block = []
cells = {}
##############################
#### Write your code here ####
##############################
step = 0
count = 0
def add(c, b):
	if (c + b > math.pi):
		return c + b - 2*math.pi
	if (c + b < -math.pi):
		return c + b + 2*math.pi
	return c + b
def sin(alpha):
	return round(math.sin(alpha))
def cos(alpha):
	return round(math.cos(alpha))
def move():
	c = 0
	global x, y
	while (get_sensors()[1] != 1 and get_sensors()[0][0] != black and get_sensors()[0][1] != black and c < step):
		move_forward()
		c += 1
	if (c < step): 
		reason = get_sensors()[1] == 1
		move_backward(c)	
		return False, reason
	if (c == step): 
		x = x + sin(a)
		y = y + cos(a)
	return True, None
def tag(m, n, status=None):
	if (m, n) in visited:
		cells[(m, n)] += 1
		return False
	elif (status == 'target'):
		target.append((m, n))
		visited.append((m, n))
		cells[(m, n)] = 1
		return True 
	elif (status == 'block'):
		block.append((m, n))
		cells[(m, n)] = 1000
		visited.append((m, n))
		return False
	elif (status == 'stop'):
		cells[(m, n)] = 1000
	else:
		cells[(m, n)] = 1
		visited.append((m, n))
		return True
def explore():
	global x, y, a, xm, ym
	result = False
	bound = True
	index, value = 0, 2000
	for i in range(4):
		next_cell = x + sin(a - i*math.pi/2), y + cos(a - i*math.pi/2)
		if (cells.get(next_cell, 0) < value):
			index = i
			value = cells.get(next_cell, 0)
	if index == 3:
		turn_left(count//4)
	else:
		turn_right(index*count//4)
	a = add(a, -index*math.pi/2)
	movable, blocking = move()
	if (movable):
		if get_sensors()[0][0] == blue and get_sensors()[0][1] == blue:
			result = tag(x, y, 'target')
		else:
			result = tag(x, y)
	else:
		if (blocking):
			tag(x + sin(a), y + cos(a), 'block')
		else:
			tag(x + sin(a), y + cos(a), 'stop')
			if a == 0:
				ym = max(ym, y + cos(a))
			elif a == math.pi/2:
				xm = max(xm, x + sin(a))
			if xm* ym != 0:
				for i in range(-1,xm+1):
					cells[i, -1] = 1000
					cells[i, ym] = 1000
				for i in range(-1,ym+1):
					cells[-1, i] = 1000
					cells[xm, i] = 1000
		turn_right(count//4)
		a = add(a, -math.pi/2)

# set_map((10,5), [(8,0), (2,0), (3,3), (4,1)], [(7,2), (0,2), (1,1), (2,1), (3,2), (4,0)])
offset = 0
if (get_sensors()[1] >= 7 or get_sensors()[1] == -1):
	while(get_sensors()[0][0] != gray):
		move_forward()
		offset += 1
	while(get_sensors()[0][0] == gray):
		move_forward()
		step += 1
	y += 1
	# visited.append((x,y))
	while(get_sensors()[0][0] != gray):
		move_forward()
		step += 1

re = 0
while(get_sensors()[0][0] != black and get_sensors()[0][1] != black and (get_sensors()[1] != 1)):
	previous = get_sensors()[0][0] == gray
	move_forward()
	re += 1
	if(previous and get_sensors()[0][0] != gray):
		y += 1
if offset == 0: offset = re
re = 0
if get_sensors()[0][0] == black:
	re = 0
	while (get_sensors()[0][0] == black):
		turn_left()
		re += 1
	while (get_sensors()[0][0] != black):
		turn_left()
		count += 1
	while (get_sensors()[0][0] == black):
		turn_left()
		count += 1
	turn_right(re)

elif get_sensors()[0][1] == black:
	while (get_sensors()[0][1] == black):
		turn_left()
	re = 0
	while (get_sensors()[0][1] != black):
		turn_left()
		count += 1
	while (get_sensors()[0][1] == black):
		turn_left()
		count += 1
	turn_right(re)
elif (get_sensors()[1] == 1):
	while (get_sensors()[1] == 1):
		turn_left()
		re += 1
	while (get_sensors()[1] != 1):
		turn_left()
		count += 1
	while (get_sensors()[1] == 1):
		turn_left()
		count += 1
	turn_right(re)
move_backward(offset)
turn_left(count//4)
a = add(a, math.pi/2)
if (step == 0 and (get_sensors()[1] >= 7 or get_sensors()[1] == -1)):
	re = 0
	while(get_sensors()[0][0] != gray):
		move_forward()
		re += 1
	while(get_sensors()[0][0] == gray):
		move_forward()
		step += 1
	while(get_sensors()[0][0] != gray):
		move_forward()
		step += 1
	move_backward(step+re)
if get_sensors()[0][0] == blue and get_sensors()[0][1] == blue:
	result = tag(x, y, 'target')
else:
	result = tag(x, y)
while(xm*ym == 0 or len(visited) != xm*ym):
	explore()
	# print(len(visited), block, target)
for cell, time in cells.items():
	if time < 1000:
		cells[cell] = cell[0] + cell[1]
while((x,y) != (0,0)):
	explore()
##############################
submit(target, block)
#### If you want to try moving around the map with your keyboard, uncomment the below lines 
# import pygame
# while True:
# 	pressed = pygame.key.get_pressed()
# 	if pressed[pygame.K_UP]: move_forward()
# 	if pressed[pygame.K_DOWN]: move_backward()
# 	if pressed[pygame.K_LEFT]: turn_left()
# 	if pressed[pygame.K_RIGHT]: turn_right()
# 	if pressed[pygame.K_n]: set_map((10,5), [(8,0), (4,9), (2,0), (3,3), (4,1)], [(7,2), (0,1), (2,3)])
# 	if pressed[pygame.K_c]: print(get_sensors())
# 	for event in pygame.event.get():
# 		if event.type == pygame.QUIT:
# 			exit("Closing...")