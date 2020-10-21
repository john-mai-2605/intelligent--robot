from simulator import move_forward, turn_left, turn_right, reset_map, set_speed, show_animation, test, set_map
import numpy as np

q_table = np.load("q_table_save2.npy")
orientation = ['N','E','S','W']
actions = [move_forward, turn_left, turn_right]
show_animation(True)

thin_ice_blocks = [(1, 3), (1, 4), (1, 5), (1, 6), (2, 5), (5, 5), (5, 6)]
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
set_speed(1000000000)
# test()
count = 0
for i in range(100):
    (x, y), ori, sensor, done = reset_map()

##############################################
#### Copy and paste your step 4 code here ####
##############################################
    step = 0
    while (not done and step < 100):
        step += 1
        ori_id = orientation.index(ori)
        surrounding = encode_sensor(ori_id, sensor)     
        new_ori_id = np.random.choice(np.flatnonzero(q_table[x*8+y, surrounding] == q_table[x*8+y, surrounding].max()))
        while(ori_id != new_ori_id):
            turn_right()
            ori_id = increase(ori_id)
        (x, y), ori, sensor, done = move_forward()
        if done and (x, y) == (6, 6):
            count += 1
print("Accuracy:", count/100)
##############################################