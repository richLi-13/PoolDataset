# This script is used to generate the state dataset for the pool game.

import numpy as np
import random
from goal import balls

w = 1.9812 / 2
l = 1.9812
R = 0.028575
epsilon = 0.001


# 如果是78*39，则是台球桌的外围尺寸，所以这里w和l表示的是外围尺寸，这么算每个袋口中心都会有坐标
def get_pos(w, l, R):
    return (w - 2 * R) * np.random.rand() + R, (l - 2 * R) * np.random.rand() + R

def gen_state_dataset(num_of_states, num_of_balls, w, l, R): #cnt includes the cue ball
    state_dataset = []
    if num_of_balls > len(balls) + 1 or num_of_balls < 2:
        raise ValueError("the number of color balls should be between 2 and the total number " + str(len(balls)) + ".")
    else:
        for _ in range(num_of_states):
            state = []

            # cue ball
            state.append(("cue ball", get_pos(w, l, R)))

            # color balls
            color_balls = balls.copy() # python这里这种赋值也是地址赋值，所以要用copy
            for i in range(num_of_balls - 1):
                ball_pos = get_pos(w, l, R)

                # if overlapped: regenerate
                while any(np.linalg.norm(np.array(ball_pos) - np.array(existing_balls[1])) < (2 + epsilon) * R for existing_balls in state):
                    ball_pos = get_pos(w, l, R)
                
                state.append((random.choice(color_balls), ball_pos))
                color_balls.remove(state[-1][0])

            state_dataset.append(tuple(state)) 

        with open("state_dataset.txt", "w") as f:
            for state in state_dataset:
                f.write(f"{state}\n")
        
        return state_dataset

gen_state_dataset(10000, 3, w, l, R)


        


