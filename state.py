# This script is used to generate the state dataset for the pool game.

import numpy as np
import random
from goal import balls

w = 1.9812 / 2
l = 1.9812
R = 0.028575
epsilon = 0.001
cushion_width = 2 * 2.54 / 100


def get_pos(w, l, R):
    return (w - 3 * R) * np.random.rand() + 1.5 * R, (l - 3 * R) * np.random.rand() + 1.5 * R

def gen_state_dataset(num_of_states, balls, num_of_balls_in_state,  w, l, R): #cnt includes the cue ball
    state_dataset = []
    if num_of_balls_in_state > len(balls) + 1 or num_of_balls_in_state < 2:
        raise ValueError("the number of color balls should be between 2 and the total number " + str(len(balls)) + ".")
    else:
        for _ in range(num_of_states):
            state = []

            # cue ball
            state.append(("cue", get_pos(w, l, R)))

            # color balls
            color_balls = balls.copy() # python这里这种赋值也是地址赋值，所以要用copy
            for i in range(num_of_balls_in_state - 1):
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

# gen_state_dataset(100, balls, 4, w, l, R)


        


