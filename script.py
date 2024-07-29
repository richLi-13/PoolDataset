
# This script is used to combine all the other code together with everything needed as the input.

import random
from state import gen_state_dataset
from goal import gen_goal_dataset
from check_feasibility import check_feasi_sig


def bigscript(times, num_of_states, balls, pockets, num_of_balls, w, l, R, epsilon, cushion_width, side_pocket_width, side_jaw_radius, corner_jaw_radius):
    state_dataset = gen_state_dataset(num_of_states, num_of_balls, w, l, R)
    goal_dataset = gen_goal_dataset(balls, pockets)

    dataset = []

    for _ in range(times):
        random_state = random.choice(state_dataset)
        random_goal = random.choice(goal_dataset)

        if check_feasi_sig(random_state, random_goal, R=R, epsilon=epsilon, w=w, l=l, cushion_width=cushion_width, 
                    side_pocket_width=side_pocket_width, side_jaw_radius=side_jaw_radius, corner_jaw_radius=corner_jaw_radius):
            dataset.append((random_state, random_goal))
    
    with open("feasible_dataset_new.txt", "w") as f:
        for data in dataset:
            f.write(f"{data}\n")

    return dataset


balls = ["yellow", "blue", "red", "purple", "orange", "green", "maroon", "black"]
pockets = ["top_left", "top_right", "middle_left", "middle_right", "bottom_left", "bottom_right"]

w = 1.9812 / 2
l = 1.9812
R = 0.028575
epsilon = 0.001
corner_jaw_radius = 0.0419 / 2
side_pocket_width = 0.137
side_jaw_radius = 0.0159 / 2
cushion_width = 2 * 2.54 / 100


bigscript(1000, 10000, balls, pockets, 3, w, l, R, epsilon, cushion_width, side_pocket_width, side_jaw_radius, corner_jaw_radius)



