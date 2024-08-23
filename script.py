
# This script is used to combine all the other code together with everything needed as the input.

import random
from pool import Pool
from utils import *
from pool_solver import PoolSolver
from state import gen_state_dataset
from goal import gen_goal_dataset
from check_feasibility import check_feasi_sig

def translate_pocket_name(pocket_name: str) -> str:
    pocket_translation = {
        "top_left": "lt",
        "top_right": "rt",
        "middle_left": "lc",
        "middle_right": "rc",
        "bottom_left": "lb",
        "bottom_right": "rb"
    }
    
    return pocket_translation.get(pocket_name, pocket_name)


def splitevents(goal):
    parts = goal.split()
    target_ball_color = parts[1]
    target_pocket = translate_pocket_name(parts[4])
    events = []

    if "directly" in goal and "indirectly" not in goal:
        events = [
            Event.stick_ball("cue"),
            Event.ball_collision("cue", target_ball_color),
            Event.ball_pocket(target_ball_color, target_pocket),
            # Event.ball_stop("cue")
        ]
    
    elif "bank" in goal:
        events = [
            Event.stick_ball("cue"),
            Event.ball_cushion("cue"),
            Event.ball_collision("cue", target_ball_color),
            Event.ball_pocket(target_ball_color, target_pocket),
            # Event.ball_stop("cue")
        ]

    else:
        events = [
            Event.stick_ball("cue"),
            Event.ball_collision("cue", target_ball_color),
            Event.ball_cushion(target_ball_color),
            Event.ball_pocket(target_ball_color, target_pocket),
            # Event.ball_stop("cue")
        ]
    
    return events


def bigscript(times, num_of_states, balls, pockets, num_of_balls_in_state, w, l, R, epsilon, corner_pocket_width, side_pocket_width):
    state_dataset = gen_state_dataset(num_of_states, balls, num_of_balls_in_state, w, l, R)
    goal_dataset = gen_goal_dataset(balls, pockets)

    # 因为非直接击打和翻袋击打很容易实现，这里在这两种数量够了之后，只取直接击打的目标
    direct_goal_dataset = [goal for goal in goal_dataset if "directly" in goal and "indirectly" not in goal]

    dataset = []
    two_shots = 0
    for _ in range(times):
        random_state = random.choice(state_dataset)
        random_goal = random.choice(goal_dataset) if two_shots < 2 / 3 * times else random.choice(direct_goal_dataset)

        if check_feasi_sig(random_state, random_goal, R=R, epsilon=epsilon, w=w, l=l, corner_pocket_width=corner_pocket_width, 
                    side_pocket_width=side_pocket_width):
            if "indirectly" in random_goal or "bank" in random_goal:
                two_shots += 1
            dataset.append((random_state, random_goal))
    
    with open("feasible_dataset_new.txt", "w") as f:
        for data in dataset:
            f.write(f"{data}\n")

    return dataset


def evaluate(file):
    with open(file, "r") as f:
        dataset = [line.strip() for line in f.readlines()]
    dataset = tuple([eval(data) for data in dataset])

    success = 0
    psolver = PoolSolver()
    for data in dataset[:]:
        pos_state, goal = data
        positions_dict = {key: list(pos) for key, pos in pos_state}
        state = State(positions=positions_dict)

        params, _, new_events, rating, _ = psolver.get_shot(state, splitevents(goal))
        print(f"Current index + 1: {dataset.index(data) + 1}")
        if rating == 1.0:
            success += 1
            print(f"Success at {pos_state} with goal {goal}, the best shot is {new_events}, with params {params}")
            print(f"Current success rate: {success} / {dataset.index(data) + 1}")

            with open("success_dataset_new.txt", "w") as f:
                f.write(f"{dataset.index(data), pos_state, goal, tuple(params.items())}\n")


        else:
            with open("failed_dataset_new.txt", "w") as f:
                f.write(f"{dataset.index(data), pos_state, goal, tuple(params.items())}\n")

            print(f"Failed at {pos_state} with goal {goal}, the best shot is {new_events}, with rating {rating} and params {params}")

    print(f"Success Rate: {success} / {len(dataset)} = {success / len(dataset)}")


def evaluate_ever_success(data, iter_times):
    psolver = PoolSolver()
    pos_state, goal = data
    positions_dict = {key: list(pos) for key, pos in pos_state}
    state = State(positions=positions_dict)

    success = False
    for i in range(iter_times):
        print(f"Iterating {i + 1} time(s):")
        params, _, new_events, rating, _ = psolver.get_shot(state, splitevents(goal))
        if rating == 1.0:
            success = True
            print(f"Success at {pos_state} with goal {goal}, the best shot is {new_events}, with params {params}")
            break
        else:
            print(f"Failed\n")
    
    print(f"Success: {success}\n")    
    return success


def evaluate_one(data, iter_times):
    psolver = PoolSolver()
    pos_state, goal = data
    positions_dict = {key: list(pos) for key, pos in pos_state}
    state = State(positions=positions_dict)

    success_times = 0
    for i in range(iter_times):
        print(f"Iterating {i + 1} time(s):")
        params, _, new_events, rating, _ = psolver.get_shot(state, splitevents(goal))
        if rating == 1.0:
            success_times += 1
            print(f"Success at {pos_state} with goal {goal}, the best shot is {new_events}, with params {params}")
            print(f"Current success rate: {success_times} / {i + 1} = {success_times / (i + 1)}\n")

        else:
            print(f"Failed\n")
        
    
    success_rate = success_times / iter_times
    print(f"Success Rate: {success_times} / {iter_times} = {success_rate}\n")
    return success_times, success_rate


def evaluate_all(file, iter_times):
    with open(file, "r") as f:
        dataset = [line.strip() for line in f.readlines()]
    dataset = tuple([eval(data) for data in dataset])

    # idx = random.sample(range(len(dataset)), 5000) # randomly sample 5,000

    for data in dataset[:]:
        print(f"Current index: {dataset.index(data) + 1}")
        success_times, success_rate = evaluate_one(data, iter_times)

        with open("success_rate_new.txt", "a") as f:
            f.write(f"{dataset.index(data), data}, Success Rate: {success_times} / {iter_times} = {success_rate}\n")



balls = ["red", "yellow", "blue"]
# balls = ["yellow", "blue", "red", "purple", "orange", "green", "maroon", "black"]
pockets = ["top_left", "top_right", "middle_left", "middle_right", "bottom_left", "bottom_right"]

w = 1.9812 / 2
l = 1.9812
R = 0.028575
epsilon = 0.001
corner_pocket_width = 0.118
side_pocket_width = 0.137

corner_jaw_radius = 0.0419 / 2
side_jaw_radius = 0.0159 / 2
cushion_width = 2 * 2.54 / 100


bigscript(20000, 20000, balls, pockets, 4, w, l, R, epsilon, corner_pocket_width, side_pocket_width)

# drawall("feasible_dataset_new.txt", 1)

# evaluate("feasible_dataset_new.txt")

# evaluate_all("feasible_dataset_new.txt", 10)




