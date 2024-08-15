#     R: The radius of the ball (*default* = 0.028575).
# # 7-foot table (78x39 in^2 playing surface)
    # l: float = field(default=1.9812)  # noqa  E741
    # w: float = field(default=1.9812 / 2)

# cushion_width: float = field(default=2 * 2.54 / 100)
# cushion_height: float = field(default=0.64 * 2 * 0.028575)


# corner_pocket_width: float = field(default=0.118)
# corner_pocket_angle: float = field(default=5.3)  # degrees
# corner_pocket_depth: float = field(default=0.0398)
# corner_pocket_radius: float = field(default=0.124 / 2)
# corner_jaw_radius: float = field(default=0.0419 / 2)


# side_pocket_width: float = field(default=0.137)
# side_pocket_angle: float = field(default=7.14)  # degrees
# side_pocket_depth: float = field(default=0.00437)
# side_pocket_radius: float = field(default=0.129 / 2)
# side_jaw_radius: float = field(default=0.0159 / 2)

import numpy as np
import math
import random
import goal
import state



w = 1.9812 / 2
l = 1.9812
R = 0.028575
corner_jaw_radius = 0.0419 / 2
corner_pocket_width = 0.118
side_pocket_width = 0.137
side_jaw_radius = 0.0159 / 2
cushion_width = 2 * 2.54 / 100

epsilon = 0.001

dataset = []

def get_pocket_pos(w=1.9812 / 2, l=1.9812):
    return {
        'top_left': (0, l),
        'top_right': (w, l),
        'bottom_left': (0, 0),
        'bottom_right': (w, 0),
        'middle_left': (0, l / 2),
        'middle_right': (w, l / 2)
    }

def which_pocket_width(pocket, corner_pocket_width=corner_pocket_width, side_pocket_width=side_pocket_width):
    if pocket in ['top_left', 'top_right', 'bottom_left', 'bottom_right']:
        return corner_pocket_width
    else:
        return side_pocket_width


def dis(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def get_intersec_x(p1, p2, x):
    # get the intersection point on the line y = kx + b
    k = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = p1[1] - k * p1[0]
    return x, k * x + b


def get_intersec_y(p1, p2, y):
    # get the intersection point on the line x = ky + b
    k = (p2[0] - p1[0]) / (p2[1] - p1[1])
    b = p1[0] - k * p1[1]
    return k * y + b, y

def get_angle(v1, v2):
    # get the angle between two vectors
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(cos_theta))


def get_bank_pos(ball_pos, target_pocket, cue_ball=False, cue_ball_pos=None, w=w, l=l):
    bank_pos = []
    xl = 0
    xr = w
    yt = l
    yb = 0

    # 看是（1）翻袋，与白球连线取bank_pos；（2）非直接击打，与目标袋连线取bank_pos
    # bank shot, cue ball touches the cushion
    if (cue_ball == True): 

        # 对目标袋反向取对称点,然后与白球连线取bank_pos
        if (target_pocket == 'top_left'):
            sym_pos_1 = (2 * xr - ball_pos[0], ball_pos[1])
            bank_pos.append(get_intersec_x(cue_ball_pos, sym_pos_1, xr))
            sym_pos_2 = (ball_pos[0], 2 * yb - ball_pos[1])
            bank_pos.append(get_intersec_y(sym_pos_2, cue_ball_pos, yb))

        elif (target_pocket == 'top_right'):
            sym_pos_1 = (2 * xl - ball_pos[0], ball_pos[1])
            bank_pos.append(get_intersec_x(cue_ball_pos, sym_pos_1, xl))
            sym_pos_2 = (ball_pos[0], 2 * yb - ball_pos[1])
            bank_pos.append(get_intersec_y(sym_pos_2, cue_ball_pos, yb))

        elif (target_pocket == 'bottom_left'):
            sym_pos_1 = (2 * xr - ball_pos[0], ball_pos[1])
            bank_pos.append(get_intersec_x(cue_ball_pos, sym_pos_1, xr))
            sym_pos_2 = (ball_pos[0], 2 * yt - ball_pos[1])
            bank_pos.append(get_intersec_y(sym_pos_2, cue_ball_pos, yt))

        elif (target_pocket == 'bottom_right'):
            sym_pos_1 = (2 * xl - ball_pos[0], ball_pos[1])
            bank_pos.append(get_intersec_x(cue_ball_pos, sym_pos_1, xl))
            sym_pos_2 = (ball_pos[0], 2 * yt - ball_pos[1])
            bank_pos.append(get_intersec_y(sym_pos_2, cue_ball_pos, yt))
        
        elif (target_pocket == 'middle_left'):
            sym_pos_1 = (2 * xr - ball_pos[0], ball_pos[1])
            bank_pos.append(get_intersec_x(cue_ball_pos, sym_pos_1, xr))
            sym_pos_2 = (ball_pos[0], 2 * yt - ball_pos[1])
            bank_pos.append(get_intersec_y(sym_pos_2, cue_ball_pos, yt))
            sym_pos_3 = (ball_pos[0], 2 * yb - ball_pos[1])
            bank_pos.append(get_intersec_y(sym_pos_3, cue_ball_pos, yb))
        
        elif (target_pocket == 'middle_right'):
            sym_pos_1 = (2 * xl - ball_pos[0], ball_pos[1])
            bank_pos.append(get_intersec_x(cue_ball_pos, sym_pos_1, xl))
            sym_pos_2 = (ball_pos[0], 2 * yt - ball_pos[1])
            bank_pos.append(get_intersec_y(sym_pos_2, cue_ball_pos, yt))
            sym_pos_3 = (ball_pos[0], 2 * yb - ball_pos[1])
            bank_pos.append(get_intersec_y(sym_pos_3, cue_ball_pos, yb))

        return bank_pos

    # indirect shot, target ball touches the cushion (only once)
    else:
        target_pocket_pos = get_pocket_pos(w=w, l=l)[target_pocket]

        # 对目标袋反向取对称点，然后与目标袋连线取bank_pos
        if (target_pocket == 'top_left'):
            sym_pos_1 = (2 * xr - ball_pos[0], ball_pos[1])
            bank_pos.append(get_intersec_x(target_pocket_pos, sym_pos_1, xr))
            sym_pos_2 = (ball_pos[0], 2 * yb - ball_pos[1])
            bank_pos.append(get_intersec_y(sym_pos_2, target_pocket_pos, yb))
        
        elif (target_pocket == 'top_right'):
            sym_pos_1 = (2 * xl - ball_pos[0], ball_pos[1])
            bank_pos.append(get_intersec_x(target_pocket_pos, sym_pos_1, xl))
            sym_pos_2 = (ball_pos[0], 2 * yb - ball_pos[1])
            bank_pos.append(get_intersec_y(sym_pos_2, target_pocket_pos, yb))
        
        elif (target_pocket == 'bottom_left'):
            sym_pos_1 = (2 * xr - ball_pos[0], ball_pos[1])
            bank_pos.append(get_intersec_x(target_pocket_pos, sym_pos_1, xr))
            sym_pos_2 = (ball_pos[0], 2 * yt - ball_pos[1])
            bank_pos.append(get_intersec_y(sym_pos_2, target_pocket_pos, yt))
        
        elif (target_pocket == 'bottom_right'):
            sym_pos_1 = (2 * xl - ball_pos[0], ball_pos[1])
            bank_pos.append(get_intersec_x(target_pocket_pos, sym_pos_1, xl))
            sym_pos_2 = (ball_pos[0], 2 * yt - ball_pos[1])
            bank_pos.append(get_intersec_y(sym_pos_2, target_pocket_pos, yt))
        
        elif (target_pocket == 'middle_left'):
            sym_pos_1 = (2 * xr - ball_pos[0], ball_pos[1])
            bank_pos.append(get_intersec_x(target_pocket_pos, sym_pos_1, xr))
            sym_pos_2 = (ball_pos[0], 2 * yt - ball_pos[1])
            bank_pos.append(get_intersec_y(sym_pos_2, target_pocket_pos, yt))
            sym_pos_3 = (ball_pos[0], 2 * yb - ball_pos[1])
            bank_pos.append(get_intersec_y(sym_pos_3, target_pocket_pos, yb))
        
        elif (target_pocket == 'middle_right'):
            sym_pos_1 = (2 * xl - ball_pos[0], ball_pos[1])
            bank_pos.append(get_intersec_x(target_pocket_pos, sym_pos_1, xl))
            sym_pos_2 = (ball_pos[0], 2 * yt - ball_pos[1])
            bank_pos.append(get_intersec_y(sym_pos_2, target_pocket_pos, yt))
            sym_pos_3 = (ball_pos[0], 2 * yb - ball_pos[1])
            bank_pos.append(get_intersec_y(sym_pos_3, target_pocket_pos, yb))
  
        return bank_pos


def is_ball_in_path(start, end, ball, tolerance=epsilon):
    # check if the ball is in the path from start to end
    d1 = dis(start, ball)
    d2 = dis(end, ball)
    d3 = dis(start, end)
    return np.sqrt(d1**2 - (2*R)**2) + np.sqrt(d2**2 - (2*R)**2) <= d3 + tolerance * R


def check_feasi_sig(state, goal, R=R, epsilon=epsilon, w=w, l=l, 
                    corner_pocket_width=corner_pocket_width, side_pocket_width=side_pocket_width):
    # check if the goal is feasible given the state

    cue_ball_pos = state[0][1]

    parts = goal.split()
    target_ball_color = parts[1]
    target_pocket = parts[4]
    target_pocket_pos = get_pocket_pos(w=w, l=l)[target_pocket]

    # 1. check if target ball in state
    for ball_color, ball_pos in state[1:]:
        if ball_color == target_ball_color:
            target_ball_pos = ball_pos
            break
    # else部分与for循环配合使用，表示如果循环正常结束（没有通过break提前终止），则执行else中的代码
    else:    
        return False 

    # 2. check whether the difference between R and pocket width is bigger than tolerance  
    if which_pocket_width(target_pocket, corner_pocket_width=corner_pocket_width, side_pocket_width=side_pocket_width) / 2 - R <= epsilon * R:
        return False   
    
    # 3. calculate the ghost ball position
    target_to_pocket = np.array(target_pocket_pos) - np.array(target_ball_pos)
    ghost_ball_x = target_ball_pos[0] - target_to_pocket[0] * R / dis(target_ball_pos, target_pocket_pos)
    ghost_ball_y = target_ball_pos[1] - target_to_pocket[1] * R / dis(target_ball_pos, target_pocket_pos)
    ghost_ball_pos = (ghost_ball_x, ghost_ball_y)


    if "directly" in goal and "indirectly" not in goal:
        # a direct shot

        # (1) check the angle between the line connecting the centre of the cue ball to the ghost ball
        #     and the line connecting the centre of the target ball to the target pocket
        cue_to_ghost = np.array(ghost_ball_pos) - np.array(cue_ball_pos)
        # target_to_pocket = np.array(target_pocket_pos) - np.array(target_ball_pos)
        angle = get_angle(cue_to_ghost, target_to_pocket)
        if angle > 75:
            return False


        # (2) check if there's no other ball (** except the target ball **) in the path 
        #     from the cue ball to the ghost ball, and from the target ball to the target pocket
        for ball_color, ball_pos in state[1:]:
            if ball_color != target_ball_color and (is_ball_in_path(cue_ball_pos, ghost_ball_pos, ball_pos) 
                                                or is_ball_in_path(target_ball_pos, target_pocket_pos, ball_pos)):
                return False
        
        return True

        
    elif "bank" in goal:
        # a bank shot, cue ball touches the cushion
        bank_pos = get_bank_pos(ghost_ball_pos, target_pocket, cue_ball=True, cue_ball_pos=cue_ball_pos, w=w, l=l)

        for pos in bank_pos:

        # (1) check if bank_pos not in the range of a side pocket 
        #### to check: 球在什么地方的时候算被球袋影响；这里是球中心从下袋口到上袋口的范围
            if pos[1] >= (l - side_pocket_width) / 2 and pos[1] <= (l + side_pocket_width) / 2:
                bank_pos.remove(pos)  
                break   
     
        # (2) check for each bank position: check the angle 
        #     between the line connecting it to the centre of the ghost ball 
        #     and the line connecting the centre of the target ball to the target pocket 
            bank_to_ghost = np.array(ghost_ball_pos) - np.array(pos)
            # target_to_pocket = np.array(target_pocket_pos) - np.array(target_ball_pos)
            angle = get_angle(bank_to_ghost, target_to_pocket)
            if angle > 75:
                bank_pos.remove(pos) 
                break
        
        # (3) check the path from the cue ball to the bank position, 
        #     from the bank position to the ghost ball, and from target ball to the target pocket
            for ball_color, ball_pos in state[1:]:
                if ball_color != target_ball_color and (is_ball_in_path(cue_ball_pos, pos, ball_pos) 
                                                    or is_ball_in_path(pos, ghost_ball_pos, ball_pos) 
                                                    or is_ball_in_path(target_ball_pos, target_pocket_pos, ball_pos)):
                    bank_pos.remove(pos) 
                    break
        
        if bank_pos == []:
            return False

        return True

    else:
        # an indirect shot, target ball touches the cushion (only once)
        bank_pos = get_bank_pos(target_ball_pos, target_pocket, w=w, l=l)

        for pos in bank_pos:

        # (1) check if bank_pos not in the range of a side pocket 
        #### to check: 球在什么地方的时候算被球袋影响；这里是球中心从下袋口到上袋口的范围
            if pos[1] >= (l - side_pocket_width) / 2 and pos[1] <= (l + side_pocket_width) / 2:
                bank_pos.remove(pos)     
                break
        
        # (2) check for each bank position: check the angle
        #     between the line connecting it to the centre of the target ball  
        #     and the line connecting cue ball to the target ball (又一个可能的误差；这里应该是白球到假想球)
            cue_to_target = np.array(target_ball_pos) - np.array(cue_ball_pos)
            target_to_bank = np.array(pos) - np.array(target_ball_pos)
            angle = get_angle(cue_to_target, target_to_bank)
            if angle > 75:
                bank_pos.remove(pos)     
                break

        # (3) calculate the ghost ball position
            ghost_ball_x = target_ball_pos[0] - target_to_bank[0] * R / dis(target_ball_pos, bank_pos)
            ghost_ball_y = target_ball_pos[1] - target_to_bank[1] * R / dis(target_ball_pos, bank_pos)
            ghost_ball_pos = (ghost_ball_x, ghost_ball_y)

        # (4) check the path from the cue ball to the ghost ball,
        #     from the target ball to the bank position, and from the bank position to the target pocket
            for ball_color, ball_pos in state[1:]:
                if ball_color != target_ball_color and (is_ball_in_path(cue_ball_pos, ghost_ball_pos, ball_pos) 
                                                    or is_ball_in_path(target_ball_pos, pos, ball_pos) 
                                                    or is_ball_in_path(pos, target_pocket_pos, ball_pos)):
                    bank_pos.remove(pos)     
                    break

        if bank_pos == []:
            return False
        
        return True


def check_feasi(times):
    goal_file = '/Users/richquinn13/python_projects/Pool Dataset/goal_dataset.txt'
    state_file = '/Users/richquinn13/python_projects/Pool Dataset/state_dataset.txt'

    for _ in range(times):
        with open(goal_file, 'r') as f:
            goals = f.readlines()
            random_goal = random.choice(goals).strip()

        states = []
        with open(state_file, 'r') as f:
            for line in f.readlines():
                state = eval(line.strip())
                states.append(state)
            random_state = random.choice(states)

        if check_feasi_sig(random_state, random_goal):
            dataset.append((random_state, random_goal))
        
    with open("feasible_dataset.txt", "w") as f:
        for data in dataset:
            f.write(str(data) + "\n")    

# check_feasi(100)


