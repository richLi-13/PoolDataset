from utils import State, Event
from pool import Pool
import pooltool as pt

p = Pool(visualizable=True)


def visualise(file, index=-1):
    with open(file, "r") as f:
        dataset = [line.strip() for line in f.readlines()]
    dataset = [eval(data) for data in dataset]

    if "success" in file or "failed" in file:
        if index == -1:
            for data in dataset[:]:
                index, pos_state, goal, params = data
                params = dict(params)
                positions_dict = {key: list(pos) for key, pos in pos_state}
                state = State(positions=positions_dict)
                print(f"goal: {goal}")
                p.from_state(state)
                p.strike(**params)
                p.visualise()

        else:
            index, pos_state, goal, params = dataset[index]
            params = dict(params)
            positions_dict = {key: list(pos) for key, pos in pos_state}
            state = State(positions=positions_dict)
            print(f"goal: {goal}")
            p.from_state(state)
            p.strike(**params)
            p.visualise()

    elif "feasible" in file:
        if index == -1:
            for data in dataset[:]:
                pos_state, goal = data
                positions_dict = {key: list(pos) for key, pos in pos_state}
                state = State(positions=positions_dict)
                print(f"goal: {goal}")
                p.from_state(state)
                p.visualise()
        else:
            pos_state, goal = dataset[index]
            positions_dict = {key: list(pos) for key, pos in pos_state}
            state = State(positions=positions_dict)
            print(f"goal: {goal}")
            p.from_state(state)
            p.visualise()
    
visualise("feasible_dataset_new.txt", 4)








