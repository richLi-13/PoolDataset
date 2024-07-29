
# This script generates a goal dataset for the pool game.

balls = ["yellow", "blue", "red", "purple", "orange", "green", "maroon", "black"]
pockets = ["top_left", "top_right", "middle_left", "middle_right", "bottom_left", "bottom_right"]

def gen_goal_dataset(balls, pockets):
    goal_dataset = [] 
    for X in balls:
        for Y in pockets:
            goal_dataset.append("pot " + str(X) + " ball into " + Y + " pocket")

    for X in balls:
        for Y in pockets:
            goal_dataset.append("pot " + str(X) + " ball into " + Y + " pocket directly")

    for X in balls:
        for Y in pockets:
            goal_dataset.append("pot " + str(X) + " ball into " + Y + " pocket with a bank shot")

    with open("goal_dataset.txt", "w") as f:
        for goal in goal_dataset:
            f.write(f"{goal}\n")
    return goal_dataset
    

gen_goal_dataset(balls, pockets)