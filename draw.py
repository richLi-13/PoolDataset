
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from check_feasibility import get_pocket_pos, w, l


def draw(data):
    state, goal = data

    plt.figure()

    x0, y0 = 0, 0
    width, height = w, l
    rect = patches.Rectangle((x0, y0), width, height, linewidth=1, edgecolor='black', facecolor='none')
    plt.gca().add_patch(rect)

    for pocket, pos in get_pocket_pos().items():
        plt.scatter(pos[0], pos[1], color = 'black')

    for ball in state:
        plt.scatter(ball[1][0], ball[1][1], edgecolor = 'black', color = 'white' if ball[0] == 'cue' else ball[0])


    plt.xlabel('X position')
    plt.ylabel('Y position')

    plt.gca().set_aspect('equal')

    plt.title(goal)
    plt.show()


def drawall(file, n=-1, m=-1):
    with open(file, "r") as f:
        dataset = [lines.strip() for lines in f.readlines()]
    dataset = tuple([eval(data) for data in dataset])

    if n == -1:
        for data in dataset:
            draw(data)
    else:
        if m == -1:
            for data in dataset[:n]:
                draw(data)
        else:
            for data in dataset[m:n]:
                draw(data)


# drawall("feasible_dataset_new.txt", 4, 3)
