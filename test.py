import numpy as np
import random

with open("sample_indices_500.txt", "r") as f:
    idx = [int(line.strip()) for line in f.readlines()]

all_numbers = set(range(13074))
new_list = list(all_numbers - set(idx))
new_idx = random.sample(new_list, 500) # randomly sample 5,000

with open("sample_indices_1000.txt", "w") as f:
    for i in idx:
        f.write(f"{i}\n")

    for i in new_idx:
        f.write(f"{i}\n")

