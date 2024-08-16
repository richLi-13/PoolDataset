import random
from summarize import get_zero_rate


idx = []


with open("sample_itr100_stp100.txt", "r") as f:
    dataset = [line.strip() for line in f.readlines()]

for data in dataset:
    index = int(data.split(',')[0].strip('('))
    idx.append(index)

with open("sample_indices.txt", "w") as f:
    for index in idx:
        f.write(f"{index}\n")



print(get_zero_rate(dataset))

