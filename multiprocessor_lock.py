import random
import multiprocessing
from multiprocessing import Pool, Lock
from script import evaluate_one

output_file = 'success_rate_new.txt'
lock = Lock()
iter_times = 10

def process_data(index, data):
    print(f"Current index: {index}")
    success_times, success_rate = evaluate_one(data, iter_times)

    with lock:
        with open(output_file, "a") as f:
             f.write(f"{index, data}, Success Rate: {success_times} / {iter_times} = {success_rate}\n")

    return (index, success_times, success_rate)


if __name__ == "__main__":
    with open("feasible_dataset_new.txt", "r") as f:
        dataset = [line.strip() for line in f.readlines()]
    dataset = tuple([eval(data) for data in dataset])
    
    # idx = random.sample(range(len(dataset)), 5000) # randomly sample 5,000

    with open("new_run_indices.txt", "r") as f:
        idx = [int(line.strip()) for line in f.readlines()]
    

    sampled_data = [(i, dataset[i]) for i in idx]

    num_processors = multiprocessing.cpu_count()
    with Pool(processes=num_processors) as pool:
        pool.starmap(process_data, sampled_data)

