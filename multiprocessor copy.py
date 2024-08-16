import random
from multiprocessing import Pool, Lock, cpu_count, Manager, Queue
from script import evaluate_one

output_file = 'success_rate_new.txt'
iter_times = 10

def process_data(index, data, queue):
    print(f"Current index: {index}")
    success_times, success_rate = evaluate_one(data, iter_times)

    # 将结果放入队列中
    queue.put((index, data, success_times, success_rate))

    return (index, success_times, success_rate)


def listener(queue, output_file):
    with open(output_file, "a") as f:
        while True:
            m = queue.get()
            if m == 'kill':
                break
            index, data, success_times, success_rate = m
            f.write(f"{index, data}, Success Rate: {success_times} / {iter_times} = {success_rate}\n")
            f.flush()  # 确保数据立即写入磁盘


if __name__ == "__main__":
    with open("feasible_dataset_new.txt", "r") as f:
        dataset = [line.strip() for line in f.readlines()]
    dataset = tuple([eval(data) for data in dataset])


    # with open("new_run_indices.txt", "r") as f:
    #     idx = [int(line.strip()) for line in f.readlines()]

    idx = random.sample(range(len(dataset)), 5000)
    
    sampled_data = [(i, dataset[i]) for i in idx]


    num_processors = cpu_count()
    manager = Manager()
    queue = manager.Queue()

    # 启动监听进程来处理队列中的写操作
    pool = Pool(processes=num_processors + 1)
    watcher = pool.apply_async(listener, (queue, output_file))

    # 分发任务给进程池中的各个进程
    pool.starmap(process_data, [(i, data, queue) for i, data in sampled_data])

    # 在所有进程结束后，发送 'kill' 信号来终止监听进程
    queue.put('kill')
    pool.close()
    pool.join()


