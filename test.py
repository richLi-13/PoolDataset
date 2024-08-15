
index = 1
data = 2
success_times = 3
iter_times = 4
success_rate = success_times / iter_times

with open("success_rate_new.txt", "a") as f:
    f.write(f"{index, data}, {success_times} / {iter_times} = {success_rate}\n")