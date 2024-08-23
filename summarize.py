import matplotlib.pyplot as plt
from collections import Counter

def get_rate_distribution():
    with open("success_rate_new.txt", "r") as f:
        dataset = [line.strip() for line in f.readlines()]
    success_rates = []

    for items in dataset:
        success_rate_str = items[-3:]
        success_rate = float(success_rate_str.split('=')[-1].strip())
        success_rates.append(success_rate)

    counter = Counter(success_rates)
    total_cnt = len(success_rates)

    per_distribution = {rate: (cnt / total_cnt) * 100 for rate, cnt in sorted(counter.items())}

    for rate, per in per_distribution.items():
        print(f'Success Rate: {rate}, Percentage: {per:.2f}%')


def get_shot_type():
    with open("feasible_dataset_new.txt", "r") as f:
        dataset = [line.strip() for line in f.readlines()]
    dataset = tuple([eval(data) for data in dataset])

    goals = {
        'bank_shot': [],
        'indirect_shot': [],
        'direct_shot': []
    }

    for line in dataset:
        state, goal = line

        if 'bank shot' in goal:
            goals['bank_shot'].append(goal)
        elif 'indirectly' in goal:
            goals['indirect_shot'].append(goal)
        else:
            goals['direct_shot'].append(goal)

    bank_shot_count = len(goals['bank_shot'])
    indirect_shot_count = len(goals['indirect_shot'])
    direct_shot_count = len(goals['direct_shot'])

    total_goals = bank_shot_count + indirect_shot_count + direct_shot_count

    bank_shot_percentage = (bank_shot_count / total_goals) * 100 if total_goals > 0 else 0
    indirect_shot_percentage = (indirect_shot_count / total_goals) * 100 if total_goals > 0 else 0
    direct_shot_percentage = (direct_shot_count / total_goals) * 100 if total_goals > 0 else 0

    print(f"Bank Shot Count: {bank_shot_count}, Percentage: {bank_shot_percentage:.2f}%")
    print(f"Indirect Shot Count: {indirect_shot_count}, Percentage: {indirect_shot_percentage:.2f}%")
    print(f"Direct Shot Count: {direct_shot_count}, Percentage: {direct_shot_percentage:.2f}%")


def get_zero_shot_type(success_rate_dataset):
    goals = {
        'bank_shot': [],
        'indirect_shot': [],
        'direct_shot': []
    }

    for line in success_rate_dataset:
        # 分离出 'Success Rate' 部分和前面的描述部分
        parts = line.rsplit('), Success Rate:', 1)
        description = parts[0]
        goal_description = description.split("), ")[-1]  
        success_rate = float(parts[1].split('=')[-1].strip())
        
        if success_rate == 0:
            if 'bank shot' in description:
                goals['bank_shot'].append(description)
            elif 'indirectly' in description:
                goals['indirect_shot'].append(description)
            else:
                goals['direct_shot'].append(description)

    bank_shot_count = len(goals['bank_shot'])
    indirect_shot_count = len(goals['indirect_shot'])
    direct_shot_count = len(goals['direct_shot'])

    total_goals = bank_shot_count + indirect_shot_count + direct_shot_count

    bank_shot_percentage = (bank_shot_count / total_goals) * 100 if total_goals > 0 else 0
    indirect_shot_percentage = (indirect_shot_count / total_goals) * 100 if total_goals > 0 else 0
    direct_shot_percentage = (direct_shot_count / total_goals) * 100 if total_goals > 0 else 0

    print(f"Bank Shot Count: {bank_shot_count}, Percentage: {bank_shot_percentage:.2f}%")
    print(f"Indirect Shot Count: {indirect_shot_count}, Percentage: {indirect_shot_percentage:.2f}%")
    print(f"Direct Shot Count: {direct_shot_count}, Percentage: {direct_shot_percentage:.2f}%")


def get_zero_rate(itr, stp):
    with open(f"sample_itr{itr}_stp{stp}.txt", "r") as f:
        success_rate_dataset = [line.strip() for line in f.readlines()]

    zero_rates_cnt = 0

    for items in success_rate_dataset:
        success_rate_str = items[-3:]
        success_rate = float(success_rate_str.split('=')[-1].strip())
        if success_rate == 0:
            zero_rates_cnt += 1

    return round(zero_rates_cnt / len(success_rate_dataset), 5)


def get_failure_rate(itr, stp):
    with open(f"sample_itr{itr}_stp{stp}.txt", "r") as f:
        dataset = [line.strip() for line in f.readlines()]
    failure_cnt = 0

    for items in dataset:
        if items.endswith("False"):
            failure_cnt += 1

    return round(failure_cnt / len(dataset), 5)

def show_proportion():
    percentage = [0.4671, 0.1193, 0.0723, 0.0720, 0.0484, 0.0451, 0.0500, 0.0419, 0.0306, 0.0236, 0.0296]
    success_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    plt.figure(figsize=(10, 5))
    plt.bar(success_rate, percentage, width=0.08, align='center')
    plt.xlabel('Success Rate')
    plt.ylabel('Percentage of Data Pairs')
    plt.title('Percentage of Different Success Rates in the Default Dataset')
    plt.xticks(success_rate)
    plt.savefig('Figure 10. Proportion of Success Rate.pdf')
    plt.show()

def show_trends_fixed_stps():
    percentage = [0.498, 0.432, 0.388, 0.38, 0.35]
    iter_times = [10, 20, 30, 40, 50]

    plt.figure(figsize=(10, 5))
    plt.plot(iter_times, percentage, marker='o')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Percentage of Zero Success Rate')
    plt.title('Percentage of Zero Success Rate over iter_times with 100 steps')
    plt.xticks(iter_times)

    plt.savefig('Figure 11. Change Iter_Times.pdf')
    plt.show()

def show_trends_fixed_itr():
    percentage = [0.498, 0.484, 0.4448, 0.4209, 0.3897]
    search_steps = [100, 150, 200, 250, 300]

    plt.figure(figsize=(10, 5))
    plt.plot(search_steps, percentage, marker='o')
    plt.xlabel('Search Steps')
    plt.ylabel('Percentage of Zero Success Rate')
    plt.title('Percentage of Zero Success Rate over search steps with 10 iterations')
    plt.xticks(search_steps)
    plt.savefig('Figure 12. Change Search_Steps.pdf')
    plt.show()

# print(get_zero_rate(10, 100))
# print(get_failure_rate(20, 100))
# show_proportion()
show_trends_fixed_stps()
# show_trends_fixed_itr()


