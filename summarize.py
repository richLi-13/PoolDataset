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


def get_zero_rate(success_rate_dataset):

    zero_rates_cnt = 0

    for items in success_rate_dataset:
        success_rate_str = items[-3:]
        success_rate = float(success_rate_str.split('=')[-1].strip())
        if success_rate == 0:
            zero_rates_cnt += 1

    return round(zero_rates_cnt / len(success_rate_dataset), 5)



