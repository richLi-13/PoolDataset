from summarize import get_zero_rate


with open("sample_itr10_stp100.txt", "r") as f:
    dataset = [line.strip() for line in f.readlines()]



print(get_zero_rate(dataset))

