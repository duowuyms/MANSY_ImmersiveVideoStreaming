import numpy as np
from prettytable import PrettyTable
from collections import defaultdict


def read_log_file(log_path, print_pretty_table=True, append_all=True):
    data = defaultdict(list)
    with open(log_path, 'r') as file:
        file.readline()
        for line in file.readlines():
            line = line.strip().split(',')
            video, user, trace = list(map(int, line[:3]))
            qoe_w1, qoe_w2, qoe_w3, qoe, qoe1, qoe2, qoe3 = list(map(float, line[3:]))
            qoe_w1, qoe_w2, qoe_w3 = int(qoe_w1), int(qoe_w2), int(qoe_w3)
            data[(qoe_w1, qoe_w2, qoe_w3)].append([video, user, trace, qoe, qoe1, qoe2, qoe3])
            if append_all:
                data[(-1, -1, -1)].append([video, user, trace, qoe, qoe1, qoe2, qoe3])

    if print_pretty_table:
        for qoe_weight, values in data.items():
            print('On QoE Weight:', qoe_weight)
            pt = PrettyTable()
            pt.field_names = ['video', 'user', 'trace', 'qoe', 'qoe1', 'qoe2', 'qoe3']
            mean_qoe, mean_qoe1, mean_qoe2, mean_qoe3 = 0., 0., 0., 0.
            for value in values:
                pt.add_row(value)
                mean_qoe += value[3]
                mean_qoe1 += value[4]
                mean_qoe2 += value[5]
                mean_qoe3 += value[6]
            mean_qoe /= len(values)
            mean_qoe1 /= len(values)
            mean_qoe2 /= len(values)
            mean_qoe3 /= len(values)
            pt.add_row([-1, -1, -1, mean_qoe, mean_qoe1, mean_qoe2, mean_qoe3])
            print(pt)

    return data


def organize_data(data: dict, print_organized_data=True):
    organized_data = {}
    for qoe_weight, values in data.items():
        organized_data[qoe_weight] = {'qoe': [], 'qoe1': [], 'qoe2': [], 'qoe3': []}
        for value in values:
            organized_data[qoe_weight]['qoe'].append(value[3])
            organized_data[qoe_weight]['qoe1'].append(value[4])
            organized_data[qoe_weight]['qoe2'].append(value[5])
            organized_data[qoe_weight]['qoe3'].append(value[6])
        for key, value in organized_data[qoe_weight].items():
            organized_data[qoe_weight][key] = np.array(value)
    
    if print_organized_data:
        for qoe_weight, values in organized_data.items():
            print('On QoE Weight:', qoe_weight)
            for key, value in values.items():
                print(key, ':', value.tolist(), '\n')
    
    return organized_data


if __name__ == '__main__':
    srl_seen_qoe0_log_path_1 = '/data/wuduo/2023_omnidirectional_vs/results/bitrate_selection/rlva/Wu2017_4G/seen_qoe0/epochs_100_bs_256_lr_0.0001_gamma_0.99_seed_1_ent_0.1_results.csv'
    srl_seen_qoe0_data_1 = read_log_file(log_path=srl_seen_qoe0_log_path_1, append_all=False)
    srl_seen_qoe0_new_data_1 = organize_data(srl_seen_qoe0_data_1)
