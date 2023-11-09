import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

from utils import get_config_from_yml


def simplify_network_trace(trace_name, raw_dataset_dir, dataset_dir, save_pkl=True):
    """
    We simplify the network trace data.
    Take the 4G trace dataset as an example. Each line in each .log file is:
    Unix timestamp | cumulative time | geo_x | geo_y | data volume | elapsed time
    Each line is simplify as
    timestamp (start from zero) | data volume / 1s
    :param trace_name: file name of the network trace
    :param raw_dataset_dir: directory of the raw network dataset
    :param dataset_dir: directory of the simplified network dataset
    :param save_pkl: whether to save the new trace as .pkl file
    """
    trace_path = os.path.join(raw_dataset_dir, trace_name)
    new_trace_path = os.path.join(dataset_dir, trace_name)

    new_trace_data = []
    with open(trace_path, 'r', encoding='utf-8') as trace:
        for line in trace:
            line = line.strip().split()
            new_trace_data.append(int(line[-2]))
        trace.close()

    with open(new_trace_path, 'w', encoding='utf-8') as new_trace:
        for i in range(len(new_trace_data)):
            new_trace.write(f'{i} {new_trace_data[i]}\n')
        new_trace.close()
    print('Simplified trace (.log) saved at:', new_trace_path)

    if save_pkl:
        pickle.dump([(i, new_trace_data[i]) for i in range(len(new_trace_data))],
                    open(new_trace_path.replace('.log', '.pkl'), 'wb'))
        print('Simplified trace (.pkl) saved at:', new_trace_path.replace('.log', '.pkl'))


def simpify_network_dataset(dataset, config):
    """
    Simplify network dataset.
    :param dataset: dataset name
    :param config: configuration 
    """
    raw_network_dataset_dir = config.raw_network_datasets_dir[dataset]
    network_dataset_dir = config.network_datasets_dir[dataset]
    if not os.path.exists(network_dataset_dir):
        os.makedirs(network_dataset_dir)
    
    if dataset == '4G':
        for file in os.listdir(raw_network_dataset_dir):
            if file.endswith('.log'):
                simplify_network_trace(file, raw_network_dataset_dir, network_dataset_dir)


def scale_trace(dataset, trace_pkl, up, low, config):
    """
    Scale the trace
    :param trace_pkl: .pkl file of the trace
    :param up: upper bound throughputs
    :param low: lower bound throughputs
    """
    trace_path = os.path.join(config.network_datasets_dir[dataset], trace_pkl)
    trace = pickle.load(open(trace_path, 'rb'))
    throughputs = [trace[i][1] for i in range(len(trace))]
    max_, min_ = max(throughputs), min(throughputs)
    k = (up - low) / (max_ - min_)
    scaled_trace = [(trace[i][0], low + k * (throughputs[i] - min_)) for i in range(len(trace))]
    scaled_trace_path = os.path.join(config.network_datasets_dir[dataset], f'scaled_up_{up}_low_{low}' + trace_pkl)
    pickle.dump(scaled_trace, open(scaled_trace_path, 'wb'))
    print('Scaled trace (.pkl) saved at:', scaled_trace_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='4G')
    args = parser.parse_known_args()[0]

    config = get_config_from_yml()
    simpify_network_dataset(args.dataset, config)
