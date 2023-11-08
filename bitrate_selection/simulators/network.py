import math
import pickle


class NetworkTrace:
    """
    A class to simulate network environment with a given network trace file
    """
    def __init__(self, trace_path, scale=None):
        self.trace = pickle.load(open(trace_path, 'rb'))
        if scale is not None:
            throughputs = [self.trace[i][1] for i in range(len(self.trace))]
            max_, min_ = max(throughputs), min(throughputs)
            up, low = scale[0], scale[1]
            k = (up - low) / (max_ - min_)
            scaled_trace = [(self.trace[i][0], low + k * (throughputs[i] - min_)) for i in range(len(self.trace))]
            self.trace = scaled_trace
        self.trace_len = len(self.trace)
        self.cur_time = 0.
        self.cur_idx = 0

    def simulate_download(self, size):
        start_time = self.cur_time
        while size > 0:
            # remain throughput in the current time segment [timestamp, timestamp + 1]
            remain_throughput = (math.floor(self.cur_time + 1) - self.cur_time) * self.trace[self.cur_idx][1]
            if size >= remain_throughput:
                self.cur_idx = (self.cur_idx + 1) % self.trace_len
                self.cur_time = math.floor(self.cur_time + 1)
                size -= remain_throughput
            else:
                self.cur_time += size / self.trace[self.cur_idx][1]
                size = 0
        download_time = self.cur_time - start_time
        return download_time

    def get_current_time(self):
        return self.cur_idx, self.cur_time

    def set_current_time(self, cur_idx, cur_time):
        self.cur_idx = cur_idx
        self.cur_time = cur_time

    def reset(self):
        self.cur_time = 0.
        self.cur_idx = 0


def _test_NetworkTrace():
    network_trace = NetworkTrace('0')
    # for i in range(10):
    #     print('(%d, %d)' % (network_trace.trace[i][0], network_trace.trace[i][1]), end='   ')
    # print()
    # print('download time %.3f\n' % network_trace.simulate_download_from_time(5000000))
    # print('download time %.3f\n' % network_trace.simulate_download_from_time(1000000))
    # print('download time %.3f\n' % network_trace.simulate_download_from_time(500000))
    # print('download time %.3f\n' % network_trace.simulate_download_from_time(6666666))
    '''
    Ground truth:
    download time 2.131
    download time 0.310
    download time 0.155
    download time 1.822
    '''
    network_trace.trace = [(0, 50), (1, 100), (2, 80), (3, 200), (4, 150), (6, 100)]  # for testing
    print('download time %.3f\n' % network_trace.simulate_download_from_time(70))
    print('download time %.3f\n' % network_trace.simulate_download_from_time(100))
    print('download time %.3f\n' % network_trace.simulate_download_from_time(30))
    print('download time %.3f\n' % network_trace.simulate_download_from_time(80))
    '''
    Ground truth:
    download time 1.200
    download time 1.050
    download time 0.375
    download time 0.625
    '''


if __name__ == '__main__':
    _test_NetworkTrace()
