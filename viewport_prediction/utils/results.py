import os
import numpy as np

from collections import namedtuple
from prettytable import PrettyTable
from utils.common import mean_square_error, find_tiles_covered_by_viewport


DEFAULT_VIDEO_WIDTH = 2560
DEFAULT_VIDEO_HEIGHT = 1440
DEFAULT_TILE_NUM_WIDTH = 8
DEFAULT_TILE_NUM_HEIGHT = 8
DEFAULT_TILE_COUNT = DEFAULT_TILE_NUM_HEIGHT * DEFAULT_TILE_NUM_WIDTH
DEFAULT_TILE_WIDTH = DEFAULT_VIDEO_WIDTH // DEFAULT_TILE_NUM_WIDTH
DEFAULT_TILE_HEIGHT = DEFAULT_VIDEO_HEIGHT // DEFAULT_TILE_NUM_HEIGHT
DEFAULT_FOV_WIDTH = 900  # 900
DEFAULT_FOV_HEIGHT = 900  # 900


Key = namedtuple('Key', 'video user timestamp')
Value = namedtuple('Value', 'k t gt pred mse accuracy prob recall precision f1')


def _compute_metrics(gt, pred, video_width, video_height, fov_width, fov_height,
                     tile_width, tile_height, tile_num_width, tile_num_height):
    gt_x, gt_y = int(gt[0] * video_width), int(gt[1] * video_height)
    gt_viewport = find_tiles_covered_by_viewport(gt_x, gt_y, video_width, video_height, fov_width, fov_height,
                                                 tile_width, tile_height, tile_num_width, tile_num_height)
    pred_x, pred_y = int(pred[0] * video_width), int(pred[1] * video_height)
    pred_viewport = find_tiles_covered_by_viewport(pred_x, pred_y, video_width, video_height, fov_width, fov_height,
                                                   tile_width, tile_height, tile_num_width, tile_num_height)
    accuracy = np.sum(gt_viewport & pred_viewport) / np.sum((gt_viewport | pred_viewport))
    tp = np.sum(gt_viewport & pred_viewport)
    fp = np.sum(pred_viewport) - tp
    fn = np.sum(gt_viewport) - tp
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    if recall + precision == 0:
        f1 = 0.
    else:
        f1 = recall * precision * 2 / (recall + precision)
    return accuracy, recall, precision, f1, gt_viewport, pred_viewport, None


def compute_accuracy(gt, pred, video_width=DEFAULT_VIDEO_WIDTH, video_height=DEFAULT_VIDEO_HEIGHT, 
                     fov_width=DEFAULT_FOV_WIDTH, fov_height=DEFAULT_FOV_HEIGHT,
                     tile_width=DEFAULT_TILE_WIDTH, tile_height=DEFAULT_TILE_HEIGHT, 
                     tile_num_width=DEFAULT_TILE_NUM_WIDTH, tile_num_height=DEFAULT_TILE_NUM_HEIGHT):
    accuracy_arr = np.zeros(pred.shape[:-1], np.float64)
    recall_arr = np.zeros(pred.shape[:-1], np.float64)
    precision_arr = np.zeros(pred.shape[:-1], np.float64)
    f1_arr = np.zeros(pred.shape[:-1], np.float64)
    for i in range(pred.shape[0]):
        for k in range(pred.shape[1]):
            for t in range(pred.shape[2]):
                accuracy_arr[i, k, t], recall_arr[i, k, t], precision_arr[i, k, t], f1_arr[
                    i, k, t], *_ = _compute_metrics(gt[i, k, t].numpy(), pred[i, k, t].numpy(), video_width, video_height,
                                                    fov_width, fov_height, tile_width, tile_height, tile_num_width,
                                                    tile_num_height)
    return accuracy_arr, recall_arr, precision_arr, f1_arr


class Results:
    def __init__(self, model_name, dimension, k, fut_window, output_dir, dataset_frequency, mse=True, nll=False, accuracy=False):
        self.dimension = dimension
        self.model_name = model_name
        self.k = k
        self.fut_window = fut_window
        self.output_dir = output_dir
        self.mse = mse
        self.nll = nll
        self.accuracy = accuracy
        self.results = []  # record the prediction results in detail
        self.dataset_frequency = dataset_frequency
        self.accuracy_results = [[[] for __ in range(fut_window)] for _ in range(k)]  # record the prediction accuracy of each prediction horizon

    def record(self, batch_size, prediction, ground_truth, video, user, timestamp):
        prediction = prediction.reshape(batch_size, self.k, self.fut_window, -1).cpu()
        ground_truth = ground_truth.reshape(batch_size, self.k, self.fut_window, -1).cpu()
        if self.mse:
            mse_arr = mean_square_error(prediction, ground_truth).cpu()
        if self.accuracy:
            accuracy_arr, recall_arr, precision_arr, f1_arr = compute_accuracy(ground_truth, prediction)

        for i in range(batch_size):
            key = Key(video=video[i], user=int(user[i]), timestamp=int(timestamp[i]))
            values = []
            for k in range(self.k):
                for t in range(self.fut_window):
                    mse = mse_arr[i][k][t].item() if self.mse else None
                    accuracy, prob, recall, precision, f1 = None, None, None, None, None
                    if self.accuracy:
                        accuracy = accuracy_arr[i, k, t]
                        recall = recall_arr[i, k, t]
                        precision = precision_arr[i, k, t]
                        f1 = f1_arr[i, k, t]
                        self.accuracy_results[k][t].append(accuracy)
                    value = Value(k=k, t=round((t + 1) * (1 / self.dataset_frequency), 3), gt=ground_truth[i][k][t].numpy(),
                                  pred=prediction[i][k][t].numpy(), mse=mse, accuracy=accuracy, prob=prob,
                                  recall=recall, precision=precision, f1=f1)
                    values.append(value)
            self.results.append({key: values})

    def write(self, log=True, label=''):
        csv_path = os.path.join(self.output_dir, label + 'results.csv')
        with open(csv_path, 'w', encoding='utf-8') as csv_file:
            head = 'video,user,timestamp,K,time,gt_1,gt_2,pred_1,pred_2,mse,accuracy,recall,precision,f1\n'
            csv_file.write(head)
            for element in self.results:
                for key, values in element.items():
                    for value in values:
                        line = f'{key.video},{key.user},{key.timestamp},'
                        line += f'{value.k},{value.t},{value.gt[0]},{value.gt[1]},'
                        for i in range(len(value.pred)):
                            line += f'{value.pred[i]},'
                        line += f'{value.mse},{value.accuracy},'
                        line += f'{value.recall},{value.precision},{value.f1}'
                        line += '\n'
                        csv_file.write(line)
            csv_file.close()
            print('Results saved at', csv_path)
        if log:
            log_path = os.path.join(self.output_dir, label + 'results.log')
            with open(log_path, 'w', encoding='utf-8') as f:
                for element in self.results:
                    for key, values in element.items():
                        f.write(f'##### Video={key[0]}, User={key[1]}, Timestamp={key[2]} #####\n')
                        for value in values:
                            line = f'K={value[0]}, time={value[1]}, gt={list(value[2])}, pred={list(value[3])}, '
                            line += f'mse={value[4]}, accuracy={value[5]}, '
                            line += f'recall={value[6]}, precision={value[7]}, f1={value[8]}'
                            line += '\n'
                            f.write(line)
            f.close()
            print('Log saved at', log_path)
        if self.accuracy:
            accuracy_csv_path = os.path.join(self.output_dir, label + 'accuracy_result.csv')
            mean_accuracy = [[] for _ in range(self.k)]
            with open(accuracy_csv_path, 'w', encoding='utf-8') as csv_file:
                head = 'K,timestamp,accuracy\n'
                csv_file.write(head)
                for k in range(self.k):
                    for i in range(self.fut_window):
                        mean_accuracy[k].append(sum(self.accuracy_results[k][i]) / len(self.accuracy_results[k][i]) * 100.)
                        # mean_accuracy[k][i] = sum(mean_accuracy[k]) / len(mean_accuracy[k])  
                        line = f'{k},{round((i + 1) * (1 / self.dataset_frequency), 3)},{mean_accuracy[k][i]}\n'
                        csv_file.write(line)
                csv_file.close()


            # print results to terminal
            print('Pretty Table...')
            pt = PrettyTable()
            pt.field_names = ['K', *[round((i + 1) * (1 / self.dataset_frequency), 3) for i in range(self.fut_window)]]
            for k in range(self.k):
                pt.add_row([k, *[round(ma, 5) for ma in mean_accuracy[k]]])
            print(pt)
            print('Copy-Friendly Table...')
            for k in range(self.k):
                print(f'{k}:', f'[{",".join([str(round(ma, 5)) for ma in mean_accuracy[k]])}]')

            # average over all timestep
            print('############################ average over all timestep #############################')
            for k in range(self.k):
                tmp = []
                for i in range(self.fut_window):
                    tmp.append(sum(mean_accuracy[k][:i + 1]) / len(mean_accuracy[k][:i + 1]))
                mean_accuracy[k] = tmp
            print('Pretty Table...')
            pt = PrettyTable()
            pt.field_names = ['K', *[round((i + 1) * (1 / self.dataset_frequency), 3) for i in range(self.fut_window)]]
            for k in range(self.k):
                pt.add_row([k, *[round(ma, 5) for ma in mean_accuracy[k]]])
            print(pt)
            print('Copy-Friendly Table...')
            for k in range(self.k):
                print(f'{k}:', f'[{",".join([str(round(ma, 5)) for ma in mean_accuracy[k]])}]')

    def reset(self):
        self.results.clear()
        self.accuracy_results = [[[] for __ in range(self.fut_window)] for _ in range(self.k)]
