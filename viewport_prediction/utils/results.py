import os
import numpy as np

from collections import namedtuple
from prettytable import PrettyTable
from utils.common import get_config_from_yml, mean_square_error, find_tiles_covered_by_viewport


Key = namedtuple('Key', 'video user timestamp')
Value = namedtuple('Value', 't gt pred mse accuracy prob recall precision f1')


def _compute_metrics(gt, pred, video_width, video_height, tile_width, tile_height, 
                     tile_num_width, tile_num_height, fov_width, fov_height):
    gt_x, gt_y = int(gt[0] * video_width), int(gt[1] * video_height)
    gt_viewport = find_tiles_covered_by_viewport(gt_x, gt_y, video_width, video_height, tile_width, tile_height, 
                                                 tile_num_width, tile_num_height, fov_width, fov_height)
    pred_x, pred_y = int(pred[0] * video_width), int(pred[1] * video_height)
    pred_viewport = find_tiles_covered_by_viewport(pred_x, pred_y, video_width, video_height, tile_width, tile_height,
                                                   tile_num_width, tile_num_height, fov_width, fov_height)
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


def compute_accuracy(gt, pred, video_width, video_height, tile_num_width, tile_num_height,
                     tile_width=None, tile_height=None, fov_width=600, fov_height=300):
    if tile_width is None:
        tile_width = video_width // tile_num_width
    if tile_height is None:
        tile_height = video_height // tile_num_height
    accuracy_arr = np.zeros(pred.shape[:-1], np.float64)
    recall_arr = np.zeros(pred.shape[:-1], np.float64)
    precision_arr = np.zeros(pred.shape[:-1], np.float64)
    f1_arr = np.zeros(pred.shape[:-1], np.float64)
    for i in range(pred.shape[0]):
        for t in range(pred.shape[1]):
            accuracy_arr[i, t], recall_arr[i, t], precision_arr[i, t], f1_arr[i, t], *_ = _compute_metrics(
                gt[i, t].numpy(), pred[i, t].numpy(), video_width, video_height, tile_width, 
                tile_height, tile_num_width, tile_num_height, fov_width, fov_height
            )
    return accuracy_arr, recall_arr, precision_arr, f1_arr


class Results:
    def __init__(self, model_name, dimension, fut_window, output_dir, dataset_frequency, mse=True, nll=False, accuracy=False):
        self.config = get_config_from_yml()
        self.model_name = model_name
        self.dimension = dimension
        self.fut_window = fut_window
        self.output_dir = output_dir
        self.mse = mse
        self.nll = nll
        self.accuracy = accuracy
        self.results = []  # record the prediction results in detail
        self.dataset_frequency = dataset_frequency
        self.accuracy_results = [[] for _ in range(fut_window)]  # record the prediction accuracy of each prediction horizon

    def record(self, batch_size, prediction, ground_truth, video, user, timestamp):
        prediction = prediction.cpu()
        ground_truth = ground_truth.cpu()
        if self.mse:
            mse_arr = mean_square_error(prediction, ground_truth).cpu()
        if self.accuracy:
            accuracy_arr, recall_arr, precision_arr, f1_arr = compute_accuracy(ground_truth, prediction, self.config.video_width,
                                                                               self.config.video_height, self.config.tile_num_width,
                                                                               self.config.tile_num_width)

        for i in range(batch_size):
            key = Key(video=video[i], user=int(user[i]), timestamp=int(timestamp[i]))
            values = []
            for t in range(self.fut_window):
                mse = mse_arr[i, t].item() if self.mse else None
                accuracy, prob, recall, precision, f1 = None, None, None, None, None
                if self.accuracy:
                    accuracy = accuracy_arr[i, t]
                    recall = recall_arr[i, t]
                    precision = precision_arr[i, t]
                    f1 = f1_arr[i, t]
                    self.accuracy_results[t].append(accuracy)
                value = Value(t=round((t + 1) * (1 / self.dataset_frequency), 3), gt=ground_truth[i][t].numpy(),
                              pred=prediction[i][t].numpy(), mse=mse, accuracy=accuracy, prob=prob,
                              recall=recall, precision=precision, f1=f1)
                values.append(value)
            self.results.append({key: values})

    def write(self, log=True, label=''):
        csv_path = os.path.join(self.output_dir, label + 'results.csv')
        with open(csv_path, 'w', encoding='utf-8') as csv_file:
            head = 'video,user,timestamp,time,gt_1,gt_2,pred_1,pred_2,mse,accuracy,recall,precision,f1\n'
            csv_file.write(head)
            for element in self.results:
                for key, values in element.items():
                    for value in values:
                        line = f'{key.video},{key.user},{key.timestamp},'
                        line += f'{value.t},{value.gt[0]},{value.gt[1]},'
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
                            line = f'time={value[0]}, gt={list(value[1])}, pred={list(value[2])}, '
                            line += f'mse={value[3]}, accuracy={value[5]}, '
                            line += f'recall={value[5]}, precision={value[6]}, f1={value[7]}'
                            line += '\n'
                            f.write(line)
            f.close()
            print('Log saved at', log_path)
        if self.accuracy:
            accuracy_csv_path = os.path.join(self.output_dir, label + 'accuracy_result.csv')
            mean_accuracy = []
            with open(accuracy_csv_path, 'w', encoding='utf-8') as csv_file:
                head = 'timestamp,accuracy\n'
                csv_file.write(head)
                for i in range(self.fut_window):
                    mean_accuracy.append(sum(self.accuracy_results[i]) / len(self.accuracy_results[i]) * 100.)
                    line = f'{round((i + 1) * (1 / self.dataset_frequency), 3)},{mean_accuracy[i]}\n'
                    csv_file.write(line)
                csv_file.close()

            print('Pretty Table...')
            # average over all timestep
            tmp = []
            for i in range(self.fut_window):
                tmp.append(sum(mean_accuracy[:i + 1]) / len(mean_accuracy[:i + 1]))
            mean_accuracy = tmp
            pt = PrettyTable()
            pt.field_names = [*[round((i + 1) * (1 / self.dataset_frequency), 3) for i in range(self.fut_window)]]
            pt.add_row([*[round(ma, 5) for ma in mean_accuracy]])
            print(pt)

    def reset(self):
        self.results.clear()
        self.accuracy_results = [[] for _ in range(self.fut_window)]
