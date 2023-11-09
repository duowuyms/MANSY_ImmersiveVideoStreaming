import os
import argparse
import random
import sys
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from models import LinearRegression, ViewportTransformerMTIO
from utils.common import get_config_from_yml, mean_square_error
from utils.load_dataset import create_dataset
from utils.results import Results
from utils.console_logger import ConsoleLogger


def train(args, model, dataloader_train, dataloader_valid, models_dir, file_prefix):
    checkpoint_path = os.path.join(models_dir, file_prefix + f'_checkpoint.pth')
    best_model_path = os.path.join(models_dir, file_prefix + f'_best_model.pth')

    if args.resume:
        assert args.resume_path is not None
        model.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print('Resume model for training from:', args.resume_path)

    train_size = len(dataloader_train)
    valid_size = len(dataloader_valid)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    best_valid_mse, best_epoch = float('inf'), 0
    print(f'Training {args.model} on {args.train_dataset} - bs: {args.bs} - lr: {args.lr} - seed: {args.seed}')
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}\n-------------------------------")
        # training
        model.train()
        total_train_loss = []
        for batch, (history, current, future, video, user, timestep) in enumerate(dataloader_train):
            history, current, future = history.to(args.device), current.to(args.device), future.to(args.device)
            pred, gt = model(history, current, future)
            train_loss = model.loss_function(pred, gt)
            total_train_loss.append(train_loss)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            print(f"\rTrain: [{batch + 1}/{train_size}] - train_loss: {train_loss:>9f}", end='')
        print(f'\rTrain: mean train loss: {(sum(total_train_loss) / len(total_train_loss)):>9f}')
        if epoch % args.epochs_per_valid == 0:
            # validation
            model.eval()
            with torch.no_grad():
                mse = []
                for history, current, future, video, user, timestep in dataloader_valid:
                    history, current, future = history.to(args.device), current.to(args.device), future.to(args.device)
                    pred, gt = model.sample(history, current), future
                    if args.model == 'dvms':
                        gt = torch.repeat_interleave(gt, 5, dim=0)
                    mse.append(torch.mean(mean_square_error(pred, gt)).item())
                mse = np.sum(mse) / valid_size
                print(f'Valid: mean square error: {mse:>9f}')

                torch.save(model.state_dict(), checkpoint_path)
                print(f'Checkpoint saved at', checkpoint_path)
                if best_valid_mse > mse:
                    best_valid_mse = mse
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), best_model_path)
                print(f'Best model (epoch {best_epoch}, loss {best_valid_mse}) saved at', best_model_path)


def test(args, model, dataloader_test_seen, dataloader_test_unseen, models_dir, results_dir, file_prefix):
    best_model_path = os.path.join(models_dir, file_prefix + f'_best_model.pth')
    notebook = Results(args.model, dimension=2, fut_window=args.fut_window,
                       dataset_frequency=args.dataset_frequency, output_dir=results_dir, mse=True, accuracy=True)

    if args.model != 'regression':  # linear regression doesn't need loading model weights
        model.load_state_dict(torch.load(best_model_path, map_location=args.device))
        print('Load model from', best_model_path)

    print(f'Testing {args.model} on {args.test_dataset} - seed: {args.seed}')
    with torch.no_grad():
        model.eval()
        print('On seen viewing patterns.')
        for batch, (history, current, future, video, user, timesteps) in enumerate(tqdm(dataloader_test_seen)):
            history, current, future = history.to(args.device), current.to(args.device), future.to(args.device)
            batch_size = history.shape[0]
            pred, gt = model.sample(history, current), future
            notebook.record(batch_size, pred, gt, video, user, timesteps)
        notebook.write(log=True, label=file_prefix + '_seen_')
        notebook.reset()
        print('On unseen viewing patterns.')
        for batch, (history, current, future, video, user, timesteps) in enumerate(tqdm(dataloader_test_unseen)):
            history, current, future = history.to(args.device), current.to(args.device), future.to(args.device)
            batch_size = history.shape[0]
            pred, gt = model.sample(history, current), future
            notebook.record(batch_size, pred, gt, video, user, timesteps)
        notebook.write(log=True, label=file_prefix + '_unseen_')
        notebook.reset()

def create_model(model_name, fut_window, hidden_dim, block_num, device, seed):
    model = None
    if model_name == 'regression':
        model = LinearRegression(fut_window=fut_window, device=device)
    elif model_name == 'mtio':
        model = ViewportTransformerMTIO(in_channel=2, fut_window=fut_window, d_model=hidden_dim, dim_feedforward=hidden_dim,
                                        num_encoder_layers=block_num, num_decoder_layers=block_num, device=device, seed=seed)
    return model


def run(args, config):
    assert args.model in ['regression', 'mtio']

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    models_dir = os.path.join(config.vp_models_dir, args.model, args.train_dataset, f'{args.dataset_frequency}Hz')
    results_dir = os.path.join(config.vp_results_dir, args.model, args.test_dataset, f'{args.dataset_frequency}Hz')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    file_prefix = f'his_{args.his_window}_fut_{args.fut_window}_hid_{args.hidden_dim}_ss_{args.sample_step}_'\
                  f'epochs_{args.epochs}_bs_{args.bs}_lr_{args.lr}_seed_{args.seed}'
    model = create_model(args.model, args.fut_window, args.hidden_dim, args.block_num, args.device, args.seed).to(args.device)
    
    if args.compile:
        assert torch.__version__ >= '2.0.0', 'Compile model requires torch version >= 2.0.0, but current torch version is ' + torch.__version__
        print("\033[33mWarning:\033[0m There seems to be some bugs in torch.compile. If batch size is too large, it will raise errors (I don't know why this happens).")
        model = torch.compile(model).to(args.device)  # recommend to compile model when you are using PyTorch 2.0
    
    torch.set_float32_matmul_precision('high')

    if args.train:
        console_log = open(os.path.join(results_dir, file_prefix + 'console.log'), 'w')
        sys.stdout = ConsoleLogger(sys.__stdout__, console_log)
        dataset_train, dataset_valid = create_dataset(args.train_dataset, config, his_window=args.his_window, fut_window=args.fut_window,
                                                      frequency=args.dataset_frequency, sample_step=args.sample_step, trim_head=args.trim_head, 
                                                      trim_tail=args.trim_tail, include=['train', 'valid'])
        dataloader_train = DataLoader(dataset_train, batch_size=args.bs, shuffle=True, pin_memory=True)
        dataloader_valid = DataLoader(dataset_valid, batch_size=args.bs, shuffle=False, pin_memory=True)
        train(args, model, dataloader_train, dataloader_valid, models_dir, file_prefix)
    if args.test:
        dataset_test_seen, dataset_test_unseen = create_dataset(args.test_dataset, config, his_window=args.his_window, fut_window=args.fut_window, sample_step=args.sample_step,
                                                                frequency=args.dataset_frequency, trim_head=args.trim_head, trim_tail=args.trim_tail, include=['test_seen', 'test_unseen'])
        dataloader_test_seen = DataLoader(dataset_test_seen, batch_size=args.bs, shuffle=False, pin_memory=True)
        dataloader_test_unseen = DataLoader(dataset_test_unseen, batch_size=args.bs, shuffle=False, pin_memory=True)
        test(args, model, dataloader_test_seen, dataloader_test_unseen, models_dir, results_dir, file_prefix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process the input parameters to train the network.')

    # ========== model settings related arguments ==========
    parser.add_argument('--train', action="store_true", help='Train model.')
    parser.add_argument('--test', action="store_true", help='Test model.')
    parser.add_argument('--device', action='store', dest='device', help='Device (cuda or cpu) to run experiment.')
    parser.add_argument('--model', action='store', dest='model', help='Model type')
    parser.add_argument('--hidden-dim', type=int, help='Number of hidden dimensions', default=512)
    parser.add_argument('--block-num', type=int, help='Number of encoder/decoder blocks', default=2)
    parser.add_argument('--compile', action='store_true', dest='compile', 
                        help='(Optional) Compile model for speed up (available only for PyTorch 2.0).')
    parser.add_argument('--resume', action='store_true', dest='resume',
                        help='(Optional) Resume model weights from checkpoint for training.')
    parser.add_argument('--resume-path', type=str, dest='resume_path',
                        help='(Optional) Resume model weights from checkpoint for training.')
    
    # ========== dataset loading/processing settings related arguments ==========
    parser.add_argument('--train-dataset', action='store', dest='train_dataset', help='Dataset for training.')
    parser.add_argument('--test-dataset', action='store', dest='test_dataset', help='Dataset for testing.')
    parser.add_argument('--his-window', action='store', dest='his_window', help='Historical window', type=int, default=5)
    parser.add_argument('--fut-window', action='store', dest='fut_window', help='Future (prediction) window.', type=int, default=15)
    parser.add_argument('--trim-head', action='store', dest='trim_head',
                        help='(Optional) Trim some part of the viewport trajectory head (default 30).', type=int)
    parser.add_argument('--trim-tail', action='store', dest='trim_tail',
                        help='(Optional) Trim some part of the viewport trajectory tail (default 10).', type=int)
    parser.add_argument('--dataset-frequency', action='store', dest='dataset_frequency',
                        help='(Optional) The frequency version of the dataset (default 5).', type=int)
    parser.add_argument('--sample-step', action='store', dest='sample_step',
                        help='(Optional) The steps for sampling viewports (default 5).', type=int)
    
    # ========== training related settings ==========
    parser.add_argument('--epochs', action="store", dest='epochs', help='Neural network learning epochs.', type=int, default=200)
    parser.add_argument('--epochs-per-valid', action='store', dest='epochs_per_valid', type=int, default=3,
                        help='(Optional) The number of epochs per validation (default 3).')
    parser.add_argument('--lr', action="store", dest='lr', help='Neural network learning rate.', type=float, default=1e-4)
    parser.add_argument('--weight-decay', action="store", dest='weight_decay', help='(Optional) Neural network weight decay.', type=float)
    parser.add_argument('--bs', action="store", dest='bs', help='Neural network batch size.', type=int)
    parser.add_argument('--seed', action="store", dest='seed', type=int, default=5, help='(Optional) Random seed.')
    args = parser.parse_args()

    # command example
    # python run_models.py --model mtio --train --test --train-dataset Jin2022 --test-dataset Jin2022 --his-window 5 --fut-window 15 --bs 512 --seed 5 --dataset-frequency 5 --sample-step 5 --hidden-dim 512 --block-num 2 --lr 0.0001 --epochs 200 --epochs-per-valid 3 --device cuda:0

    # handle defautl settings
    config = get_config_from_yml()
    args.trim_head = config.trim_head if args.trim_head is None else args.trim_head
    args.trim_tail = config.trim_tail if args.trim_tail is None else args.trim_tail
    args.dataset_frequency = config.frequency if args.dataset_frequency is None else args.dataset_frequency
    args.sample_step = config.sample_step if args.sample_step is None else args.sample_step

    if args.model == 'regression':
        args.train = False
        args.compile = False
        args.device = 'cpu'
        print('Detect model: regression. Automatically disenable train and compile mode and set device to cpu.')
        
    print(args)
    run(args, config)
