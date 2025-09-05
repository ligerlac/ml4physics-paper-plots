import argparse
import random
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from typing import List


from neurodifflogic.models.difflog_layers.linear import GroupSum, LogicLayer
from neurodifflogic.models.difflog_layers.conv import LogicConv2d, OrPoolingLayer, LogicConv3d


from utils import get_training_data, CreateFolder
from transforms import get_transform, get_log_threshold_transform

#from drawing import Draw


BITS_TO_TORCH_FLOATING_POINT_TYPE = {
    16: torch.float16,
    32: torch.float32,
    64: torch.float64
}

IMPL_TO_DEVICE = {
    'cuda': 'cuda',
    'python': 'cpu'
}


class ClampLayer(torch.nn.Module):
    def __init__(self, min_value=0.0, max_value=256.0):
        super(ClampLayer, self).__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        return torch.clamp(x, self.min_value, self.max_value)
    

def get_model(
    n_channels: int, n_kernels: int, stride: int, tree_depth: int, receptive_field_size: int, learning_rate=0.01, arch="conv2d"
):
    n_neurons_last_layer = 256 * 16
    tau = n_neurons_last_layer / 512
    beta = -n_neurons_last_layer / 2 + 32 * tau
    if arch == "conv2d":
        out_shape_conv = (
            (18 - receptive_field_size) // stride + 1,
            (14 - receptive_field_size) // stride + 1
        )
        model = torch.nn.Sequential(
            LogicConv2d(
                in_dim=(18, 14),
                stride=stride,
                num_kernels=n_kernels,
                channels=n_channels,
                tree_depth=tree_depth,
                receptive_field_size=receptive_field_size,
                padding=0,
                connections='random',
                implementation='python',
                device='cpu'
            ),
            torch.nn.Flatten(),
            LogicLayer(in_dim=out_shape_conv[0]*out_shape_conv[1] * n_kernels, out_dim=64*n_kernels,
                        connections='random', implementation='python', device='cpu'),
            LogicLayer(in_dim=64*n_kernels, out_dim=32*n_kernels,
                        connections='random', implementation='python', device='cpu'),
            LogicLayer(in_dim=32*n_kernels, out_dim=16*n_kernels,
                        connections='random', implementation='python', device='cpu'),
            LogicLayer(in_dim=16*n_kernels, out_dim=n_neurons_last_layer,
                        connections='random', implementation='python', device='cpu'),
            GroupSum(1, tau=tau, beta=beta),
        )
    elif arch == "conv3d":
        out_shape_conv = (
            (18 - 4) // stride + 1,
            (14 - 4) // stride + 1,
            (n_channels - 1) // stride + 1
        )
        print(f"out_shape_conv = {out_shape_conv}")
        model = torch.nn.Sequential(
            LogicConv3d(
                in_dim=(18, 14, n_channels),
                stride=stride,
                num_kernels=n_kernels,
                channels=1,
                tree_depth=tree_depth,
                receptive_field_size=(4, 4, 1),
                padding=0,
                connections='random',
                implementation='python',
                device='cpu'
            ),
            torch.nn.Flatten(),
            LogicLayer(in_dim=out_shape_conv[0]*out_shape_conv[1]*out_shape_conv[2] * n_kernels, out_dim=64*n_kernels,
                        connections='random', implementation='python', device='cpu'),
            LogicLayer(in_dim=64*n_kernels, out_dim=32*n_kernels,
                        connections='random', implementation='python', device='cpu'),
            LogicLayer(in_dim=32*n_kernels, out_dim=16*n_kernels,
                        connections='random', implementation='python', device='cpu'),
            LogicLayer(in_dim=16*n_kernels, out_dim=n_neurons_last_layer,
                        connections='random', implementation='python', device='cpu'),
            GroupSum(1, tau=tau, beta=beta),
        )

    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return model, loss_fn, optimizer


def train(model, x, y, loss_fn, optimizer, weights=None):
    x = model(x)
    loss = loss_fn(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train_with_regularization(model, x, y, loss_fn, optimizer, conv_entropy_lambda=0.0, ff_entropy_lambda=0.0):
    x = model(x)
    loss = loss_fn(x, y)

    conv_entropy, ff_entropy = 0.0, 0.0
    for layer in model.modules():
        if isinstance(layer, LogicConv2d):
            for level_weights in layer.tree_weights:
                for kernel_node_weights in level_weights:
                    probs = torch.softmax(kernel_node_weights, dim=1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                    conv_entropy += entropy.sum()
        
        elif isinstance(layer, LogicLayer):
            probs = torch.softmax(layer.weight, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            ff_entropy += entropy.sum()

    loss = loss + conv_entropy_lambda * conv_entropy + ff_entropy_lambda * ff_entropy
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), conv_entropy.item(), ff_entropy.item()


def predict_batch(model, x, batch_size=128, device='cuda'):
    with torch.no_grad():
        y_pred = []
        for i in range(0, x.size(0), batch_size):
            y_pred.append(model(x[i:i+batch_size].to(device)))
        y_pred = torch.cat(y_pred)
    return y_pred


def eval(model, x, y, loss_fn, mode, device='cuda'):
    orig_mode = model.training
    with torch.no_grad():
        model.train(mode=mode)
        # do in batches to avoid OOM
        y_pred = predict_batch(model, x, batch_size=128, device=device)
        res = loss_fn(y, y_pred)
        model.train(mode=orig_mode)
    return res.item()


def bits_eval(model, x, y, loss_fn, device='cuda'):
    x = x.bool().float()
    orig_mode = model.training
    with torch.no_grad():
        model.train(False)
        # do in batches to avoid OOM
        y_pred = predict_batch(model, x, batch_size=128, device=device)
        res = loss_fn(y, y_pred)
        model.train(mode=orig_mode)
    return res.item()


def main(args):
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    x_train, y_train, is_outlier_train, x_val, y_val, is_outlier_val, x_test, y_test, is_outlier_test = \
        get_training_data("saved_inputs_targets", backend="torch", zb_frac=4, use_outliers=True)
        # get_training_data("/scratch/network/lg0508/cicada-data/saved_inputs_targets", backend="torch", zb_frac=4, use_outliers=True)

    x_val = x_val[:1_000]
    y_val = y_val[:1_000]
    is_outlier_val = is_outlier_val[:1_000]
    
    x_test = x_test[:10_000]
    y_test = y_test[:10_000]
    is_outlier_test = is_outlier_test[:10_000]

    thresholds = [int(t) for t in args.thresholds]
    transform = get_log_threshold_transform(thresholds)
    x_train, x_val, x_test = transform(x_train.int()), transform(x_val.int()), transform(x_test.int())
    n_channels = x_train.shape[1]

    # reshape from (m, c, 18, 14) to (m, 1, 18, 14, c)
    if args.arch == "conv3d":
        x_train = x_train.permute(0, 2, 3, 1).unsqueeze(1)
        x_val = x_val.permute(0, 2, 3, 1).unsqueeze(1)
        x_test = x_test.permute(0, 2, 3, 1).unsqueeze(1)
        # n_channels = x_train.shape[4]

    print(f"x_train.shape = {x_train.shape}, y_train.shape = {y_train.shape}")

    model, loss_fn, optim = get_model(
        n_channels = n_channels,
        n_kernels = args.n_kernels,
        stride = args.stride,
        tree_depth = args.tree_depth,
        receptive_field_size = args.receptive_field_size,
        learning_rate = args.learning_rate,
        arch = args.arch
    )

    # print total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    model.to(args.device)

    print(f"x_val.shape = {x_val.shape}, y_val.shape = {y_val.shape}")

    x_val = x_val.to(args.device)
    y_val = y_val.to(args.device)

    print(model(x_val[:1]).shape)

    print(model(x_val[:20]))

    losses = defaultdict(list)
    conv_entropy_lambda, ff_entropy_lambda = 0.0, 0.0
    for i in tqdm(range(args.num_iterations)):
        beg = (i * args.batch_size) % x_train.shape[0]
        end = beg + args.batch_size
        x = x_train[beg:end]
        y = y_train[beg:end]

        x = x.to(BITS_TO_TORCH_FLOATING_POINT_TYPE[args.training_bit_count]).to(args.device)
        y = y.to(args.device)

        loss = train(model, x, y, loss_fn, optim)
        # loss, total_entropy, ff_entropy = train_with_regularization(
        #     model, x, y, loss_fn, optim, conv_entropy_lambda=conv_entropy_lambda, ff_entropy_lambda=ff_entropy_lambda
        # )

        if (i+1) % args.eval_freq == 0:
            losses['train_train_mode'].append(loss)
            losses['train_eval_mode'].append(eval(model, x, y, loss_fn, mode=False, device=args.device))
            losses['val_train_mode'].append(eval(model, x_val, y_val, loss_fn, mode=True, device=args.device))
            losses['val_eval_mode'].append(eval(model, x_val, y_val, loss_fn, mode=False, device=args.device))
            losses['train_bits_mode'].append(bits_eval(model, x, y, loss_fn, device=args.device))
            losses['val_bits_mode'].append(bits_eval(model, x_val, y_val, loss_fn, device=args.device))
            # losses['conv_entropy'].append(total_entropy)
            # losses['ff_entropy'].append(ff_entropy)

            print({k: round(v[-1], 3) for k, v in losses.items()})

        if (i+1) % args.save_freq == 0:
            df = pd.DataFrame(losses)
            df.to_csv(f"{args.output}/losses.csv", index=False)
            torch.save(model, f"{args.output}/model_iter{i+1}.pt")

        # after some epochs, add clamp layer
        if i == 3000:
            print(f"Adding clamp layer now")
            model = torch.nn.Sequential(
                model,
                ClampLayer(0, 256)
            ).to(args.device)

        # if (i+1) % 2000 == 0:
        #     conv_entropy_lambda += 0.001 * ((i + 1) // 1000)
        #     ff_entropy_lambda += 0.0001 * ((i + 1) // 1000)
        #     print(f"Increased entropy regularization to {conv_entropy_lambda} (conv), {ff_entropy_lambda} (ff)")


    torch.save(model, f"{args.output}/model.pt")

    y_pred = predict_batch(model, x_test, batch_size=128, device=args.device)
    y_pred = y_pred.cpu().detach().numpy().flatten()
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train logic gate network on the various datasets.')

    parser.add_argument('--thresholds', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('--n-kernels', type=int, default=64, help='Number of convolutional kernels')
    parser.add_argument('--stride', type=int, default=1, help='Stride of the convolutional layers')
    parser.add_argument('--tree-depth', type=int, default=3, help='Depth of the tree structure')
    parser.add_argument('--receptive-field-size', type=int, default=5, help='Size of the receptive field')

    parser.add_argument('--seed', '-s', type=int, default=0, help='seed (default: 0)')
    parser.add_argument('--batch-size', '-bs', type=int, default=128, help='batch size (default: 128)')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--training-bit-count', '-c', type=int, default=32, help='training bit count (default: 32)')

    parser.add_argument('--num-iterations', '-ni', type=int, default=100_000, help='Number of iterations (default: 100_000)')
    parser.add_argument('--eval-freq', '-ef', type=int, default=2_000, help='Evaluation frequency (default: 2_000)')
    parser.add_argument('--save-freq', '-sf', type=int, default=5_000, help='Save frequency (default: 5_000)')

    parser.add_argument('--output', type=str, default='data/models/latest', action=CreateFolder, help='path to save the trained model to')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training')
    parser.add_argument('--arch', type=str, default='conv2d', help='Architecture to use (conv2d or conv3d)')

    main(parser.parse_args())
