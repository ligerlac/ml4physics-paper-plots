import argparse
import random
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd

from neurodifflogic.models.difflog_layers.linear import GroupSum, LogicLayer
from neurodifflogic.models.difflog_layers.conv import LogicConv2d, OrPoolingLayer
from neurodifflogic.models import CNN

from utils import get_training_data, CreateFolder
from transforms import get_transform

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
    

def get_model(n_channels: int, n_kernels: int, stride: int, tree_depth: int, receptive_field_size: int, learning_rate=0.01):
    n_neurons_last_layer = 256 * 16
    tau = n_neurons_last_layer / 512
    beta = -n_neurons_last_layer / 2 + 32 * tau
    out_shape_conv = (
        (18 - receptive_field_size) // stride + 1,
        (14 - receptive_field_size) // stride + 1
    )
    print(f"out_shape_conv = {out_shape_conv}")
    print(f"flattened: {out_shape_conv[0]*out_shape_conv[1]}")
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

    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return model, loss_fn, optimizer


def train(model, x, y, loss_fn, optimizer, weights=None):
    # print("TRAINING STEP")
    # print(f"x.shape = {x.shape}, y.shape = {y.shape}")
    x = model(x)
    # print(f"pred.shape = {x.shape},pred[:5] =\n{x[:5]}")
    loss = loss_fn(x, y)
    # print(f"loss.shape = {loss.shape}, loss = {loss}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train_with_regularization(model, x, y, loss_fn, optimizer, conv_entropy_lambda=0.0, ff_entropy_lambda=0.0):
    # print("TRAINING STEP")
    # print(f"x.shape = {x.shape}, y.shape = {y.shape}")
    x = model(x)
    # print(f"pred.shape = {x.shape},pred[:5] =\n{x[:5]}")
    loss = loss_fn(x, y)

    # if entropy_lambda > 0.0:
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

    # print(f"total_entropy = {total_entropy}, entropy_lambda = {entropy_lambda}")
    loss = loss + conv_entropy_lambda * conv_entropy + ff_entropy_lambda * ff_entropy
    # print(f"loss.shape = {loss.shape}, loss = {loss}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # return loss.item(), total_entropy.item() if entropy_lambda > 0.0 else 0.0
    return loss.item(), conv_entropy.item(), ff_entropy.item()


def predict_batch(model, x, batch_size=128, device='cuda'):
    # print(f"predict_batch. input shape = {x.shape}")
    with torch.no_grad():
        y_pred = []
        # for i in tqdm(range(0, x.size(0), batch_size)):
        for i in range(0, x.size(0), batch_size):
            y_pred.append(model(x[i:i+batch_size].to(device)))
        y_pred = torch.cat(y_pred)
        # print(f"pred step: y_pred[:3] = {y_pred[:3]}")
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


def packbits_eval(model, x, y, device='cuda'):
    orig_mode = model.training
    with torch.no_grad():
        model.eval()
        res = np.mean(
            [
                (model(PackBitsTensor(x.to(device).reshape(x.shape[0], -1).round().bool())).argmax(-1) == y.to(
                    device)).to(torch.float32).mean().item()
                for x, y in loader
            ]
        )
        model.train(mode=orig_mode)
    return res.item()


def main(args):
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # x_train, y_train, is_outlier_train, x_val, y_val, is_outlier_val, x_test, y_test, is_outlier_test = \
    #     get_training_data("saved_inputs_targets", backend="torch", zb_frac=-1, use_outliers=False)
    x_train, y_train, is_outlier_train, x_val, y_val, is_outlier_val, x_test, y_test, is_outlier_test = \
        get_training_data("/scratch/network/lg0508/cicada-data/saved_inputs_targets", backend="torch", zb_frac=4, use_outliers=True)
    # x_train, y_train, is_outlier_train, x_val, y_val, is_outlier_val, x_test, y_test, is_outlier_test = \
    #     get_training_data("saved_inputs_targets", backend="torch", zb_frac=0, use_outliers=True)

    # zero-pad to get shape (18, 14) -> (18, 18)
    # x_train = F.pad(x_train, (0, 4), "constant", 0)
    # x_val = F.pad(x_val, (0, 4), "constant", 0)
    # x_test = F.pad(x_test, (0, 4), "constant", 0)
    
    x_val = x_val[:1_000]
    y_val = y_val[:1_000]
    is_outlier_val = is_outlier_val[:1_000]
    
    x_test = x_test[:10_000]
    y_test = y_test[:10_000]
    is_outlier_test = is_outlier_test[:10_000]

    if args.transform_name != "identity":
        transform = get_transform(args.transform_name)
        x_train, x_val, x_test = transform(x_train.int()), transform(x_val.int()), transform(x_test.int())

    print(f"x_train.shape = {x_train.shape}, y_train.shape = {y_train.shape}")

    plt.hist(x_train.detach().cpu().numpy().flatten(), bins=100)
    plt.show()

    # model, loss_fn, optim = get_model(args, in_dim=(N_BITS, 18, 18))
    model, loss_fn, optim = get_modelski(args.model_name, channels=4, learning_rate=args.learning_rate)

    # print total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    # device = "cpu"
    # model = model.to(device)
    # x_val = x_val.to(device)
    # model.train(True)
    # print(f"CPU (train): {model(x_val[:3])}")

    # model.train(False)
    # print(f"CPU (eval): {model(x_val[:3])}")

    # # Switch to GPU
    # device = "cuda"
    # model = model.to(device)
    # x_val = x_val.to(device)
    # model.train(True)
    # print(f"GPU (train): {model(x_val[:3])}")

    # model.train(False)
    # print(f"GPU (eval): {model(x_val[:3])}")

    # exit(0)

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

        # loss = train(model, x, y, loss_fn, optim)
        loss, total_entropy, ff_entropy = train_with_regularization(
            model, x, y, loss_fn, optim, conv_entropy_lambda=conv_entropy_lambda, ff_entropy_lambda=ff_entropy_lambda
        )

        if (i+1) % args.eval_freq == 0:
            losses['train_train_mode'].append(loss)
            losses['conv_entropy'].append(total_entropy)
            losses['ff_entropy'].append(ff_entropy)
            losses['train_eval_mode'].append(eval(model, x, y, loss_fn, mode=False, device=args.device))
            losses['val_train_mode'].append(eval(model, x_val, y_val, loss_fn, mode=True, device=args.device))
            losses['val_eval_mode'].append(eval(model, x_val, y_val, loss_fn, mode=False, device=args.device))
            if args.packbits_eval:
                losses['train_bits_mode'].append(packbits_eval(model, x, y, device=args.device))
                losses['valid_bits_mode'].append(packbits_eval(model, x_val, y_val, device=args.device))

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

    plt.hist(y_pred, bins=100)
    plt.show()

    np.save(f"data/predictions/{args.model_name}_x_test.npy", y_pred)

    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train logic gate network on the various datasets.')

    parser.add_argument('--tau', '-t', type=float, default=1, help='the softmax temperature tau (as multiple of last layer neurons)')
    parser.add_argument('--seed', '-s', type=int, default=0, help='seed (default: 0)')
    parser.add_argument('--batch-size', '-bs', type=int, default=128, help='batch size (default: 128)')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--training-bit-count', '-c', type=int, default=32, help='training bit count (default: 16)')

    parser.add_argument('--implementation', type=str, default='python', choices=['cuda', 'python'],
                        help='`cuda` is the fast CUDA implementation and `python` is simpler but much slower '
                        'implementation intended for helping with the understanding.')

    parser.add_argument('--packbits_eval', action='store_true', help='Use the PackBitsTensor implementation for an '
                                                                     'additional eval step.')
    parser.add_argument('--compile_model', action='store_true', help='Compile the final model with C for CPU.')

    parser.add_argument('--num-iterations', '-ni', type=int, default=100_000, help='Number of iterations (default: 100_000)')
    parser.add_argument('--eval-freq', '-ef', type=int, default=2_000, help='Evaluation frequency (default: 2_000)')
    parser.add_argument('--save-freq', '-sf', type=int, default=5_000, help='Save frequency (default: 5_000)')

    parser.add_argument('--valid-set-size', '-vss', type=float, default=0., help='Fraction of the train set used for validation (default: 0.)')
    parser.add_argument('--extensive-eval', action='store_true', help='Additional evaluation (incl. valid set eval).')

    parser.add_argument('--connections', type=str, default='unique', choices=['random', 'unique'])
    parser.add_argument('--architecture', '-a', type=str, default='randomly_connected')
    parser.add_argument('--num_neurons', '-k', type=int, default=8000)
    parser.add_argument('--num_layers', '-l', type=int, default=6)
    parser.add_argument('--beta', '-b', type=float, default=0, help='offset in groupsum')
    parser.add_argument('--interactive', action='store_true', help='Interactively display plots as they are created', default=False)

    parser.add_argument('--model-name', type=str, default='clgn-zb-only-4qb', help='Name of the model')
    parser.add_argument('--output', type=str, default='data/models/latest', action=CreateFolder, help='path to save the trained model to')
    parser.add_argument('--transform-name', type=str, default='identity', help='Name of the bit transform to use')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training')

    main(parser.parse_args())
