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
from utils import get_training_data, CreateFolder
from transforms import get_transform

from drawing import Draw


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
    

def get_modelski(name: str, channels: int, learning_rate=0.01):
    llkw = {
        'connections': 'random',
        'implementation': 'python',
        # 'device': 'cpu'
    }
    if name == "clgn-single-channel":
        n_kernels = 4
        n_neurons_last_layer = 256 * 8
        tau = n_neurons_last_layer / 1024
        beta = -n_neurons_last_layer / 2 + 32 * tau
        stride = 2
        model = torch.nn.Sequential(
            LogicConv2d(
                in_dim=(18, 14),
                stride=stride,
                num_kernels=n_kernels,
                channels=1,
                tree_depth=6,
                receptive_field_size=4,
                padding=0,
                connections='random',
                implementation='python',
                device='cpu'
            ),
            torch.nn.Flatten(),
            LogicLayer(in_dim=int((18-4+1)*(14-4+1)*n_kernels / stride**2), out_dim=1000,
                       grad_factor=1.0, connections='random', implementation='python', device='cpu'),
            LogicLayer(in_dim=1000, out_dim=1000,
                       grad_factor=1.0, connections='random', implementation='python', device='cpu'),
            LogicLayer(in_dim=1000, out_dim=n_neurons_last_layer,
                       grad_factor=1.0, connections='random', implementation='python', device='cpu'),
            GroupSum(1, tau=tau, beta=beta),
        )
    elif name == "clgn-3ch":
        n_kernels_1 = 4
        n_neurons_last_layer = 256 * 8
        tau = n_neurons_last_layer / 1024
        beta = -n_neurons_last_layer / 2 + 32 * tau
        model = torch.nn.Sequential(
            LogicConv2d(
                in_dim=(18, 14),
                stride=1,
                num_kernels=n_kernels_1,
                channels=3,
                tree_depth=6,
                receptive_field_size=4,
                padding=0,
                # connections="random",
                grad_factor=1.0,
                connections='random',
                implementation='python',
                device='cpu'
            ),
            torch.nn.Flatten(),
            LogicLayer(in_dim=(18-4+1)*(14-4+1)*n_kernels_1, out_dim=1000,
                       grad_factor=1.0, connections='random', implementation='python', device='cpu'),
            LogicLayer(in_dim=1000, out_dim=1000,
                       grad_factor=1.0, connections='random', implementation='python', device='cpu'),
            LogicLayer(in_dim=1000, out_dim=n_neurons_last_layer,
                       grad_factor=1.0, connections='random', implementation='python', device='cpu'),
            GroupSum(1, tau=tau, beta=beta),
        )
    elif name == "clgn-zb-only-3qb":
        n_kernels_1 = 4
        n_neurons_last_layer = 256 * 8
        tau = n_neurons_last_layer / 1024
        beta = -n_neurons_last_layer / 2 + 32 * tau
        model = torch.nn.Sequential(
            LogicConv2d(
                in_dim=(18, 14),
                stride=1,
                num_kernels=n_kernels_1,
                channels=3,
                tree_depth=6,
                receptive_field_size=4,
                padding=0,
                # connections="random",
                grad_factor=1.0,
                connections='random',
                implementation='python',
                device='cpu'
            ),
            torch.nn.Flatten(),
            LogicLayer(in_dim=(18-4+1)*(14-4+1)*n_kernels_1, out_dim=1000,
                       grad_factor=1.0, connections='random', implementation='python', device='cpu'),
            LogicLayer(in_dim=1000, out_dim=1000,
                       grad_factor=1.0, connections='random', implementation='python', device='cpu'),
            LogicLayer(in_dim=1000, out_dim=n_neurons_last_layer,
                       grad_factor=1.0, connections='random', implementation='python', device='cpu'),
            GroupSum(1, tau=tau, beta=beta),
        )
    elif name == "clgn-zb-only-4qb":
        n_kernels_1 = 4
        n_neurons_last_layer = 256 * 8
        tau = n_neurons_last_layer / 1024
        beta = -n_neurons_last_layer / 2 + 32 * tau
        model = torch.nn.Sequential(
            LogicConv2d(
                in_dim=(18, 14),
                stride=1,
                num_kernels=n_kernels_1,
                channels=channels,
                tree_depth=6,
                receptive_field_size=4,
                padding=0,
                # connections="random",
                grad_factor=1.0,
                connections='random',
                implementation='python',
                device='cpu'
            ),
            torch.nn.Flatten(),
            LogicLayer(in_dim=(18-4+1)*(14-4+1)*n_kernels_1, out_dim=1000,
                       grad_factor=1.0, connections='random', implementation='python', device='cpu'),
            LogicLayer(in_dim=1000, out_dim=1000,
                       grad_factor=1.0, connections='random', implementation='python', device='cpu'),
            LogicLayer(in_dim=1000, out_dim=n_neurons_last_layer,
                       grad_factor=1.0, connections='random', implementation='python', device='cpu'),
            GroupSum(1, tau=tau, beta=beta),
        )
    elif name == "clgn-5qb":
        n_kernels_1 = 4
        n_neurons_last_layer = 256 * 8
        tau = n_neurons_last_layer / 1024
        beta = -n_neurons_last_layer / 2 + 32 * tau
        model = torch.nn.Sequential(
            LogicConv2d(
                in_dim=(18, 14),
                stride=1,
                num_kernels=n_kernels_1,
                channels=channels,
                tree_depth=7,
                receptive_field_size=4,
                padding=0,
                connections="random",
            ),
            torch.nn.Flatten(),
            LogicLayer(in_dim=(18-4+1)*(14-4+1)*n_kernels_1, out_dim=1000, **llkw),
            LogicLayer(in_dim=1000, out_dim=1000, **llkw),
            LogicLayer(in_dim=1000, out_dim=1000, **llkw),
            LogicLayer(in_dim=1000, out_dim=n_neurons_last_layer, **llkw),
            GroupSum(1, tau=tau, beta=beta)
        )

    elif name == "lgn-single-channel":
        n_neurons_last_layer = 256 * 8
        tau = n_neurons_last_layer/1024
        beta = -n_neurons_last_layer/2 + 32 * tau
        in_dim = 18 * 14

        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            LogicLayer(in_dim=in_dim, out_dim=4000, **llkw),
            LogicLayer(in_dim=4000, out_dim=2000, **llkw),
            LogicLayer(in_dim=2000, out_dim=1000, **llkw),
            # LogicLayer(in_dim=1000, out_dim=1000, **llkw),
            LogicLayer(in_dim=1000, out_dim=n_neurons_last_layer, **llkw),
            GroupSum(1, tau=tau, beta=beta),
            # ClampLayer(0, 256)
        )

    elif name == "lgn-lt2":
        n_neurons_last_layer = 256 * 8
        tau = n_neurons_last_layer/1024
        beta = -n_neurons_last_layer/2 + 32 * tau
        in_dim = 2 * 18 * 14

        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            LogicLayer(in_dim=in_dim, out_dim=4000, **llkw),
            LogicLayer(in_dim=4000, out_dim=2000, **llkw),
            LogicLayer(in_dim=2000, out_dim=1000, **llkw),
            # LogicLayer(in_dim=1000, out_dim=1000, **llkw),
            LogicLayer(in_dim=1000, out_dim=n_neurons_last_layer, **llkw),
            GroupSum(1, tau=tau, beta=beta),
            # ClampLayer(0, 256)
        )

    elif name == "lgn-3ch":
        n_neurons_last_layer = 256 * 8
        tau = n_neurons_last_layer/1024
        beta = -n_neurons_last_layer/2 + 32 * tau
        in_dim = 3 * 18 * 14

        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            LogicLayer(in_dim=in_dim, out_dim=5000, **llkw),
            LogicLayer(in_dim=5000, out_dim=2000, **llkw),
            LogicLayer(in_dim=2000, out_dim=1000, **llkw),
            # LogicLayer(in_dim=1000, out_dim=1000, **llkw),
            LogicLayer(in_dim=1000, out_dim=n_neurons_last_layer, **llkw),
            GroupSum(1, tau=tau, beta=beta),
            # ClampLayer(0, 256)
        )

    elif name == "lgn-4ch":
        n_neurons_last_layer = 256 * 8
        tau = n_neurons_last_layer/1024
        beta = -n_neurons_last_layer/2 + 32 * tau
        in_dim = 4 * 18 * 14

        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            LogicLayer(in_dim=in_dim, out_dim=6000, **llkw),
            LogicLayer(in_dim=6000, out_dim=2000, **llkw),
            LogicLayer(in_dim=2000, out_dim=1000, **llkw),
            # LogicLayer(in_dim=1000, out_dim=1000, **llkw),
            LogicLayer(in_dim=1000, out_dim=n_neurons_last_layer, **llkw),
            GroupSum(1, tau=tau, beta=beta),
            # ClampLayer(0, 256)
        )

    elif name == "lgn-4qb":
        n_neurons_last_layer = 256 * 8
        # tau has to be adjusted to the desired spread of values (let's say a spread of 1024)
        # we really only need a spread of 256, but that makes training difficult as ALL neurons
        # have to be False (True) to reach the edges
        tau = n_neurons_last_layer/1024
        # adjust beta to center the values around the target mean of ~32
        beta = -n_neurons_last_layer/2 + 32 * tau
        in_dim = channels * 18 * 14

        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            LogicLayer(in_dim=in_dim, out_dim=4000, **llkw),
            LogicLayer(in_dim=4000, out_dim=2000, **llkw),
            LogicLayer(in_dim=2000, out_dim=1000, **llkw),
            # LogicLayer(in_dim=1000, out_dim=1000, **llkw),
            LogicLayer(in_dim=1000, out_dim=n_neurons_last_layer, **llkw),
            GroupSum(1, tau=tau, beta=beta),
            # ClampLayer(0, 256)
        )
    else:
        raise ValueError(f"Unknown model name: {name}")

    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return model, loss_fn, optimizer



# def get_model(args, in_dim=(1, 18, 14)):
#     """
#     252 input neurons means 252*(252-1)/2 = 31626 possible unique connections (=max size for next layer)

#     """

#     llkw = {
#         'grad_factor': args.grad_factor,
#         'connections': args.connections,
#         # 'connections': 'unique',
#         'implementation': args.implementation,
#         'device': IMPL_TO_DEVICE[args.implementation]
#     }

#     # quantized precision of cicada anomaly score is 256* 256.
#     # Will calculate the mean of output nodes for regression
#     # single class (1-d regression). Choose beta and tau to adjust range.
#     # Number of neurons in second-to-last layer determines precision.
#     # beta (offset) not yet implemented in difflogic
#     class_count = 1

#     logic_layers = []

#     arch = args.architecture
#     k = args.num_neurons
#     l = args.num_layers
#     t = args.tau

#     ####################################################################################################################

#     print(f'llkw = {llkw}')
#     llkw['connections'] = 'random'

#     # group sum does (sum + beta) / tau
#     # we adjust beta and tau such that pre-training mean matches target mean (~32)
#     # this determnines the precision (e.g. 256 means 256 distinct values)
#     n_neurons_last_layer = 256 * 8
#     # tau has to be adjusted to the desired spread of values (let's say a spread of 1024)
#     # we really only need a spread of 256, but that makes training difficult as ALL neurons
#     # have to be False (True) to reach the edges
#     tau = n_neurons_last_layer/1024
#     # adjust beta to center the values around the target mean of ~32
#     beta = -n_neurons_last_layer/2 + 32 * tau

#     n_kernels_1 = 4
#     n_kernels_2 = 16
#     channels = in_dim[0]

#     # in the receptive field, there will be channels * receptive_field_size * receptive_field_size inputs
#     # 4 x 4 x 4 = 64

#     model = torch.nn.Sequential(
#         LogicConv2d(
#             in_dim=(18, 14),
#             stride=1,
#             num_kernels=n_kernels_1,
#             channels=channels,
#             tree_depth=6,  # 128 connections from receptive field
#             receptive_field_size=4,
#             padding=0,
#             # connections="random",
#             **llkw
#         ),
#         # OrPoolingLayer(kernel_size=2, stride=2, padding=0),
#         # LogicConv2d(
#         #     in_dim=(7, 7),
#         #     stride=1,
#         #     num_kernels=n_kernels_2,
#         #     channels=channels,
#         #     tree_depth=4,
#         #     receptive_field_size=3,
#         #     padding=0,
#         #     # connections="random",
#         #     **llkw
#         # ),
#         torch.nn.Flatten(),
#         # LogicLayer(in_dim=(7-3+1)**2*n_kernels_2, out_dim=1000, **llkw),
#         # LogicLayer(in_dim=(18-4+1)**2*n_kernels_1, out_dim=1000, **llkw),
#         LogicLayer(in_dim=(18-4+1)*(14-4+1)*n_kernels_1, out_dim=1000, **llkw),
#         LogicLayer(in_dim=1000, out_dim=1000, **llkw),
#         # LogicLayer(in_dim=1000, out_dim=1000, **llkw),
#         LogicLayer(in_dim=1000, out_dim=n_neurons_last_layer, **llkw),
#         GroupSum(class_count, tau=tau, beta=beta),
#         # ClampLayer(0, 256)
#     )

#     ####################################################################################################################

#     total_num_neurons = sum(map(lambda x: x.num_neurons, logic_layers[1:-1]))
#     print(f'total_num_neurons={total_num_neurons}')
#     total_num_weights = sum(map(lambda x: x.num_weights, logic_layers[1:-1]))
#     print(f'total_num_weights={total_num_weights}')

#     # model = model.to(llkw['device'])

#     # print(model)

#     # loss_fn = torch.nn.CrossEntropyLoss()
#     # loss_fn = torch.nn.MSELoss()
#     loss_fn = torch.nn.L1Loss()

#     optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

#     return model, loss_fn, optimizer


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


def predict_batch(model, x, batch_size=128, device='cuda'):
    # print(f"predict_batch. input shape = {x.shape}")
    with torch.no_grad():
        y_pred = []
        # for i in tqdm(range(0, x.size(0), batch_size)):
        for i in range(0, x.size(0), batch_size):
            y_pred.append(model(x[i:i+batch_size].to(device)))
        y_pred = torch.cat(y_pred)
        print(f"pred step: y_pred[:3] = {y_pred[:3]}")
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
    
    draw = Draw('plots', interactive=args.interactive)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # x_train, y_train, is_outlier_train, x_val, y_val, is_outlier_val, x_test, y_test, is_outlier_test = \
    #     get_training_data("saved_inputs_targets", backend="torch", zb_frac=-1, use_outliers=False)
    x_train, y_train, is_outlier_train, x_val, y_val, is_outlier_val, x_test, y_test, is_outlier_test = \
        get_training_data("saved_inputs_targets", backend="torch", zb_frac=4, use_outliers=True)
    # x_train, y_train, is_outlier_train, x_val, y_val, is_outlier_val, x_test, y_test, is_outlier_test = \
    #     get_training_data("saved_inputs_targets", backend="torch", zb_frac=0, use_outliers=True)
    
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

    model.to(args.device)

    # print(f"model = {model}")
    # for layer in model:
    #     print(f"layer = {layer}, #neurons = {getattr(layer, 'num_neurons', None)}, #weights = {getattr(layer, 'num_weights', None)}")
    #     if layer.__class__.__name__ == 'LogicLayer':
    #         print(f"layer.weight = {layer.weight}")
    #         argmaxes = layer.weight.argmax(dim=1)
    #         plt.hist(argmaxes.detach().cpu().numpy().flatten(), bins=100)
    #         plt.show()
    #     # print(f"  layer parameters = {[p.shape for p in layer.parameters()]}")

    # exit(0)

    # preds = model(x_val.to(device))
    # plt.hist(preds.detach().numpy().flatten(), bins=100)
    # plt.show()

    x_val = x_val.to(args.device)
    y_val = y_val.to(args.device)

    print(model(x_val[:20]))

    losses = defaultdict(list)
    for i in tqdm(range(args.num_iterations)):
        beg = (i * args.batch_size) % x_train.shape[0]
        end = beg + args.batch_size
        x = x_train[beg:end]
        y = y_train[beg:end]

        x = x.to(BITS_TO_TORCH_FLOATING_POINT_TYPE[args.training_bit_count]).to(args.device)
        y = y.to(args.device)

        loss = train(model, x, y, loss_fn, optim)

        if (i+1) % args.eval_freq == 0:
            losses['train_train_mode'].append(loss)
            losses['train_eval_mode'].append(eval(model, x, y, loss_fn, mode=False, device=args.device))
            losses['val_train_mode'].append(eval(model, x_val, y_val, loss_fn, mode=True, device=args.device))
            losses['val_eval_mode'].append(eval(model, x_val, y_val, loss_fn, mode=False, device=args.device))
            if args.packbits_eval:
                losses['train_bits_mode'].append(packbits_eval(model, x, y, device=args.device))
                losses['valid_bits_mode'].append(packbits_eval(model, x_val, y_val, device=args.device))

            print({k: v[-1] for k, v in losses.items()})

        if (i+1) % args.save_freq == 0:
            df = pd.DataFrame(losses)
            df.to_csv(f"{args.output}/losses.csv", index=False)
            torch.save(model, f"{args.output}/model_iter{i+1}.pt")

        # after some epochs, add clamp layer
        if i == 5000:
            print(f"Adding clamp layer now")
            model = torch.nn.Sequential(
                model,
                ClampLayer(0, 256)
            ).to(args.device)

    torch.save(model, f"{args.output}/model.pt")

    y_pred = predict_batch(model, x_test, batch_size=128, device=args.device)
    y_pred = y_pred.cpu().detach().numpy().flatten()

    plt.hist(y_pred, bins=100)
    plt.show()

    np.save(f"data/predictions/{args.model_name}_x_test.npy", y_pred)

    ####################################################################################
    ###################################   PLOTTING   ###################################
    ####################################################################################
    
    draw.make_scatter_density_plot(y_test.cpu().detach().numpy().flatten(), y_pred, 'Inc', 'Target', 'Difflogic')
    if (is_outlier_test==0).sum() > 0:
        draw.make_scatter_density_plot(y_test[~is_outlier_test].cpu().detach().numpy().flatten(), y_pred[~is_outlier_test], 'ZB', 'Target', 'Difflogic')
    if is_outlier_test.sum() > 0:
        draw.make_scatter_density_plot(y_test[is_outlier_test].cpu().detach().numpy().flatten(), y_pred[is_outlier_test], 'TT', 'Target', 'Difflogic')

    draw.make_scatter_plot(
        teacher_scores=[y_test[~is_outlier_test].cpu().detach().numpy().flatten(), y_test[is_outlier_test].cpu().detach().numpy().flatten()],
        student_scores=[y_pred[~is_outlier_test], y_pred[is_outlier_test]],
        labels=['Difflogic ZB', 'Difflogic TT'],
    )

    draw.plot_anomaly_score_distribution(
        [y_test[~is_outlier_test].cpu().detach().numpy(), y_pred[~is_outlier_test], y_test[is_outlier_test].cpu().detach().numpy(), y_pred[is_outlier_test]],
        ['Target ZB', 'Difflogic ZB', 'Target TT', 'Difflogic TT'],
        "anomaly-scores",
    )

    y_diff_float = model(x_test).detach().numpy()
    y_diff_bin = model(x_test.bool().int()).detach().numpy()

    model.train(False)
    y_bin_float = model(x_test).numpy()
    y_bin_bin = model(x_test.bool().int()).numpy()
    model.train(True)

    draw.plot_anomaly_score_distribution(
        [y_test.detach().numpy(), y_diff_float, y_diff_bin, y_bin_float, y_bin_bin],
        ['Target', 'Diff(float)', 'Diff(bin)', 'Bin(float)', 'Bin(bin)'],
        "anomaly-scores-modes-inputs",
    )

    exit(0)

    #####################################################################################
    ###################################   COMPILING   ###################################
    #####################################################################################

    model.train(False)

    print("Debug info", flush=True)

    compiled_model = CompiledLogicNet(
        model=model,            # the trained model (should be a `torch.nn.Sequential` with `LogicLayer`s)
        num_bits=8,            # the number of bits of the datatype used for inference (typically 64 is fastest, should not be larger than batch size)
        # num_bits=8,            # the number of bits of the datatype used for inference (typically 64 is fastest, should not be larger than batch size)
        cpu_compiler='gcc',     # the compiler to use for the c code (alternative: clang)
        # cpu_compiler='clang',     # the compiler to use for the c code (alternative: clang)
        verbose=True
    )
    compiled_model.compile(
        save_lib_path='my_model_binary.so',  # the (optional) location for storing the binary such that it can be reused
        verbose=True
    )

    print("Testing compiled model:")
    x_val_test = x_val[:51,:,:,:]
    print("x_val_test.shape = ", x_val_test.shape)
    print(f"model(x_val_test) =\n{model(x_val_test)}")
    print(f"compiled_model(x_val_test) =\n{compiled_model(x_val_test.bool().numpy(), verbose=True)}")

    compiled_model_results = compiled_model(x_val.bool().numpy(), verbose=True)
    with open("y_val_lgn_ref.txt", "w") as f:
        for val in compiled_model_results:
            f.write(f"{val.item()}\n")


    model_c_code = compiled_model.get_c_code()
    with open("cicada_model_opendata_highoutlier.c", "w") as f:
        f.write(model_c_code)


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
