import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import h5py
from utils import get_training_data, get_input_data
from transforms import get_transform
from drawing import Draw


class ClampLayer(torch.nn.Module):
    def __init__(self, min_value=0.0, max_value=256.0):
        super(ClampLayer, self).__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        return torch.clamp(x, self.min_value, self.max_value)
    

class OrderedLearnableThermometer(nn.Module):
    def __init__(self, init_thresholds, slope=10.0):
        super().__init__()
        self.num_thresholds = len(init_thresholds)
        self.slope = slope
        self._frozen = False  # switch to control hard/soft behavior

        init_t = torch.tensor(init_thresholds, dtype=torch.float32)
        first = init_t[:1]
        diffs = torch.diff(init_t, prepend=first.new_zeros(1))
        self.raw_diffs = nn.Parameter(diffs)

    def get_thresholds(self):
        if self._frozen:
            return torch.cumsum(self.raw_diffs, dim=0)
        else:
            diffs_pos = F.softplus(self.raw_diffs)
            return torch.cumsum(diffs_pos, dim=0)

    def freeze_thresholds(self):
        with torch.no_grad():
            thresholds = self.get_thresholds().round()
            first = thresholds[:1]
            diffs = torch.diff(thresholds, prepend=first.new_zeros(1))
            self.raw_diffs.copy_(diffs)
        self.raw_diffs.requires_grad = False
        self._frozen = True

    def forward(self, x):
        thresholds = self.get_thresholds()  # (T,)
        if x.ndim == 3:  # (B, H, W)
            x = x.unsqueeze(1)  # -> (B, 1, H, W)
        thresholds = thresholds.view(1, -1, 1, 1)

        if self._frozen:
            # Hard thermometer encoding
            outputs = (x > thresholds).float()
        else:
            # Soft, differentiable approximation
            outputs = torch.tanh(self.slope * (x - thresholds))
            outputs = (outputs + 1.0) / 2.0
        return outputs


def predict_batch(model, x, batch_size=128, device='cuda'):
    print(f"x.shape = {x.shape}")
    with torch.no_grad():
        y_pred = []
        for i in tqdm(range(0, x.size(0), batch_size)):
            y_pred.append(model(x[i:i+batch_size].to(device)))
        y_pred = torch.cat(y_pred)
    return y_pred


def main(args):
    x_train, y_train, is_outlier_train, x_val, y_val, is_outlier_val, x_test, y_test, is_outlier_test = \
        get_training_data("saved_inputs_targets", backend="torch", zb_frac=-1, use_outliers=False)
    # x_train, y_train, is_outlier_train, x_val, y_val, is_outlier_val, x_test, y_test, is_outlier_test = \
    #     get_training_data("saved_inputs_targets", backend="torch", zb_frac=1, use_outliers=True)
    
    x_test = x_test[:100_000]
    y_test = y_test[:100_000]

    # x_tt_had = None
    with h5py.File("data/inputs/tt_hadronic_with_l1evtnPV.h5", "r") as f:
        x_tt_had = torch.from_numpy(f["CaloRegions"][:])

    with h5py.File("data/inputs/tt_semileptonic_with_l1evtnPV.h5", "r") as f:
        x_tt_semilep = torch.from_numpy(f["CaloRegions"][:])

    with h5py.File("data/inputs/qcd_with_l1evtnPV.h5", "r") as f:
        x_qcd = torch.from_numpy(f["CaloRegions"][:])

    with h5py.File("data/inputs/singleneutrino_with_l1evtnPV.h5", "r") as f:
        x_sn = torch.from_numpy(f["CaloRegions"][:])

    model = torch.load(args.model_path, map_location="cpu", weights_only=False)
    model.to(args.device)

    model.train(False)

    if args.transform_name != "identity":
        trafo = get_transform(args.transform_name)
        x_train = trafo(x_train.int()).float()
        x_test = trafo(x_test.int()).float()
        x_tt_had = trafo(x_tt_had.int()).float()
        x_tt_semilep = trafo(x_tt_semilep.int()).float()
        x_qcd = trafo(x_qcd.int()).float()
        x_sn = trafo(x_sn.int()).float()

    # x_test = x_test.bool().int().float().to(args.device)
    # x_tt_had = x_tt_had.bool().int().float().to(args.device)
    # x_tt_semilep = x_tt_semilep.bool().int().float().to(args.device)
    # x_qcd = x_qcd.bool().int().float().to(args.device)
    # x_sn = x_sn.bool().int().float().to(args.device)

    print(f"{x_test.shape=}")
    if args.conv_3d:
        x_train = x_train.permute(0, 3, 1, 2).unsqueeze(1)
        x_test = x_test.permute(0, 2, 3, 1).unsqueeze(1)
        x_tt_had = x_tt_had.permute(0, 2, 3, 1).unsqueeze(1)
        x_tt_semilep = x_tt_semilep.permute(0, 2, 3, 1).unsqueeze(1)
        # x_qcd = x_qcd.permute(0, 2, 3, 1).unsqueeze(1)
        # x_sn = x_sn.permute(0, 2, 3, 1).unsqueeze(1)

    print(f"{x_test.shape=}")
    print(f"x_tt_had.shape = {x_tt_had.shape}")
    print(f"x_tt_semilep.shape = {x_tt_semilep.shape}")

    preds_zb = predict_batch(model, x_test, batch_size=256, device=args.device).cpu().numpy().flatten()

    # draw = Draw('plots', interactive=True)
    # draw.make_scatter_density_plot(y_test.flatten(), preds_zb, 'Inc', 'Target', 'Difflogic')

    preds_tt_had = predict_batch(model, x_tt_had, batch_size=256, device=args.device).cpu().numpy().flatten()
    preds_tt_semilep = predict_batch(model, x_tt_semilep, batch_size=256, device=args.device).cpu().numpy().flatten()
    # # preds_qcd = predict_batch(model, x_qcd, batch_size=512, device=args.device).cpu().numpy().flatten()
    # # preds_sn = predict_batch(model, x_sn, batch_size=512, device=args.device).cpu().numpy().flatten()

    np.save(f"scores/{args.model_name}_ZeroBias.npy", preds_zb)
    np.save(f"scores/{args.model_name}_TTtoHadronic.npy", preds_tt_had)
    np.save(f"scores/{args.model_name}_TTtoSemileptonic.npy", preds_tt_semilep)
    # # np.save(f"scores/{args.model_name}_QCD.npy", preds_qcd)
    # # np.save(f"scores/{args.model_name}_SingleNeutrino.npy", preds_sn)

    exit(0)

    # _, _, _, _, _, _, x_test, y_test, is_outlier_test = \
    #     get_training_data("saved_inputs_targets", zb_frac=5, backend="torch")
    # x_train, y_train, is_outlier_train, x_val, y_val, is_outlier_val, x_test, y_test, is_outlier_test = \
    #     get_training_data("saved_inputs_targets", backend="torch", zb_frac=5, use_outliers=True)
    
    n_max = 1_000
    x_test = x_test[:n_max]
    y_test = y_test[:n_max]
    is_outlier_test = is_outlier_test[:n_max]

    x_train = x_train[:n_max]
    y_train = y_train[:n_max]
    is_outlier_train = is_outlier_train[:n_max]

    y_test = y_test.cpu().detach().numpy().flatten()
    y_train = y_train.cpu().detach().numpy().flatten()

    # print("before transform:")
    # print(f"x_test[0, :9, 0] =\n{x_test[0, :9, 0]}")

    # if args.transform_name != "identity":
    #     transform = get_transform(args.transform_name)
    #     if "st" in args.transform_name:
    #         bool_transform = get_transform(args.transform_name.replace("st", "ct"))
    #         x_bool = bool_transform(x_test.int()).bool().int().float()
    #         x_test = transform(x_test.int())
    #         x_train = transform(x_train.int())
    #     else:
    #         x_test = transform(x_test.int())
    #         x_train = transform(x_train.int())
    #         x_bool = x_test.bool().int().float()        

    # print("after transform:")
    # print(f"x_test[0, :, :9, 0] =\n{x_test[0, :, :9, 0]}")
    # print(f"x_bool[0, :, :9, 0] =\n{x_bool[0, :, :9, 0]}")

    model = torch.load(args.model_path, map_location="cpu", weights_only=False)
    model.to("cpu")

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

    print(f"x_test.shape = {x_test.shape}, x_test.dtype = {x_test.dtype}")

    preds_train = predict_batch(model, x_test, batch_size=256, device="cpu").cpu().numpy().flatten()
    print(f"preds_train[:10] =\n{preds_train[:10].flatten()}")
    model.train(False)
    preds_eval = predict_batch(model, x_test, batch_size=256, device="cpu").cpu().numpy().flatten()
    print(f"preds_eval[:10] =\n{preds_eval[:10].flatten()}")
    # preds_eval_bin = predict_batch(model, x_bool, batch_size=256, device="cpu").cpu().numpy().flatten()

    print(f"preds_train[:10] =\n{preds_train[:10].flatten()}")
    print(f"preds_eval[:10] =\n{preds_eval[:10].flatten()}")
    # print(f"preds_eval_bin[:10] =\n{preds_eval_bin[:10].flatten()}")

    draw = Draw('plots', interactive=args.interactive)
    draw.make_scatter_density_plot(y_test, preds_train, 'Inc', 'Target', 'Difflogic')
    if (is_outlier_test==0).sum() > 0:
        draw.make_scatter_density_plot(y_test[~is_outlier_test], preds_train[~is_outlier_test], 'ZB', 'Target', 'Difflogic')
    if is_outlier_test.sum() > 0:
        draw.make_scatter_density_plot(y_test[is_outlier_test], preds_train[is_outlier_test], 'TT', 'Target', 'Difflogic')

    draw.make_scatter_plot(
        teacher_scores=[y_test[~is_outlier_test], y_test[is_outlier_test]],
        student_scores=[preds_train[~is_outlier_test], preds_train[is_outlier_test]],
        labels=['Difflogic ZB', 'Difflogic TT'],
    )

    model.train(True)
    print(f"x_train[:2] =\n{x_train[:2]}")
    print(f"x_test[:2] =\n{x_test[:2]}")
    train_preds = predict_batch(model, x_train, batch_size=256, device="cpu").cpu().numpy().flatten()
    _ = predict_batch(model, x_test, batch_size=256, device="cpu").cpu().numpy().flatten()

    print(f"train_preds[:10] =\n{train_preds[:10].flatten()}")
    print(f"_[:10] =\n{_[:10].flatten()}")
    draw.make_scatter_plot(
        teacher_scores=[y_train[~is_outlier_train], y_train[is_outlier_train]],
        student_scores=[train_preds[~is_outlier_train], train_preds[is_outlier_train]],
        labels=['TRAIN Difflogic ZB', 'TRAIN Difflogic TT'],
    )

    draw.plot_anomaly_score_distribution(
        [y_test[~is_outlier_test], preds_train[~is_outlier_test], y_test[is_outlier_test], preds_train[is_outlier_test]],
        ['Target ZB', 'Pred ZB', 'Target TT', 'Pred TT'],
        "anomaly-scores-train-mode",
    )

    # draw.plot_anomaly_score_distribution(
    #     [y_test, preds_train, preds_eval, preds_eval_bin],
    #     ['Target', 'train mode', 'eval mode', 'eval mode bin input'],
    #     "anomaly-scores-modes-inputs",
    # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load a LGN model and make predictions')
    parser.add_argument('--model-path', type=str, default='data/models/latest.pt', help='Directory containing the LGN model')
    parser.add_argument('--model-name', type=str, default='my-model', help='Name of the LGN model')
    parser.add_argument('--transform-name', type=str, default='t4qb', help='Name of the transform to apply')
    parser.add_argument('--interactive', action='store_true', help='Interactively display plots as they are created', default=False)
    parser.add_argument('--conv-3d', action='store_true', help='Use 3D convolutional layers', default=False)
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on')
    main(parser.parse_args())
