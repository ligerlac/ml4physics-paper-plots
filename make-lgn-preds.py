import argparse
import torch
from tqdm import tqdm
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


def predict_batch(model, x, batch_size=128, device='cuda'):
    print(f"x.shape = {x.shape}")
    with torch.no_grad():
        y_pred = []
        for i in tqdm(range(0, x.size(0), batch_size)):
            y_pred.append(model(x[i:i+batch_size].to(device)))
        y_pred = torch.cat(y_pred)
    return y_pred


def main(args):
    # _, _, _, _, _, _, x_test, y_test, is_outlier_test = \
    #     get_training_data("saved_inputs_targets", zb_frac=5, backend="torch")
    x_train, y_train, is_outlier_train, x_val, y_val, is_outlier_val, x_test, y_test, is_outlier_test = \
        get_training_data("saved_inputs_targets", backend="torch", zb_frac=5, use_outliers=True)
    
    n_max = 10_000
    x_test = x_test[:n_max]
    y_test = y_test[:n_max]
    is_outlier_test = is_outlier_test[:n_max]

    x_train = x_train[:n_max]
    y_train = y_train[:n_max]
    is_outlier_train = is_outlier_train[:n_max]

    y_test = y_test.cpu().detach().numpy().flatten()
    y_train = y_train.cpu().detach().numpy().flatten()

    print("before transform:")
    print(f"x_test[0, :9, 0] =\n{x_test[0, :9, 0]}")


    if args.transform_name != "identity":
        transform = get_transform(args.transform_name)
        if "st" in args.transform_name:
            bool_transform = get_transform(args.transform_name.replace("st", "ct"))
            x_bool = bool_transform(x_test.int()).bool().int().float()
            x_test = transform(x_test.int())
            x_train = transform(x_train.int())
        else:
            x_test = transform(x_test.int())
            x_train = transform(x_train.int())
            x_bool = x_test.bool().int().float()        

    print("after transform:")
    print(f"x_test[0, :, :9, 0] =\n{x_test[0, :, :9, 0]}")
    print(f"x_bool[0, :, :9, 0] =\n{x_bool[0, :, :9, 0]}")

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


    preds_train = predict_batch(model, x_test, batch_size=256, device="cpu").cpu().numpy().flatten()
    model.train(False)
    preds_eval = predict_batch(model, x_test, batch_size=256, device="cpu").cpu().numpy().flatten()
    preds_eval_bin = predict_batch(model, x_bool, batch_size=256, device="cpu").cpu().numpy().flatten()

    print(f"preds_train[:10] =\n{preds_train[:10].flatten()}")
    print(f"preds_eval[:10] =\n{preds_eval[:10].flatten()}")
    print(f"preds_eval_bin[:10] =\n{preds_eval_bin[:10].flatten()}")

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
    train_preds = predict_batch(model, x_train, batch_size=256, device="cpu").cpu().numpy().flatten()
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

    draw.plot_anomaly_score_distribution(
        [y_test, preds_train, preds_eval, preds_eval_bin],
        ['Target', 'train mode', 'eval mode', 'eval mode bin input'],
        "anomaly-scores-modes-inputs",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load a LGN model and make predictions')
    parser.add_argument('--model-path', type=str, default='data/models/latest.pt', help='Directory containing the LGN model')
    parser.add_argument('--transform-name', type=str, default='t4qb', help='Name of the transform to apply')
    parser.add_argument('--interactive', action='store_true', help='Interactively display plots as they are created', default=False)
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on')
    main(parser.parse_args())
