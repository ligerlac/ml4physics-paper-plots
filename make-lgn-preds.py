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
    with torch.no_grad():
        y_pred = []
        for i in tqdm(range(0, x.size(0), batch_size)):
            y_pred.append(model(x[i:i+batch_size].to(device)))
        y_pred = torch.cat(y_pred)
        print(f"pred step: y_pred[:3] = {y_pred[:3]}")
    return y_pred


def main(args):
    _, _, _, _, _, _, x_test, y_test, is_outlier_test = \
        get_training_data("saved_inputs_targets", zb_frac=5, backend="torch")
    
    # shuffle the test data
    perm = torch.randperm(x_test.size(0))

    perm = perm[:1000]  # for quick testing

    x_test = x_test[perm]
    y_test = y_test[perm]
    is_outlier_test = is_outlier_test[perm]

    y_test = y_test.cpu().detach().numpy().flatten()

    if args.transform_name != "identity":
        transform = get_transform(args.transform_name)
        x_test = transform(x_test.int())

    model = torch.load(args.model_path, map_location="cpu", weights_only=False)
    model.to("cpu")

    print(f"model = {model}")
    for layer in model:
        print(f"layer = {layer}, #neurons = {getattr(layer, 'num_neurons', None)}, #weights = {getattr(layer, 'num_weights', None)}")
        if layer.__class__.__name__ == 'LogicLayer':
            print(f"layer.weight = {layer.weight}")
            argmaxes = layer.weight.argmax(dim=1)
            plt.hist(argmaxes.detach().cpu().numpy().flatten(), bins=100)
            plt.show()
        # print(f"  layer parameters = {[p.shape for p in layer.parameters()]}")

    exit(0)


    preds_train = predict_batch(model, x_test, batch_size=256, device="cpu").cpu().numpy()
    model.train(False)
    preds_eval = predict_batch(model, x_test, batch_size=256, device="cpu").cpu().numpy()

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

    draw.plot_anomaly_score_distribution(
        [y_test[~is_outlier_test], preds_train[~is_outlier_test], y_test[is_outlier_test], preds_train[is_outlier_test]],
        ['Target ZB', 'Difflogic ZB', 'Target TT', 'Difflogic TT'],
        "anomaly-scores-train-mode",
    )

    # y_diff_float = model(x_test).detach().numpy()
    # y_diff_bin = model(x_test.bool().int()).detach().numpy()

    # model.train(False)
    # y_bin_float = model(x_test).numpy()
    # y_bin_bin = model(x_test.bool().int()).numpy()
    # model.train(True)

    # draw.plot_anomaly_score_distribution(
    #     [y_test.detach().numpy(), y_diff_float, y_diff_bin, y_bin_float, y_bin_bin],
    #     ['Target', 'Diff(float)', 'Diff(bin)', 'Bin(float)', 'Bin(bin)'],
    #     "anomaly-scores-modes-inputs",
    # )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load a LGN model and make predictions')
    parser.add_argument('--model-path', type=str, default='data/models/latest.pt', help='Directory containing the LGN model')
    parser.add_argument('--transform-name', type=str, default='t4qb', help='Name of the transform to apply')
    parser.add_argument('--interactive', action='store_true', help='Interactively display plots as they are created', default=False)
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on')
    main(parser.parse_args())
