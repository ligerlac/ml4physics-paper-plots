from typing import List
import torch


def get_int_to_bits_transform(num_bits=4):
    def transform(x):
        bit_masks = torch.tensor([2**i for i in range(num_bits)], device=x.device, dtype=x.dtype)
        shape = [1] * x.ndim
        shape.insert(-2, num_bits)  # Insert num_bits at third-to-last position
        bit_masks = bit_masks.view(shape)
        bits = ((x.unsqueeze(-3) & bit_masks) > 0).float()
        return bits
    return transform


def get_threshold_transform(thresholds: List[float]):
    def transform(x):
        result = torch.stack([(x > t).float() for t in thresholds], dim=1)
        return result
    return transform


def get_smooth_threshold_transform(thresholds: List[float], steepness: float = 1.0):
    thresholds = torch.tensor(thresholds)
    def transform(x):
        # Using sigmoid to create smooth transitions
        # result = torch.stack([torch.sigmoid(steepness * (x - t - 0.5)/(torch.sqrt(t) + 1e-6)) for t in thresholds], dim=1)
        result = torch.stack([torch.sigmoid(steepness * ((x - t)/(t + 1e-6) - 0.5)) for t in thresholds], dim=1)
        return result
    return transform


def get_clipped_threshold_transform(thresholds: List[float]):
    thresholds.append(1024)
    def transform(x):
        result = []
        for i in range(len(thresholds)-1):
            l = thresholds[i]
            u = thresholds[i+1]
            result.append(torch.clamp((x-l) / (u-l), 0, 1))
        result = torch.stack(result, dim=1).float()
        return result
    return transform


def get_log_threshold_transform(thresholds: List[float]):
    def transform(x):
        result = torch.stack([torch.log1p(torch.clamp(x - t, min=0) / 2.) for t in thresholds], dim=1)
        return result
    return transform


def get_transform(name: str):
    if name == "log":
        transform = lambda x: torch.unsqueeze(torch.log1p(x) / 2.0, dim=1)
    elif name == "lt2":
        transform = get_log_threshold_transform([0, 2])
    elif name == "lt2h":
        transform = get_log_threshold_transform([1, 8])
    elif name == "lt3":
        transform = get_log_threshold_transform([1, 3, 5])
    elif name == "lt4":
        transform = get_log_threshold_transform([1, 3, 5, 10])
    elif name == "lt6":
        transform = get_log_threshold_transform([0, 1, 2, 4, 8, 16])
    elif name == "lt7":
        transform = get_log_threshold_transform([0, 1, 2, 4, 8, 16, 32])
    elif name == "lt3l":
        transform = get_log_threshold_transform([0, 1, 2])
    elif name == "t3qb":
        transform = get_threshold_transform([0, 1, 4])
    elif name == "t4qb":
        transform = get_threshold_transform([0, 1, 2, 32])
    elif name == "t5qb":
        transform = get_threshold_transform([0, 1, 2, 16, 64])
    elif name == "c3qb":
        transform = get_clipped_threshold_transform([0, 1, 4])
    elif name == "ct2":
        transform = get_clipped_threshold_transform([0, 2])
    elif name == "ct3":
        transform = get_clipped_threshold_transform([0, 2, 16])
    elif name == "ct4":
        transform = get_clipped_threshold_transform([0, 1, 2, 4])
    elif name == "st2":
        transform = get_smooth_threshold_transform([0, 2], steepness=2)
    elif name == "st3":
        transform = get_smooth_threshold_transform([0, 2, 16], steepness=2)
    elif name == "10bit":
        transform = get_int_to_bits_transform(num_bits=10)
    else:
        raise ValueError(f"Unknown transform name: {name}")
    return transform
    # transform = get_smooth_threshold_transform([0, 1, 2, 4, 64])
    transform = get_threshold_transform([0, 1, 2, 32])
    # transform = get_threshold_transform([0, 1, 2, 16, 64])
    # transform = get_threshold_transform([0, 1, 2, 64, 128])
    # transform = get_threshold_transform([0, 4])
    # transform = get_clipped_threshold_transform([0])
    # transform2 = get_clipped_threshold_transform([0, 1, 2, 64])
    # transform2 = get_smooth_threshold_transform([0, 1, 2, 4], steepness=5)

    # _ = transform(torch.tensor([[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]]))
    # _ = transform2(torch.tensor([[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]]))

    # exit(0)


        # ###   Optional: logarithmic scaling for reduced number of bits   ###
    # N_BITS = 10

    # # scale logarithmically w/ ideal base (no wasted range)
    # base = 1025 ** (1 / 2**N_BITS)
    # x_train = (torch.log(x_train + 1) / torch.log(torch.tensor(base))).int()
    # x_val = (torch.log(x_val + 1) / torch.log(torch.tensor(base))).int()

    # # represent in binary (expand dims to (m, 18, 14, N_BITS))
    # x_train = x_train.int().unsqueeze(-1).bitwise_and(1 << torch.arange(N_BITS-1, -1, -1)).ne(0).int()
    # x_val = x_val.int().unsqueeze(-1).bitwise_and(1 << torch.arange(N_BITS-1, -1, -1)).ne(0).int()

    # print(f'x_train.shape = {x_train.shape}')
    # print(f'x_val.shape = {x_val.shape}')

    # import matplotlib.pyplot as plt
    # plt.hist(x_train.detach().numpy().flatten(), bins=100)
    # plt.show()

    # plt.hist(x_train.detach().numpy().flatten(), bins=100)
    # plt.yscale("log")
    # plt.show()

    # exit()

if __name__ == "__main__":
    trafo = get_transform("ct4")
    x = torch.tensor([0, 1, 2, 3, 10])
    y = trafo(x)
    print(f"x =\n{x}")
    print(f"y =\n{y}")

    exit(0)

    transform1 = get_smooth_threshold_transform([0, 2, 4], steepness=2)
    transform2 = get_threshold_transform([0, 2, 4])
    x = torch.tensor([0, 1, 2, 3, 10])
    t1 = transform1(x)
    t2 = transform2(x)
    b = t2.bool().int()
    print(f"x = \n{x}")
    print(f"t1 = \n{t1}")
    print(f"b = \n{b}")
