from typing import List
import torch


def get_int_to_bits_transform(num_bits=4):
    def transform(x):
        bit_masks = torch.tensor([2**i for i in range(num_bits)], device=x.device, dtype=x.dtype)
        shape = [1] * x.ndim
        shape.insert(-1, num_bits)  # Insert num_bits at second-to-last position
        bit_masks = bit_masks.view(shape)
        bits = ((x.unsqueeze(-2) & bit_masks) > 0).float()
        return bits
    return transform


def get_threshold_transform(thresholds: List[float]):
    def transform(x):
        print(f"before:\n{x[0, 0, :]}")

        result = torch.stack([(x > t).float() for t in thresholds], dim=1)
        print(f"after:\n{result[0, :, 0, :]}")
        return result
    return transform


def get_smooth_threshold_transform(thresholds: List[float], steepness: float = 1.0):
    thresholds = torch.tensor(thresholds)
    def transform(x):
        # Using sigmoid to create smooth transitions
        print(f"before:\n{x[0, 0, :]}")
        # result = torch.stack([torch.sigmoid(steepness * (x - t - 0.5)/(torch.sqrt(t) + 1e-6)) for t in thresholds], dim=1)
        result = torch.stack([torch.sigmoid(steepness * ((x - t)/(t + 1e-6) - 0.5)) for t in thresholds], dim=1)
        print(f"after:\n{result[0, :, 0, :]}")
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
        print(f"before:\n{x[0, 0, :]}")
        result = torch.stack(result, dim=1).float()
        print(f"after:\n{result[0, :, 0, :]}")
        return result
    return transform


def get_transform(name: str):
    if name == "t4qb":
        transform = get_threshold_transform([0, 1, 2, 32])
    elif name == "t5qb":
        transform = get_threshold_transform([0, 1, 2, 16, 64])
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