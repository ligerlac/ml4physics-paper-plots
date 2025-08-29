from typing import List
import numpy as np
from sklearn.metrics import roc_curve, auc
import h5py


def get_fractions_above_threshold(scores):
    thresholds = scores.flatten()
    thresholds.sort()
    fractions = np.linspace(1, 0, len(thresholds))
    return thresholds, fractions


def get_rounded_str(value) -> str:
    """
    Applies relative rounding to a number based on its magnitude:
    - 0 to 10: 2 decimal places
    - 10 to 100: 1 decimal place
    - 100 and above: integer (0 decimal places)
    
    Parameters:
    -----------
    value : float or int
        The number to round
        
    Returns:
    --------
    float or int
        The rounded number
    """
    # Handle edge cases
    if value is None or np.isnan(value):
        return value
    
    # Take absolute value for comparison (to handle negative numbers)
    abs_value = abs(value)
    
    # Apply rounding based on magnitude
    if abs_value < 10:
        # Round to 2 decimal places
        return str(round(value, 2))
    elif abs_value < 100:
        # Round to 1 decimal place
        return str(round(value, 1))
    else:
        # Round to integer
        return str(int(round(value, 0)))


def get_anomaly_scores_ae(
        inputs: np.ndarray, outputs: np.ndarray, 
) -> np.ndarray:
    """
    Calculate anomaly score from in- and outputs of an autoencoder.
    """
    mse = np.mean((inputs - outputs) ** 2, axis=(1,2))
    score = np.log(mse * 32)
    return score


def quantize(arr: np.ndarray, precision: tuple = (16, 8)) -> np.ndarray:
    word, int_ = precision
    decimal = word - int_
    step = 1 / 2**decimal
    max_ = 2**int_ - step
    arrq = step * np.round(arr / step)
    arrc = np.clip(arrq, 0, max_)
    return arrc


def get_roc_from_scores(
        bg_scores: np.ndarray, sig_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    y_trues = np.concatenate([np.ones_like(sig_scores), np.zeros_like(bg_scores)])
    y_preds = np.concatenate([sig_scores, bg_scores])
    fpr, tpr, _ = roc_curve(y_trues, y_preds)
    return fpr, tpr


def get_roc_dict(
        score_dict: dict, bg_label: str, sig_labels: List[str]
) -> dict:
    """
    Get a dictionary of the form
    {
        proc_0: (fpr, tpr),
        ...,
        proc_n: (fpr, tpr),
    }
    """
    d = {}
    bg_score = score_dict[bg_label]
    for proc in sig_labels:
        sig_score = score_dict[proc]
        d[proc] = get_roc_from_scores(bg_score, sig_score)
    return d


def get_training_data(base_dir: str = "saved_inputs_targets", backend: str = "numpy", zb_frac: int = 1, use_outliers: bool = True):

    X_ZB_train = np.load(base_dir + "/ZB_train.npy")
    X_ZB_val = np.load(base_dir + "/ZB_val.npy")
    X_ZB_test = np.load(base_dir + "/ZB_test.npy")

    y_ZB_train = np.load(base_dir + "/ZB_student_target_train.npy")
    y_ZB_val = np.load(base_dir + "/ZB_student_target_val.npy")
    y_ZB_test = np.load(base_dir + "/ZB_student_target_test.npy")

    X_outlier_train = np.load(base_dir + "/tt2l2nu_train.npy")
    X_outlier_val = np.load(base_dir + "/tt2l2nu_val.npy")
    X_outlier_test = np.load(base_dir + "/tt2l2nu_test.npy")

    y_outlier_train = np.load(base_dir + "/tt2l2nu_student_target_train.npy")
    y_outlier_val = np.load(base_dir + "/tt2l2nu_student_target_val.npy")
    y_outlier_test = np.load(base_dir + "/tt2l2nu_student_target_test.npy")

    print(f"y_ZB_train.shape = {y_ZB_train.shape}")
    print(f"y_outlier_train.shape = {y_outlier_train.shape}")

    print(f"y_ZB_val.shape = {y_ZB_val.shape}")
    print(f"y_outlier_val.shape = {y_outlier_val.shape}")

    print(f"y_ZB_test.shape = {y_ZB_test.shape}")
    print(f"y_outlier_test.shape = {y_outlier_test.shape}")

    if zb_frac != -1:
        X_ZB_train = X_ZB_train[:zb_frac*len(y_outlier_train)]
        y_ZB_train = y_ZB_train[:zb_frac*len(y_outlier_train)]
        X_ZB_val = X_ZB_val[:zb_frac*len(y_outlier_val)]
        y_ZB_val = y_ZB_val[:zb_frac*len(y_outlier_val)]
        X_ZB_test = X_ZB_test[:zb_frac*len(y_outlier_test)]
        y_ZB_test = y_ZB_test[:zb_frac*len(y_outlier_test)]

    if use_outliers:
        X_train = np.concatenate([X_ZB_train, X_outlier_train], axis=0)
        y_train = np.concatenate([y_ZB_train, y_outlier_train], axis=0)
        X_val = np.concatenate([X_ZB_val, X_outlier_val], axis=0)
        y_val = np.concatenate([y_ZB_val, y_outlier_val], axis=0)
        X_test = np.concatenate([X_ZB_test, X_outlier_test], axis=0)
        y_test = np.concatenate([y_ZB_test, y_outlier_test], axis=0)
    else:
        X_train = X_ZB_train
        y_train = y_ZB_train
        X_val = X_ZB_val
        y_val = y_ZB_val
        X_test = X_ZB_test
        y_test = y_ZB_test

    is_outlier_train = np.concatenate(
        [np.zeros_like(y_ZB_train), np.ones_like(y_outlier_train)], axis=0
    ).astype(bool)
    is_outlier_val = np.concatenate(
        [np.zeros_like(y_ZB_val), np.ones_like(y_outlier_val)], axis=0
    ).astype(bool)
    is_outlier_test = np.concatenate(
        [np.zeros_like(y_ZB_test), np.ones_like(y_outlier_test)], axis=0
    ).astype(bool)
    

    perm = np.random.permutation(X_train.shape[0])
    X_train = X_train[perm]
    y_train = y_train[perm]
    is_outlier_train = is_outlier_train[perm]

    if backend == "torch":
        import torch
        X_train = torch.squeeze(torch.from_numpy(X_train), -1)
        X_val = torch.squeeze(torch.from_numpy(X_val), -1)
        X_test = torch.squeeze(torch.from_numpy(X_test), -1)

        y_train = torch.unsqueeze(torch.from_numpy(y_train), -1)
        y_val = torch.unsqueeze(torch.from_numpy(y_val), -1)
        y_test = torch.unsqueeze(torch.from_numpy(y_test), -1)

        # y_train = torch.from_numpy(y_train)
        # y_val = torch.from_numpy(y_val)
        # y_test = torch.from_numpy(y_test)

        is_outlier_train = torch.tensor(is_outlier_train, dtype=torch.bool)
        is_outlier_val = torch.tensor(is_outlier_val, dtype=torch.bool)
        is_outlier_test = torch.tensor(is_outlier_test, dtype=torch.bool)

    return X_train, y_train, is_outlier_train, X_val, y_val, is_outlier_val, X_test, y_test, is_outlier_test


def get_input_data(base_dir: str = "saved_inputs_targets"):
    data = {
        "ZB": np.load(f"{base_dir}/ZB_test.npy")
    }
    with h5py.File(f"{base_dir}/tt_hadronic_with_l1evtnPV.h5", 'r') as f:
        data["ttbar_had"] = f["CaloRegions"][:]

    with h5py.File(f"{base_dir}/tt_semileptonic_with_l1evtnPV.h5", 'r') as f:
        data["ttbar_semilep"] = f["CaloRegions"][:]

    with h5py.File(f"{base_dir}/qcd_with_l1evtnPV.h5", 'r') as f:
        data["qcd"] = f["CaloRegions"][:]

    with h5py.File(f"{base_dir}/singleneutrino_with_l1evtnPV.h5", 'r') as f:
        data["singleneu"] = f["CaloRegions"][:]

    return data

