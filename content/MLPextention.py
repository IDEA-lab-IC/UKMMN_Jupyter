# Functions used in MLP dataset processing, model training and testing, etc.
# --- Standard library
import time
import math
import random

# --- Core scientific stack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- PyTorch
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# --- Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from MLPpackage import MLP, count_params

#### helper to build, train, time, and evaluate one model ####
def run_experiment(hidden, *, in_dim, out_dim, train_loader, val_loader, test_loader,
                   model_cls=MLP, lr=1e-3, wd=1e-5, epochs=120, patience=20, device=None):
    """
    Builds an MLP with 'hidden', trains it, measures wall-clock time,
    and returns metrics, parameter count, and learning curves.
    """
    # Build fresh model
    m = MLP(in_dim=in_dim, hidden=hidden, out_dim=out_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = m.to(device)

    # Optimizer/loss
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)

    # Train with early stopping (simple inline loop)
    best_val = float('inf')
    best_state, wait = None, 0
    train_losses, val_losses = [], []

    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        # --- train ---
        m.train()
        tr_sum = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = m(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            tr_sum += loss.item() * xb.size(0)
        tr_loss = tr_sum / len(train_loader.dataset)

        # --- validate ---
        m.eval()
        va_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = m(xb)
                loss = criterion(pred, yb)
                va_sum += loss.item() * xb.size(0)
        va_loss = va_sum / len(val_loader.dataset)

        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        # early stopping
        if va_loss < best_val - 1e-8:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    # Load best weights and measure time
    if best_state is not None:
        m.load_state_dict(best_state)
    elapsed = time.perf_counter() - t0

    # --- Test evaluation (overall MAE/RMSE) in original units if scaler_out exists ---
    m.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            pb = m(xb.to(device)).cpu().numpy()
            preds.append(pb)
            trues.append(yb.cpu().numpy())
    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)

    if 'SCALE_OUTPUTS' in globals() and SCALE_OUTPUTS and 'scaler_out' in globals():
        y_pred_eval = scaler_out.inverse_transform(y_pred)
        y_true_eval = scaler_out.inverse_transform(y_true)
    else:
        y_pred_eval = y_pred
        y_true_eval = y_true

    mae = mean_absolute_error(y_true_eval, y_pred_eval, multioutput='raw_values')
    mse = mean_squared_error(y_true_eval, y_pred_eval, multioutput='raw_values')
    rmse = np.sqrt(mean_squared_error(y_true_eval, y_pred_eval, multioutput='raw_values'))
    overall_mae = mae.mean()
    overall_mse = mse.mean()
    overall_rmse = rmse.mean()

    return {
        "model": m,
        "hidden": tuple(hidden),
        "params": count_params(m),
        "best_val_mse": float(best_val),
        "overall_test_mae": float(overall_mae),
        "overall_test_mse": float(overall_mse),
        "overall_test_rmse": float(overall_rmse),
        "time_sec": float(elapsed),
        "train_loss_curve": train_losses,
        "val_loss_curve": val_losses,
    }


#### Plot the comparison of model performances ####
def plot_model_comparison(results, names=None, test_key="overall_test_mse",
                          title="Model Comparison Summary (log-scaled errors)",
                          figsize=(12, 8), rotate=25):
    """
    Plot a 2x2 summary: Val MSE (log), Test metric (log), #Params (log), Time (s).
    
    Args:
        results: dict[name] -> {
            "best_val_mse": float,
            test_key: float (e.g., "overall_test_mse" or "overall_test_rmse"),
            "params": int,
            "time_sec": float,
        }
        names: optional list[str] to set/lock order; defaults to list(results.keys()).
        test_key: key in results for test error (default "overall_test_mse").
        title: figure suptitle.
        figsize: tuple figure size.
        rotate: xtick label rotation degrees.
    """
    if names is None:
        names = list(results.keys())

    val_mse = [results[k]["best_val_mse"] for k in names]
    test_err = [results[k][test_key] for k in names]
    params = [results[k]["params"] for k in names]
    times = [results[k]["time_sec"] for k in names]

    x = np.arange(len(names))
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, y=1.02)

    # (1) Validation MSE (log)
    axes[0, 0].bar(x, val_mse, color=colors[0])
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xticks(x); axes[0, 0].set_xticklabels(names, rotation=rotate, ha="right")
    axes[0, 0].set_ylabel("Best Val MSE (log)")
    axes[0, 0].set_title("Validation Loss")

    # (2) Test error (log)
    axes[0, 1].bar(x, test_err, color=colors[1])
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xticks(x); axes[0, 1].set_xticklabels(names, rotation=rotate, ha="right")
    axes[0, 1].set_ylabel(f"Test {test_key.split('_')[-1].upper()} (log)")
    axes[0, 1].set_title("Test Error")

    # (3) Parameter count (log)
    axes[1, 0].bar(x, params, color=colors[2])
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(names, rotation=rotate, ha="right")
    axes[1, 0].set_ylabel("# Trainable Parameters (log)")
    axes[1, 0].set_title("Model Size")

    # (4) Training time (linear)
    axes[1, 1].bar(x, times, color=colors[3])
    axes[1, 1].set_xticks(x); axes[1, 1].set_xticklabels(names, rotation=rotate, ha="right")
    axes[1, 1].set_ylabel("Training Time (s)")
    axes[1, 1].set_title("Training Time")

    plt.tight_layout()
    plt.show()


#### plot the validations curves of the models ####
def plot_val_curves(results, names=None, log=True, figsize=(5,4), title="Validation Curves by Architecture"):
    """
    Plot validation-loss curves for multiple runs stored in `results`.

    Args:
        results: dict[name] -> {"val_loss_curve": list/array of floats}
        names: optional list[str] to control plotting order; defaults to results.keys()
        log: if True, use log scale on y-axis
        figsize: tuple for figure size
        title: plot title
    """
    if names is None:
        names = list(results.keys())

    plt.figure(figsize=figsize)
    for name in names:
        curve = results[name]["val_loss_curve"]
        plt.plot(curve, label=name)
    if log:
        plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Val MSE" + (" (log)" if log else ""))
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

    
#### to visualise the split of dataset based on volume fraction ####
def plot_vf_split(vol_sorted, mid, figsize=(6,3)):
    plt.figure(figsize=figsize)
    x = np.arange(len(vol_sorted))
    plt.scatter(x, vol_sorted, s=10, alpha=0.6, label="samples")
    plt.axvline(mid, color="purple", linestyle="--", label="split index")
    plt.xlabel("Sample index (sorted by volume fraction)")
    plt.ylabel("Volume fraction")
    plt.title("Volume fraction of dataset with split point")
    plt.legend()
    plt.tight_layout()
    plt.show()


#### list the original and rotated cell properties to compare ####
def print_property_table(rows, headers=("pattern","D11","D12","D13","D22","D23","D33","vf"),
                         d_decimals=5, vf_decimals=4, title="Property Comparison for Rotated Unit Cells"):
    """
    Print a neatly aligned table of properties without pandas.

    Args:
        rows: iterable of (name, feat) where feat = [D11,D12,D13,D22,D23,D33,vf]
        headers: column names
        d_decimals: decimals for D-components
        vf_decimals: decimals for volume fraction
        title: heading printed above the table
    """
    # format rows
    table_rows = []
    for name, feat in rows:
        vals = [f"{feat[i]:.{d_decimals}f}" for i in range(6)] + [f"{feat[6]:.{vf_decimals}f}"]
        table_rows.append([name] + vals)

    # column widths
    col_w = [max(len(headers[i]), max(len(r[i]) for r in table_rows)) + 2 for i in range(len(headers))]

    # header + separator
    header_row = "".join(headers[i].ljust(col_w[i]) for i in range(len(headers)))
    sep = "-" * len(header_row)

    # print
    print(f"=== {title} ===")
    print(header_row)
    print(sep)
    for r in table_rows:
        print("".join(r[i].ljust(col_w[i]) for i in range(len(headers))))
