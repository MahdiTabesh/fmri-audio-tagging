# utils/fmri_encoding.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import pearsonr
import gc
#referenced https://medium.com/data-science/linear-regression-with-pytorch-eb6dedead817
#and https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html
# z-score normalization for features and fMRI
# normalize each column to have 0 mean and unit variance
def normalize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normed = (data - mean) / (std + 1e-8)
    return normed

# create time windowed feature-label pairs
# use current and previous window_length-1 timesteps to predict current Y label
def create_windowed_data(X, Y, window_length):
    n_time = X.shape[0]
    X_windows = []
    Y_targets = []
    for t in range(n_time - window_length + 1):
        window = X[t : t + window_length, :]
        X_windows.append(window.flatten())
        Y_targets.append(Y[t + window_length - 1, :])
    X_windows = np.array(X_windows)
    Y_targets = np.array(Y_targets)
    return X_windows, Y_targets

# PyTorch ridge regression model (linear regression + L2 weight penalty)
# includes training, test, and voxel-wise correlation computation
def mod(X_win, Y_win, model_name, alpha=1.0, epochs=50, batch_size=1024, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_samples = X_win.shape[0]
    split_idx = int(0.8 * n_samples) # 80% training, 20% test
    X_trainf, X_testf = X_win[:split_idx], X_win[split_idx:]
    Y_trainf, Y_testf = Y_win[:split_idx], Y_win[split_idx:]

    X_train = torch.tensor(X_trainf, dtype=torch.float32).to(device)
    Y_train = torch.tensor(Y_trainf, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_testf, dtype=torch.float32).to(device)
    Y_test = torch.tensor(Y_testf, dtype=torch.float32).to(device)
    dataset = TensorDataset(X_train, Y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = nn.Linear(X_train.shape[1], Y_train.shape[1], bias=True).to(device) # linear regression model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x1, y1 in loader:
            optimizer.zero_grad()
            pred = model(x1)
            loss = F.mse_loss(pred, y1)+alpha*torch.sum(model.weight**2)  # MSE + L2 regularization
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    model.eval()
    with torch.no_grad():
        Y_pred = model(X_test).cpu().numpy()
        Y_true = Y_test.cpu().numpy()

    # compute voxel-wise Pearson r between prediction and ground truth
    r_values = np.full(Y_true.shape[1], np.nan)  # some voxels may be flat
    for i in range(Y_true.shape[1]):
        y_true = Y_true[:, i]
        y_pred = Y_pred[:, i]
        if np.std(y_true) < 1e-8 or np.std(y_pred) < 1e-8:  # skip flat signals
            continue
        r_values[i], _ = pearsonr(y_true, y_pred)

    torch.cuda.empty_cache()
    gc.collect()

    return r_values

def compute_voxel_correlations(Y_true, Y_pred):
    n_vox = Y_true.shape[1]
    r_vals = np.full(n_vox, np.nan)
    for i in range(n_vox):
        y_true = Y_true[:, i]
        y_pred = Y_pred[:, i]
        # skip if zero var
        if np.std(y_true) < 1e-8 or np.std(y_pred) < 1e-8:
            continue
        r_vals[i], _ = pearsonr(y_true, y_pred)
    return r_vals
