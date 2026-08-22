
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

def metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }

def train_xgb(X_train, y_train, cfg, use_gpu=True):
    params = dict(
        n_estimators=int(cfg["xgb_estimators"]),
        max_depth=int(cfg["xgb_max_depth"]),
        learning_rate=float(cfg["xgb_learning_rate"]),
        subsample=.9, colsample_bytree=.9,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=int(cfg["seed"]),
        n_jobs=-1,
    )
    if use_gpu:
        params["device"] = "cuda"
    model = XGBRegressor(**params)
    try:
        model.fit(X_train, y_train)
        return model, ("cuda" if use_gpu else "cpu")
    except Exception as e:
        if not use_gpu:
            raise
        print(f"[WARN] XGBoost CUDA failed ({e}); retrying on CPU.")
        params["device"] = "cpu"
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        return model, "cpu"

def train_rf(X_train, y_train, cfg):
    m = RandomForestRegressor(
        n_estimators=int(cfg["rf_estimators"]),
        max_depth=18,
        min_samples_leaf=2,
        max_features=.8,
        n_jobs=-1,
        random_state=int(cfg["seed"])
    )
    m.fit(X_train, y_train)
    return m

class EnergyMLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(.10),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

class TorchRegressor:
    def __init__(self, model, scaler, y_mean, y_std, device):
        self.model, self.scaler, self.y_mean, self.y_std, self.device = model, scaler, y_mean, y_std, device
    def predict(self, X):
        z = self.scaler.transform(X).astype("float32")
        self.model.eval()
        out=[]
        with torch.no_grad():
            for i in range(0, len(z), 8192):
                t=torch.from_numpy(z[i:i+8192]).to(self.device)
                out.append(self.model(t).cpu().numpy())
        return np.concatenate(out) * self.y_std + self.y_mean

def train_mlp(X_train, y_train, cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = StandardScaler().fit(X_train)
    X = scaler.transform(X_train).astype("float32")
    y_raw = np.asarray(y_train, dtype="float32")
    y_mean=float(y_raw.mean()); y_std=float(y_raw.std()+1e-8)
    y = (y_raw-y_mean)/y_std
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=int(cfg["batch_size"]), shuffle=True,
                        pin_memory=(device=="cuda"))
    model=EnergyMLP(X.shape[1]).to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn=nn.HuberLoss()
    model.train()
    for epoch in range(int(cfg["mlp_epochs"])):
        total=0.0
        for xb,yb in loader:
            xb,yb=xb.to(device, non_blocking=True),yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss=loss_fn(model(xb), yb)
            loss.backward(); opt.step()
            total += loss.item()*len(xb)
        if epoch in {0, int(cfg["mlp_epochs"])-1} or (epoch+1)%5==0:
            print(f"MLP epoch {epoch+1:02d}: loss={total/len(ds):.4f}")
    return TorchRegressor(model, scaler, y_mean, y_std, device), device
