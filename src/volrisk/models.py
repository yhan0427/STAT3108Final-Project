import copy
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class TrainOnlyScaler:
    """Median imputation, 1/99% clipping and scaling, plus explicit missing masks."""
    def fit(self, values):
        x = np.asarray(values, dtype=float)
        self.median = np.array([np.nanmedian(v) if np.isfinite(v).any() else 0.0 for v in x.T])
        clean = np.where(np.isfinite(x), x, self.median)
        self.low, self.high = np.quantile(clean, [0.01, 0.99], axis=0)
        clean = np.clip(clean, self.low, self.high)
        self.mean, self.std = clean.mean(axis=0), clean.std(axis=0)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)
        return self

    def transform(self, values, missing=False):
        x = np.asarray(values, dtype=float)
        mask = ~np.isfinite(x)
        clean = np.where(mask, self.median, x)
        scaled = (np.clip(clean, self.low, self.high) - self.mean) / self.std
        if missing:
            scaled = np.concatenate([scaled, mask.astype(float)], axis=1)
        return scaled.astype(np.float32)


class Sequences(Dataset):
    def __init__(self, market, fundamentals, targets, indices, lookback):
        self.market, self.fundamentals = market, fundamentals
        self.targets, self.indices, self.lookback = targets, np.asarray(indices), lookback

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        i = self.indices[item]
        return self.market[i - self.lookback + 1:i + 1], self.fundamentals[i], self.targets[i]


class VolatilityLSTM(nn.Module):
    def __init__(self, market_dim, fund_dim=0, hidden=24):
        super().__init__()
        self.fund_dim = fund_dim
        self.encoder = nn.LSTM(market_dim, hidden, batch_first=True)
        if fund_dim:
            self.fund_encoder = nn.Sequential(nn.Linear(fund_dim, 16), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden + (16 if fund_dim else 0), 16), nn.ReLU(), nn.Dropout(0.1), nn.Linear(16, 1))

    def forward(self, market, fundamentals):
        _, (hidden, _) = self.encoder(market)
        representation = hidden[-1]
        if self.fund_dim:
            representation = torch.cat([representation, self.fund_encoder(fundamentals)], dim=1)
        return self.head(representation).squeeze(-1)


def predict_network(model, dataset, batch_size=1024):
    model.eval()
    predictions = []
    with torch.inference_mode():
        for market, funds, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            predictions.append(model(market, funds).numpy())
    return np.concatenate(predictions)


def train_network(market, funds, target, train_indices, validation_indices, config, seed, use_fundamentals):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    cfg = config["lstm"]
    # Zero-input branch in market-only model keeps architecture/parameter count
    # identical. Only the fundamental information supplied to the branch changes.
    model = VolatilityLSTM(market.shape[1], funds.shape[1], cfg["hidden_size"])
    if not use_fundamentals:
        funds = np.zeros_like(funds)
    train_data = Sequences(market, funds, target, train_indices, config["lookback"])
    val_data = Sequences(market, funds, target, validation_indices, config["lookback"])
    loader = DataLoader(train_data, batch_size=cfg["batch_size"], shuffle=True, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    best_loss, best_state, stale, history = np.inf, None, 0, []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        losses = []
        for x, f, y in loader:
            optimizer.zero_grad()
            loss = torch.mean((model(x, f) - y) ** 2)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()) * len(y))
        val_prediction = predict_network(model, val_data)
        val_loss = float(np.mean((val_prediction - target[validation_indices]) ** 2))
        history.append({"epoch": epoch, "train_mse_standardized_log": sum(losses) / len(train_data), "validation_mse_standardized_log": val_loss})
        print(f"LSTM funds={use_fundamentals} seed={seed} epoch={epoch}: val={val_loss:.5f}", flush=True)
        if val_loss < best_loss - 1e-5:
            best_loss, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return model, history
