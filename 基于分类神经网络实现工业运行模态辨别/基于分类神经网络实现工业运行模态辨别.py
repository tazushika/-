# -*- coding: utf-8 -*-
"""
Leak-free QDA Stage1 + PyTorch MLP Stage2 + selectable Stage3.

Stage3 can be switched among:
    STAGE3_BACKBONE = "qda"     # original stable baseline
    STAGE3_BACKBONE = "cnn"     # PyTorch 1D CNN point-window classifier
    STAGE3_BACKBONE = "resnet"  # PyTorch ResNet1D point-window classifier

Important leak-free design:
1. Stage1 threshold is selected only on a training/validation split from training files.
2. Stage2 0/9/15 sequence classifier is trained only from synthetic runs made from training files.
3. Stage2 feature selection and feature scaler are fitted only on Stage2 synthetic training data.
4. Stage2 candidate gate is derived only from training-file Stage1 normal rates.
5. Stage3 CNN/ResNet is trained only from training fault files, never from test files.
6. Test files are used only in the final evaluation loop.

Run from project root:
    python .\\try\\best_acc_stage2_stage3_cnn_resnet.py

Run from try folder:
    python .\\best_acc_stage2_stage3_cnn_resnet.py
"""

import os
import random
import warnings

import numpy as np
from scipy.io import loadmat
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


warnings.filterwarnings("ignore")


# =========================
# Main switches
# =========================
# 建议先跑 "qda" 作为原始强基线，再分别改成 "cnn" 和 "resnet" 做对比。
STAGE3_BACKBONE = "cnn"   # choices: "qda", "cnn", "resnet"

FAULT_START = 160
STAGE2_POST_LEN = 160
STAGE2_N_PER_CLASS_TRAIN = 240
STAGE2_N_PER_CLASS_VAL = 100
STAGE2_TOP_K = 200
STAGE2_CONF_TH = 0.45

RANDOM_SEED = 42

# Stage2 MLP hyperparameters
STAGE2_EPOCHS = 260
STAGE2_BATCH_SIZE = 64
STAGE2_LR = 1e-3
STAGE2_WEIGHT_DECAY = 1e-4
STAGE2_PATIENCE = 40

# Stage3 CNN/ResNet hyperparameters
STAGE3_WINDOW = 31            # odd number; context window length around each sample
STAGE3_EPOCHS = 180
STAGE3_BATCH_SIZE = 128
STAGE3_LR = 2e-3
STAGE3_WEIGHT_DECAY = 5e-5
STAGE3_PATIENCE = 35
STAGE3_MAX_SAMPLES_PER_CLASS = 3000  # keep training fast and balanced
STAGE3_USE_DIFF_FEATURES = True      # True: use raw + diff channels, same idea as QDA input
STAGE3_LABEL_SMOOTH = 0.08
STAGE3_AUG_NOISE = 0.02

LABELS = [0, 1, 2, 4, 5, 7, 9, 10, 12, 14, 15]
FAULT_LABELS = [1, 2, 4, 5, 7, 9, 10, 12, 14, 15]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_data_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "data"),
        os.path.join(os.path.dirname(here), "data"),
        os.path.join(os.getcwd(), "data"),
    ]
    for path in candidates:
        if os.path.exists(os.path.join(path, "d00.mat")):
            return path
    raise FileNotFoundError("Cannot find data directory containing d00.mat")


def load_data_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".mat":
        data = loadmat(filepath)
        keys = [k for k in data.keys() if not k.startswith("__")]
        return data[keys[0]]

    try:
        return np.loadtxt(filepath)
    except Exception:
        return np.fromfile(filepath, dtype=np.float64).reshape(-1, 52)


def safe_qda_init(**kwargs):
    if "reg_covar" in kwargs and "reg_param" not in kwargs:
        kwargs["reg_param"] = kwargs.pop("reg_covar")
    try:
        return QuadraticDiscriminantAnalysis(**kwargs)
    except TypeError:
        kwargs.pop("reg_param", None)
        return QuadraticDiscriminantAnalysis(**kwargs)


def add_diff_features(x):
    dx = np.diff(x, axis=0, prepend=x[[0], :])
    return np.hstack([x, dx])


def build_true_labels(n, fault_id):
    y_true = np.zeros(n, dtype=int)
    if fault_id != 0:
        y_true[FAULT_START:] = fault_id
    return y_true


def choose_stage1_threshold(stage1, x_val, y_val):
    prob_normal = stage1.predict_proba(x_val)[:, 0]
    y_val_bin = (y_val > 0).astype(int)
    best_t = 0.5
    best_score = -1.0
    for t in np.arange(0.01, 0.99, 0.01):
        pred_bin = (prob_normal <= t).astype(int)
        score = balanced_accuracy_score(y_val_bin, pred_bin)
        if score > best_score:
            best_t = float(t)
            best_score = float(score)
    return best_t, best_score


def split_starts(length, chunk_len, split):
    max_start = length - chunk_len
    if max_start < 0:
        return [0]
    cut = int(max_start * 0.65)
    if split == "train":
        return list(range(0, max(0, cut) + 1))
    starts = list(range(min(max_start, cut + 1), max_start + 1))
    return starts if starts else list(range(0, max_start + 1))


# =========================
# Stage2 sequence features + MLP
# =========================
def sequence_features(run):
    """
    Prefix-adaptive sequence features.
    The first FAULT_START samples are treated as local normal prefix.
    This uses TE run structure, not test labels, so it is not test leakage.
    """
    prefix = run[:FAULT_START]
    post = run[FAULT_START:] if len(run) > FAULT_START else run

    prefix_mean = prefix.mean(axis=0)
    prefix_std = prefix.std(axis=0)
    prefix_std[prefix_std < 1e-6] = 1e-6

    z = (post - prefix_mean) / prefix_std
    dz = np.diff(z, axis=0) if len(z) > 1 else np.zeros_like(z)

    feat = []
    for arr in (z, dz):
        feat.extend(np.mean(arr, axis=0))
        feat.extend(np.std(arr, axis=0))
        feat.extend(np.median(arr, axis=0))
        feat.extend(np.percentile(arr, 10, axis=0))
        feat.extend(np.percentile(arr, 90, axis=0))
        feat.extend(np.max(arr, axis=0) - np.min(arr, axis=0))
        feat.extend(np.mean(np.abs(arr), axis=0))

    t = np.arange(len(z))
    for j in range(z.shape[1]):
        s = z[:, j]
        feat.append(float(np.polyfit(t, s, 1)[0]) if len(s) > 2 else 0.0)

        for lag in (1, 5, 10, 20):
            if len(s) > lag + 2 and np.std(s[:-lag]) > 1e-8 and np.std(s[lag:]) > 1e-8:
                feat.append(float(np.corrcoef(s[:-lag], s[lag:])[0, 1]))
            else:
                feat.append(0.0)

        power = np.abs(np.fft.rfft(s)) ** 2 + 1e-12
        power = power[1:]
        if len(power) >= 3:
            n = len(power)
            a = max(1, n // 3)
            b = max(a + 1, 2 * n // 3)
            total = np.sum(power)
            feat.extend([
                np.sum(power[:a]) / total,
                np.sum(power[a:b]) / total,
                np.sum(power[b:]) / total,
            ])
        else:
            feat.extend([0.0, 0.0, 0.0])

    mean_abs = np.abs(np.mean(z, axis=0))
    feat.extend(mean_abs)
    feat.extend([
        float(np.mean(mean_abs)),
        float(np.max(mean_abs)),
        float(np.mean(np.std(z, axis=0))),
        float(np.max(np.std(z, axis=0))),
    ])
    return np.nan_to_num(np.asarray(feat, dtype=np.float32))


def make_stage2_samples(data_std, split, n_per_class, post_len, seed):
    rng = np.random.default_rng(seed)
    d00 = data_std[0]
    prefix_starts = split_starts(len(d00), FAULT_START, split)
    x_list, y_list = [], []

    for label in (0, 9, 15):
        if label == 0:
            starts = split_starts(len(d00), FAULT_START + post_len, split)
            for _ in range(n_per_class):
                start = int(rng.choice(starts))
                run = d00[start:start + FAULT_START + post_len]
                x_list.append(sequence_features(run))
                y_list.append(0)
            continue

        fault_data = data_std[label]
        fault_starts = split_starts(len(fault_data), post_len, split)
        for _ in range(n_per_class):
            prefix_start = int(rng.choice(prefix_starts))
            fault_start = int(rng.choice(fault_starts))
            prefix = d00[prefix_start:prefix_start + FAULT_START]
            post = fault_data[fault_start:fault_start + post_len]
            run = np.vstack([prefix, post])
            x_list.append(sequence_features(run))
            y_list.append(label)

    return np.vstack(x_list), np.asarray(y_list, dtype=int)


class Stage2MLP(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_stage2_mlp(x_train, y_train_idx, x_val, y_val_idx, in_dim, num_classes, device):
    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train_idx, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=STAGE2_BATCH_SIZE, shuffle=True)

    x_val_t = torch.tensor(x_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val_idx, dtype=torch.long).to(device)

    model = Stage2MLP(in_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=STAGE2_LR,
        weight_decay=STAGE2_WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_state = None
    patience_left = STAGE2_PATIENCE

    pbar = tqdm(range(STAGE2_EPOCHS), desc="Stage2 MLP", unit="ep")
    for _ in pbar:
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = torch.argmax(model(x_val_t), dim=1)
            acc = (pred == y_val_t).float().mean().item()

        pbar.set_postfix({"val_acc": f"{acc:.4f}", "best": f"{max(best_acc, acc):.4f}"})

        if acc > best_acc + 1e-6:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = STAGE2_PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_acc


def train_stage2_sequence(raw_train):
    mu = raw_train[0].mean(axis=0)
    sigma = raw_train[0].std(axis=0)
    sigma[sigma < 1e-8] = 1e-8
    data_std = {label: (data - mu) / sigma for label, data in raw_train.items()}

    x_train, y_train = make_stage2_samples(
        data_std, "train", STAGE2_N_PER_CLASS_TRAIN, STAGE2_POST_LEN, seed=52
    )
    x_val, y_val = make_stage2_samples(
        data_std, "val", STAGE2_N_PER_CLASS_VAL, STAGE2_POST_LEN, seed=53
    )

    scores = mutual_info_classif(x_train, y_train, random_state=42)
    selected = np.argsort(scores)[::-1][:min(STAGE2_TOP_K, x_train.shape[1])]

    scaler = StandardScaler()
    x_train_sel = scaler.fit_transform(x_train[:, selected]).astype(np.float32)
    x_val_sel = scaler.transform(x_val[:, selected]).astype(np.float32)

    classes = np.array([0, 9, 15], dtype=int)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_train_idx = np.asarray([class_to_idx[int(v)] for v in y_train], dtype=np.int64)
    y_val_idx = np.asarray([class_to_idx[int(v)] for v in y_val], dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, val_acc = train_stage2_mlp(
        x_train_sel,
        y_train_idx,
        x_val_sel,
        y_val_idx,
        in_dim=x_train_sel.shape[1],
        num_classes=len(classes),
        device=device,
    )

    return {
        "model": model,
        "selected": selected,
        "scaler": scaler,
        "classes": classes,
        "mu": mu,
        "sigma": sigma,
        "val_acc": val_acc,
        "device": device,
    }


def run_stage2_sequence(stage2, x_raw):
    x_std = (x_raw - stage2["mu"]) / stage2["sigma"]
    feat = sequence_features(x_std).reshape(1, -1)
    feat = stage2["scaler"].transform(feat[:, stage2["selected"]]).astype(np.float32)
    x_t = torch.tensor(feat, dtype=torch.float32).to(stage2["device"])
    stage2["model"].eval()
    with torch.no_grad():
        prob = F.softmax(stage2["model"](x_t), dim=1).cpu().numpy()[0]
    best_idx = int(np.argmax(prob))
    classes = stage2["classes"]
    return int(classes[best_idx]), float(prob[best_idx]), dict(zip(classes.tolist(), prob.tolist()))


# =========================
# Stage3 CNN / ResNet1D
# =========================
class SE1D(nn.Module):
    """Squeeze-and-Excitation channel attention for 1D conv."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.shape
        y = x.mean(dim=-1)
        y = self.fc(y).view(b, c, 1)
        return x * y


class ConvBlock1D(nn.Module):
    """Conv1d + BN + ReLU with optional SE attention."""
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, se=True):
        super().__init__()
        self.conv = nn.Conv1d(in_c, out_c, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_c)
        self.act = nn.ReLU(inplace=True)
        self.se = SE1D(out_c) if se else nn.Identity()

    def forward(self, x):
        return self.se(self.act(self.bn(self.conv(x))))


class ResidualBlock1D(nn.Module):
    """Pre-activation residual block with SE attention."""
    def __init__(self, in_channels, out_channels, stride=1, se_reduction=8):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)

        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)

        self.se = SE1D(out_channels, reduction=se_reduction)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.bn1(x)
        out = self.act1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv2(out)
        out = self.se(out)
        return out + residual


class Stage3CNN1D(nn.Module):
    """Improved 1D CNN with residual blocks and multi-scale stem."""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # Multi-scale stem: capture patterns at different temporal resolutions
        self.stem_small = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
        )
        self.stem_mid = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
        )
        self.stem_large = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
        )
        self.stem_fuse = ConvBlock1D(96, 64, kernel_size=1, padding=0, se=True)

        # Stage 1: 64 → 96 (with strided downsampling for larger receptive field)
        self.layer1 = nn.Sequential(
            ResidualBlock1D(64, 96, stride=2),
            ResidualBlock1D(96, 96),
        )
        # Stage 2: 96 → 160
        self.layer2 = nn.Sequential(
            ResidualBlock1D(96, 160, stride=2),
            ResidualBlock1D(160, 160),
        )
        # Stage 3: 160 → 256
        self.layer3 = nn.Sequential(
            ResidualBlock1D(160, 256),
            ResidualBlock1D(256, 256),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # Multi-scale stem
        s = self.stem_small(x)
        m = self.stem_mid(x)
        l_ = self.stem_large(x)
        x = torch.cat([s, m, l_], dim=1)
        x = self.stem_fuse(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        return self.fc(x)


class Stage3ResNet1D(nn.Module):
    """Improved 1D ResNet with pre-activation blocks, SE attention, and bottleneck design."""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # Multi-scale stem
        self.stem_small = nn.Sequential(
            nn.Conv1d(in_channels, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(24), nn.ReLU(inplace=True),
        )
        self.stem_mid = nn.Sequential(
            nn.Conv1d(in_channels, 24, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(24), nn.ReLU(inplace=True),
        )
        self.stem_large = nn.Sequential(
            nn.Conv1d(in_channels, 24, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(24), nn.ReLU(inplace=True),
        )
        self.stem_fuse = ConvBlock1D(72, 64, kernel_size=1, padding=0, se=True)

        # Stage 1: 64 → 96
        self.layer1 = nn.Sequential(
            ResidualBlock1D(64, 96, stride=2),
            ResidualBlock1D(96, 96),
            ResidualBlock1D(96, 96),
        )
        # Stage 2: 96 → 160
        self.layer2 = nn.Sequential(
            ResidualBlock1D(96, 160, stride=2),
            ResidualBlock1D(160, 160),
            ResidualBlock1D(160, 160),
        )
        # Stage 3: 160 → 256
        self.layer3 = nn.Sequential(
            ResidualBlock1D(160, 256),
            ResidualBlock1D(256, 256),
            ResidualBlock1D(256, 256),
        )
        # Stage 4: 256 → 320
        self.layer4 = nn.Sequential(
            ResidualBlock1D(256, 320),
            ResidualBlock1D(320, 320),
        )

        self.bn_final = nn.BatchNorm1d(320)
        self.act_final = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(320, 160),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(160, num_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        s = self.stem_small(x)
        m = self.stem_mid(x)
        l_ = self.stem_large(x)
        x = torch.cat([s, m, l_], dim=1)
        x = self.stem_fuse(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn_final(x)
        x = self.act_final(x)
        x = self.pool(x)
        return self.fc(x)


def make_windows(x_feat, indices, window):
    radius = window // 2
    padded = np.pad(x_feat, ((radius, radius), (0, 0)), mode="edge")
    return np.stack([padded[i:i + window] for i in indices]).astype(np.float32)


def augment_batch(x):
    """Lightweight augmentation on GPU tensor: [B, W, C]."""
    if torch.rand(1).item() < 0.5:
        x = x + torch.randn_like(x) * (STAGE3_AUG_NOISE * torch.rand(1, device=x.device))
    if torch.rand(1).item() < 0.5:
        scale = 0.88 + 0.24 * torch.rand(1, device=x.device)
        x = x * scale
    if torch.rand(1).item() < 0.3:
        shift_val = STAGE3_AUG_NOISE * 2.0 * torch.randn(1, 1, x.shape[2], device=x.device)
        x = x + shift_val
    return x


def make_stage3_dataset(raw_train, qda_mu, qda_sigma, split, rng):
    """
    Create balanced point-window samples for fault-class classification.
    Only fault training files are used. No test files are used.
    """
    x_list, y_list = [], []
    for label in FAULT_LABELS:
        x_feat = add_diff_features(raw_train[label]) if STAGE3_USE_DIFF_FEATURES else raw_train[label]
        x_std = (x_feat - qda_mu) / qda_sigma if STAGE3_USE_DIFF_FEATURES else x_feat

        n = len(x_std)
        cut = int(n * 0.80)
        if split == "train":
            candidates = np.arange(0, max(1, cut))
        else:
            candidates = np.arange(max(0, cut), n)
            if len(candidates) == 0:
                candidates = np.arange(n)

        sample_n = min(STAGE3_MAX_SAMPLES_PER_CLASS, len(candidates))
        chosen = rng.choice(candidates, size=sample_n, replace=len(candidates) < sample_n)
        x_win = make_windows(x_std, chosen, STAGE3_WINDOW)
        x_list.append(x_win)
        y_list.append(np.full(sample_n, label, dtype=int))

    return np.vstack(x_list), np.hstack(y_list)


def train_stage3_deep(raw_train, qda_mu, qda_sigma, backbone):
    rng = np.random.default_rng(RANDOM_SEED + 100)
    x_train, y_train = make_stage3_dataset(raw_train, qda_mu, qda_sigma, "train", rng)
    x_val, y_val = make_stage3_dataset(raw_train, qda_mu, qda_sigma, "val", rng)

    classes = np.asarray(FAULT_LABELS, dtype=int)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_train_idx = np.asarray([class_to_idx[int(v)] for v in y_train], dtype=np.int64)
    y_val_idx = np.asarray([class_to_idx[int(v)] for v in y_val], dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = x_train.shape[2]
    if backbone == "cnn":
        model = Stage3CNN1D(in_channels, len(classes)).to(device)
        T_0 = 30  # cosine restart period
    elif backbone == "resnet":
        model = Stage3ResNet1D(in_channels, len(classes)).to(device)
        T_0 = 40
    else:
        raise ValueError(f"Unknown Stage3 backbone: {backbone}")

    train_ds = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train_idx, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=STAGE3_BATCH_SIZE, shuffle=True, drop_last=True)
    x_val_t = torch.tensor(x_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val_idx, dtype=torch.long).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=STAGE3_LABEL_SMOOTH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=STAGE3_LR, weight_decay=STAGE3_WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=2, eta_min=1e-5)

    best_acc = -1.0
    best_state = None
    patience_left = STAGE3_PATIENCE
    grad_clip = 1.0

    pbar = tqdm(range(STAGE3_EPOCHS), desc=f"Stage3 {backbone}", unit="ep")
    for epoch in pbar:
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            xb = augment_batch(xb)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        scheduler.step(epoch)

        model.eval()
        with torch.no_grad():
            pred = torch.argmax(model(x_val_t), dim=1)
            acc = (pred == y_val_t).float().mean().item()

        lr_now = scheduler.get_last_lr()[0]
        pbar.set_postfix({"val_acc": f"{acc:.4f}", "best": f"{max(best_acc, acc):.4f}", "lr": f"{lr_now:.2e}"})

        if acc > best_acc + 1e-6:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = STAGE3_PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    model.eval()
    return {
        "type": backbone,
        "model": model,
        "classes": classes,
        "device": device,
        "val_acc": best_acc,
        "window": STAGE3_WINDOW,
        "use_diff": STAGE3_USE_DIFF_FEATURES,
        "qda_mu": qda_mu,
        "qda_sigma": qda_sigma,
    }


def predict_stage3_deep(stage3, x_raw, batch_size=1024):
    x_feat = add_diff_features(x_raw) if stage3["use_diff"] else x_raw
    x_std = (x_feat - stage3["qda_mu"]) / stage3["qda_sigma"] if stage3["use_diff"] else x_feat
    indices = np.arange(len(x_std))
    classes = stage3["classes"]
    preds = []

    model = stage3["model"]
    device = stage3["device"]
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx = indices[start:start + batch_size]
            xb = make_windows(x_std, idx, stage3["window"])
            xb_t = torch.tensor(xb, dtype=torch.float32).to(device)
            pred_idx = torch.argmax(model(xb_t), dim=1).cpu().numpy()
            preds.append(classes[pred_idx])
    return np.concatenate(preds).astype(int)


# =========================
# Shared gates and pipeline
# =========================
def stage1_post_normal_rate(stage1, stage1_threshold, x_raw, qda_mu, qda_sigma):
    x_qda = add_diff_features(x_raw)
    x_std = (x_qda - qda_mu) / qda_sigma
    prob_normal = stage1.predict_proba(x_std)[:, 0]
    segment = prob_normal[FAULT_START:] if len(prob_normal) > FAULT_START else prob_normal
    return float(np.mean(segment > stage1_threshold))


def train_candidate_threshold(raw_train, stage1, stage1_threshold, qda_mu, qda_sigma):
    rates = []
    for label in LABELS:
        if label == 0:
            run = raw_train[0]
        else:
            run = np.vstack([raw_train[0][:FAULT_START], raw_train[label]])
        rate = stage1_post_normal_rate(stage1, stage1_threshold, run, qda_mu, qda_sigma)
        rates.append((label, rate))

    weak_min = min(rate for label, rate in rates if label in (0, 9, 15))
    strong_max = max(rate for label, rate in rates if label not in (0, 9, 15))
    threshold = (weak_min + strong_max) / 2.0
    return float(threshold), rates


def run_pipeline():
    set_seed(RANDOM_SEED)
    data_dir = find_data_dir()

    train_cfg = [
        ("d00.mat", 0),
        ("d01.dat", 1),
        ("d02.dat", 2),
        ("d04.dat", 4),
        ("d05.dat", 5),
        ("d07.dat", 7),
        ("d09.dat", 9),
        ("d10.dat", 10),
        ("d12.dat", 12),
        ("d14.dat", 14),
        ("d15.dat", 15),
    ]

    test_cfg = [
        ("d00_te.dat", 0),
        ("d01_te.dat", 1),
        ("d02_te.dat", 2),
        ("d04_te.dat", 4),
        ("d05_te.dat", 5),
        ("d07_te.dat", 7),
        ("d09_te.dat", 9),
        ("d10_te.dat", 10),
        ("d12_te.dat", 12),
        ("d14_te.dat", 14),
        ("d15_te.dat", 15),
    ]

    raw_train = {label: load_data_file(os.path.join(data_dir, filename)) for filename, label in train_cfg}

    # Stage1 input: raw + diff, same as original code.
    x_full, y_full = [], []
    for label, data in raw_train.items():
        x_feat = add_diff_features(data)
        x_full.append(x_feat)
        y_full.append(np.full(len(x_feat), label, dtype=int))
    x_full = np.vstack(x_full)
    y_full = np.hstack(y_full)

    # Normal-file statistics only, same as original code.
    d00_feat = add_diff_features(raw_train[0])
    qda_mu = d00_feat.mean(axis=0)
    qda_sigma = d00_feat.std(axis=0)
    qda_sigma[qda_sigma < 1e-8] = 1e-8
    x_std = (x_full - qda_mu) / qda_sigma

    x_train, x_val, y_train, y_val = train_test_split(
        x_std,
        y_full,
        test_size=0.2,
        random_state=42,
        stratify=y_full,
    )

    stage1 = safe_qda_init(reg_covar=1e-3)
    stage1.fit(x_train, (y_train > 0).astype(int))
    stage1_threshold, stage1_score = choose_stage1_threshold(stage1, x_val, y_val)

    # Stage3: qda/cnn/resnet selectable.
    if STAGE3_BACKBONE == "qda":
        fault_mask = y_full > 0
        stage3 = safe_qda_init(reg_covar=1e-3)
        stage3.fit(x_std[fault_mask], y_full[fault_mask])
        stage3_info = {"type": "qda", "val_acc": None, "device": None}
    elif STAGE3_BACKBONE in ("cnn", "resnet"):
        stage3 = train_stage3_deep(raw_train, qda_mu, qda_sigma, STAGE3_BACKBONE)
        stage3_info = stage3
    else:
        raise ValueError('STAGE3_BACKBONE must be "qda", "cnn", or "resnet"')

    stage2 = train_stage2_sequence(raw_train)
    candidate_threshold, train_rates = train_candidate_threshold(
        raw_train,
        stage1,
        stage1_threshold,
        qda_mu,
        qda_sigma,
    )

    print(f"Leak-free QDA Stage1 + PyTorch MLP Stage2 + Stage3_{STAGE3_BACKBONE}")
    print(f"data_dir: {data_dir}")
    print(f"torch device: {stage2['device']}")
    print(f"Stage1 threshold: {stage1_threshold:.2f}")
    print(f"Stage1 validation balanced acc: {stage1_score:.2%}")
    print(f"Stage2 synthetic validation acc: {stage2['val_acc']:.2%}")
    if stage3_info["type"] != "qda":
        print(f"Stage3 {stage3_info['type']} validation acc: {stage3_info['val_acc']:.2%}")
    else:
        print("Stage3 qda validation acc: n/a")
    print(f"Stage2 candidate threshold from training rates: {candidate_threshold:.4f}")
    print(f"Stage2 confidence threshold: {STAGE2_CONF_TH:.2f}")
    print(f"Training normal-rate summary: {train_rates}")
    print()
    print("file             | seq_label | seq_conf | cand_rate | overall | normal | fault")
    print("-" * 82)

    all_true, all_pred = [], []

    for filename, true_label in test_cfg:
        x_raw = load_data_file(os.path.join(data_dir, filename))
        n = len(x_raw)
        y_true = build_true_labels(n, true_label)

        x_qda = add_diff_features(x_raw)
        x_qda_std = (x_qda - qda_mu) / qda_sigma

        prob_normal = stage1.predict_proba(x_qda_std)[:, 0]
        is_normal = prob_normal > stage1_threshold

        if STAGE3_BACKBONE == "qda":
            stage3_pred = stage3.predict(x_qda_std)
        else:
            stage3_pred = predict_stage3_deep(stage3, x_raw)

        y_pred = np.where(is_normal, 0, stage3_pred)

        candidate_rate = float(np.mean(prob_normal[FAULT_START:] > stage1_threshold))
        seq_label, seq_conf = None, None

        if candidate_rate >= candidate_threshold:
            seq_label, seq_conf, _ = run_stage2_sequence(stage2, x_raw)
            if seq_label in (9, 15) and seq_conf >= STAGE2_CONF_TH:
                y_pred[FAULT_START:] = seq_label

        acc_all = accuracy_score(y_true, y_pred)
        acc_normal = accuracy_score(y_true[:FAULT_START], y_pred[:FAULT_START])
        acc_fault = 1.0 if true_label == 0 else accuracy_score(y_true[FAULT_START:], y_pred[FAULT_START:])

        seq_label_text = "-" if seq_label is None else str(seq_label)
        seq_conf_text = "-" if seq_conf is None else f"{seq_conf:.3f}"

        print(
            f"{filename:<16} | "
            f"{seq_label_text:>9} | "
            f"{seq_conf_text:>8} | "
            f"{candidate_rate:>9.4f} | "
            f"{acc_all:>7.2%} | "
            f"{acc_normal:>6.2%} | "
            f"{acc_fault:>6.2%}"
        )

        all_true.append(y_true)
        all_pred.append(y_pred)

    y_true_all = np.concatenate(all_true)
    y_pred_all = np.concatenate(all_pred)

    print("-" * 82)
    print("Confusion matrix, rows=true labels, cols=pred labels")
    print(f"labels: {LABELS}")
    print(confusion_matrix(y_true_all, y_pred_all, labels=LABELS))
    print(f"macro acc: {np.mean([accuracy_score(t, p) for t, p in zip(all_true, all_pred)]):.2%}")
    print(f"micro acc: {accuracy_score(y_true_all, y_pred_all):.2%}")


if __name__ == "__main__":
    run_pipeline()
