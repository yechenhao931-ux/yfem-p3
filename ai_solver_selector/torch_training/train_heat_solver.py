"""
train_heat_solver.py
====================
用 PyTorch MLP 训练热传导问题的最优求解器选择器。
 
数据源：heat_benchmark.cpp 生成的 CSV
  - heat_results.csv  (所有求解器记录)
  - heat_best.csv     (每组最优求解器，作为训练标签)
 
网络：残差 MLP
  Input → Linear-BN-ReLU → ResBlock × N → Head(softmax)
 
输出：
  - solver_heat_model.json  (C++ 推理所需，权重扁平化)
  - solver_heat_model.pth   (PyTorch 继续微调用)
  - training_curve.png      (训练曲线)
  - confusion_matrix.png    (测试集混淆矩阵)
 
依赖：
  pip install torch numpy pandas scikit-learn matplotlib
 
运行：
  python train_heat_solver.py \
      --data   ../data/heat_best.csv \
      --output ../models/ \
      --epochs 200
"""
 
 
import argparse
import json
import time
from pathlib import Path
 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
# ──────────────────────────────────────────────────────────────
# CSV 列定义（必须与 heat_benchmark.cpp 的 WriteCSVHeader 完全对齐）
# ──────────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "dim", "ref_level", "poly_order", "n_elements",
    "n_dof", "nnz", "nnz_per_row",
    "asymmetry_rel", "asymmetry_abs", "matrix_norm_inf",
    "is_spd", "diag_dominance", "log10_cond_estimate",
    "k_ratio", "peclet", "material_contrast",
]

CATEGORICAL_FEATURES = ["mesh_type", "problem"]   # 通过 one-hot 嵌入

TARGET = "solver"

# 最终喂给神经网络的特征列表
ENHANCED_FEATURES = NUMERIC_FEATURES + [
    "log_n_dof", "log_n_elements", "log_nnz",
    "log_k_ratio", "log_peclet", "log_mat_contrast", "log_norm_inf",
    "is_3d", "is_high_order", "is_anisotropic",
    "is_convection", "is_heterogeneous", "is_large",
    "complexity", "sym_score",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    将原始 CSV 中的列转换为适合神经网络的特征：
      - 对数化大范围数值（DOF、nnz、k_ratio 等）
      - 派生特征（是否 3D、是否高阶、是否各向异性强等）
 
    注意：`np.maximum(x, eps)` 是**逐元素取较大值**，只在 x <= eps 时
    将其替换为 eps（防止 log10(0) = -inf）；不会影响正常值。
    例如：
        np.maximum([1000, 0, 50], 1)  →  [1000, 1, 50]
    """
    df = df.copy()
 
    # log10 变换（先 clip 避免 log(0)）
    df["log_n_dof"]        = np.log10(np.maximum(df["n_dof"],       1))
    df["log_n_elements"]   = np.log10(np.maximum(df["n_elements"],  1))
    df["log_nnz"]          = np.log10(np.maximum(df["nnz"],         1))
    df["log_k_ratio"]      = np.log10(np.maximum(df["k_ratio"],     1e-6))
    df["log_peclet"]       = np.log10(np.maximum(df["peclet"],      1e-6))
    df["log_mat_contrast"] = np.log10(np.maximum(df["material_contrast"], 1e-6))
    df["log_norm_inf"]     = np.log10(np.maximum(df["matrix_norm_inf"],   1e-6))
 
    # 派生分类特征
    df["is_3d"]            = (df["dim"] == 3).astype(np.float32)
    df["is_high_order"]    = (df["poly_order"] > 1).astype(np.float32)
    df["is_anisotropic"]   = (df["k_ratio"] > 5).astype(np.float32)
    df["is_convection"]    = (df["peclet"] > 1).astype(np.float32)
    df["is_heterogeneous"] = (df["material_contrast"] > 10).astype(np.float32)
    df["is_large"]         = (df["log_n_dof"] > 4.5).astype(np.float32)
 
    # 复杂度代理
    df["complexity"]       = df["log_n_dof"] + df["log10_cond_estimate"]
    df["sym_score"]        = 1.0 - df["asymmetry_rel"]   # 反转：1=对称
 
    return df

# ──────────────────────────────────────────────────────────────
# Scaler 工厂：支持 standard / minmax / robust
# ──────────────────────────────────────────────────────────────
def make_scaler(kind: str):
    """
    归一化策略说明：
 
    standard (默认): Z-score, (x - μ) / σ
        - 对 MLP + BatchNorm 最稳健，均值 0 方差 1
        - 对异常值敏感（单个极端值会拉偏 μ, σ）
 
    minmax:  Min-Max [0, 1] = (x - min) / (max - min)
        - 所有值压到 [0, 1]，与 one-hot 同量纲
        - 对异常值非常敏感（训练集一个异常大值就把其他值压到接近 0）
        - 必须先 log 变换再做 Min-Max（否则跨 6 个数量级的特征会全变 0/1）
        - 推理时对训练范围外的数据 clip 到 [0, 1]
 
    robust:  (x - median) / IQR
        - 对异常值稳健（50% 数据点）
        - 跨数量级时表现不如 Min-Max
    """
    kind = kind.lower()
    if kind == "standard":
        return StandardScaler()
    if kind == "minmax":
        return MinMaxScaler(feature_range=(0.0, 1.0), clip=True)
    if kind == "robust":
        return RobustScaler()
    raise ValueError(f"未知 scaler 类型: {kind}. 支持: standard, minmax, robust")
 
def prepare_data(csv_path: str, test_size: float = 0.2, seed: int = 42,
                  scaler_kind: str = "standard"):
    """
    读取 CSV → 过滤收敛失败的行 → 特征工程 → one-hot 编码分类特征
    → 归一化数值特征 → 划分训练/测试集。
    """
    df = pd.read_csv(csv_path)
    print(f"[数据] 读取 {len(df)} 条记录")

    # 仅保留收敛的最优求解器记录
    if "converged" in df.columns:
        df = df[df["converged"] == 1].copy()
        print(f"[数据] 筛除未收敛样本后: {len(df)} 条")
  
    # 若用户给的是 heat_results.csv (全量)，自动取 is_best
    if "is_best" in df.columns and df["is_best"].sum() > 0:
        df = df[df["is_best"] == 1].copy()
        print(f"[数据] 仅保留 is_best=1 的记录: {len(df)} 条")
    
    # 特征工程
    df = engineer_features(df)

    # Problem 类型提取主类（忽略超参后缀 k10/Pe100 等）
    df["problem_family"] = df["problem"].str.split("_").str[0]
    print(f"[数据] 物理类型分布:\n{df['problem_family'].value_counts().to_string()}")
    print(f"[数据] 网格类型分布:\n{df['mesh_type'].value_counts().to_string()}")
    print(f"[数据] 最优求解器分布:\n{df[TARGET].value_counts().to_string()}")

    # One-hot 编码 mesh_type 和 problem_family
    df_cat = pd.get_dummies(df[["mesh_type", "problem_family"]],
                             prefix=["mesh", "phys"])
    cat_cols = df_cat.columns.tolist()
 
    # 组合数值 + one-hot 特征
    X_num = df[ENHANCED_FEATURES].values.astype(np.float32)
    X_cat = df_cat.values.astype(np.float32)
    X = np.concatenate([X_num, X_cat], axis=1)

    all_feature_names = ENHANCED_FEATURES + cat_cols
    print(f"[数据] 最终特征维度: {X.shape[1]} "
          f"(数值 {len(ENHANCED_FEATURES)} + 分类 {len(cat_cols)})")
    
    # 标签编码
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET].values)
    class_names = le.classes_.tolist()
    print(f"[数据] 类别: {class_names}")
    print(y)
    # 划分训练 / 测试
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # ─────────────────────────────────────────────────
    # 归一化（仅对数值列；one-hot 本身已是 0/1）
    # ─────────────────────────────────────────────────
    scaler = make_scaler(scaler_kind)
    n_num = len(ENHANCED_FEATURES)
     # 归一化前做诊断（帮助发现量纲问题）
    print(f"\n[归一化] 策略: {scaler_kind}")
    print(f"  数值特征数: {n_num}   one-hot 特征数: {len(cat_cols)}")
    print(f"  归一化前数值特征统计（前 5 列）:")
    for i in range(min(5, n_num)):
        col = X_tr[:, i]
        print(f"    {ENHANCED_FEATURES[i]:22s}  "
              f"min={col.min():10.3e}  max={col.max():10.3e}  "
              f"range={col.max()-col.min():10.3e}")
 
    X_tr[:, :n_num] = scaler.fit_transform(X_tr[:, :n_num])
    X_te[:, :n_num] = scaler.transform(X_te[:, :n_num])
 
    print(f"  归一化后数值特征统计（前 5 列）:")
    for i in range(min(5, n_num)):
        col = X_tr[:, i]
        print(f"    {ENHANCED_FEATURES[i]:22s}  "
              f"min={col.min():10.3e}  max={col.max():10.3e}  "
              f"mean={col.mean():10.3e}  std={col.std():10.3e}")
 
    return {
        "X_train": X_tr, "X_test": X_te,
        "y_train": y_tr, "y_test": y_te,
        "scaler": scaler,
        "feature_names": all_feature_names,
        "class_names": class_names,
        "n_numeric": n_num,
        "raw_df": df,
    }

# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────
class HeatDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]
 
 
# ──────────────────────────────────────────────────────────────
# 残差 MLP（与 mlp_inference.hpp 完全匹配的结构）
# ──────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.15):
        super().__init__()
        self.fc   = nn.Linear(in_dim, out_dim)
        self.bn   = nn.BatchNorm1d(out_dim)
        self.act  = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.proj = (nn.Linear(in_dim, out_dim, bias=False)
                     if in_dim != out_dim else nn.Identity())
 
    def forward(self, x):
        # 主路径：Linear → BN → Dropout → 加残差 → ReLU
        h = self.drop(self.bn(self.fc(x)))
        skip = self.proj(x)
        return self.act(h + skip)
 
 
class HeatSolverNet(nn.Module):
    """
    Input → ResBlock(256) → ResBlock(128) → ResBlock(64) → Linear(n_classes)
    """
    def __init__(self, n_features: int, n_classes: int,
                 hidden: list = None, dropout: float = 0.15):
        super().__init__()
        if hidden is None:
            hidden = [256, 128, 64]
 
        blocks = []
        in_dim = n_features
        for h in hidden:
            blocks.append(ResBlock(in_dim, h, dropout))
            in_dim = h
 
        self.backbone = nn.Sequential(*blocks)
        self.head     = nn.Linear(in_dim, n_classes)
 
        # Kaiming 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
 
    def forward(self, x):
        return self.head(self.backbone(x))
 
 
# ──────────────────────────────────────────────────────────────
# 训练辅助
# ──────────────────────────────────────────────────────────────
class LabelSmoothingCE(nn.Module):
    def __init__(self, n_classes: int, smoothing: float = 0.05):
        super().__init__()
        self.smoothing = smoothing
        self.n_classes = n_classes
 
    def forward(self, logits, y):
        logp = torch.log_softmax(logits, dim=-1)
        sm   = self.smoothing / (self.n_classes - 1)
        soft = torch.full_like(logp, sm)
        soft.scatter_(1, y.unsqueeze(1), 1.0 - self.smoothing)
        return -(soft * logp).sum(dim=-1).mean()
 
 
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        loss_sum += loss.item() * len(y)
        correct  += (out.argmax(1) == y).sum().item()
        total    += len(y)
    return loss_sum / total, correct / total
 
 
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out = model(X)
        loss_sum += criterion(out, y).item() * len(y)
        correct  += (out.argmax(1) == y).sum().item()
        total    += len(y)
    return loss_sum / total, correct / total
 
# ──────────────────────────────────────────────────────────────
# 模型导出为 JSON（供 C++ 端 mlp_inference.hpp 加载）
# ──────────────────────────────────────────────────────────────
def export_model_json(model: HeatSolverNet,
                       scaler,
                       feature_names: list,
                       class_names: list,
                       n_numeric: int,
                       accuracy: float,
                       scaler_kind: str,
                       out_path: Path):
    """
    导出为 C++ mlp_inference.hpp 可读的 JSON 格式。
 
    统一归一化约定（C++ 端通用）：
        x_scaled = clip((x - offset) / scale, lo, hi)
 
    各种 scaler 的映射关系：
      - StandardScaler: offset = mean,   scale = stddev
      - MinMaxScaler:   offset = data_min, scale = data_max - data_min
      - RobustScaler:   offset = median,   scale = IQR
 
    字段说明：
      - offset/scale        : 长度 = n_features，前 n_numeric 个为真实值，
                              后面的 one-hot 列为 (0, 1) 恒等变换
      - scaler_kind         : "standard" | "minmax" | "robust"
      - clip_lo / clip_hi   : 推理时 clip 的上下界（MinMax 时为 [0, 1]，否则 null）
    """
    def to_list(t: torch.Tensor) -> list:
        return t.detach().float().cpu().numpy().tolist()
 
    total_feat = len(feature_names)
    offset = np.zeros(total_feat, dtype=np.float64)
    scale  = np.ones(total_feat, dtype=np.float64)
 
    # 根据 scaler 类型提取 (offset, scale)
    if isinstance(scaler, MinMaxScaler):
        data_min = scaler.data_min_
        data_max = scaler.data_max_
        offset[:n_numeric] = data_min
        scale[:n_numeric]  = data_max - data_min
    elif isinstance(scaler, StandardScaler):
        offset[:n_numeric] = scaler.mean_
        scale[:n_numeric]  = scaler.scale_
    elif isinstance(scaler, RobustScaler):
        offset[:n_numeric] = scaler.center_
        scale[:n_numeric]  = scaler.scale_
    else:
        raise ValueError(f"未支持的 scaler: {type(scaler)}")
 
    # 防零除：任何 scale[i] ≈ 0 的列（常数列）scale 置 1，offset 置 0
    near_zero = np.abs(scale) < 1e-12
    if near_zero.any():
        bad = np.where(near_zero)[0]
        bad_names = [feature_names[i] for i in bad]
        print(f"[警告] 发现常数列 (scale≈0)，自动置为 (offset=0, scale=1): {bad_names}")
        offset[near_zero] = 0.0
        scale [near_zero] = 1.0
 
    # MinMax 推理时 clip 到 [0, 1]，其他保留
    if scaler_kind == "minmax":
        clip_lo, clip_hi = 0.0, 1.0
    else:
        clip_lo, clip_hi = None, None
 
    # 每个 ResBlock 的参数
    layers = []
    for block in model.backbone:
        layer = {
            "type":            "resblock",
            "in_dim":          block.fc.in_features,
            "out_dim":         block.fc.out_features,
            "fc_weight":       to_list(block.fc.weight),
            "fc_bias":         to_list(block.fc.bias),
            "bn_weight":       to_list(block.bn.weight),
            "bn_bias":         to_list(block.bn.bias),
            "bn_running_mean": to_list(block.bn.running_mean),
            "bn_running_var":  to_list(block.bn.running_var),
            "bn_eps":          float(block.bn.eps),
        }
        if not isinstance(block.proj, nn.Identity):
            layer["proj_weight"] = to_list(block.proj.weight)
        layers.append(layer)
 
    model_dict = {
        "model_type":    "MLP_ResNet",
        "task":          "heat_solver_selection",
        "accuracy":      round(float(accuracy), 6),
        "n_features":    total_feat,
        "n_classes":     len(class_names),
        "n_numeric":     n_numeric,
        "feature_names": feature_names,
        "class_names":   class_names,
        "scaler": {
            "kind":    scaler_kind,
            # C++ 端统一公式：x_scaled = clip((x - mean) / scale, clip_lo, clip_hi)
            # 所以 standard/minmax/robust 都映射到同一个字段名
            "mean":    offset.tolist(),
            "scale":   scale.tolist(),
            "clip_lo": clip_lo,
            "clip_hi": clip_hi,
        },
        "layers": layers,
        "head": {
            "weight": to_list(model.head.weight),
            "bias":   to_list(model.head.bias),
        },
    }
 
    with open(out_path, "w") as f:
        json.dump(model_dict, f, separators=(",", ":"))
 
    size_kb = out_path.stat().st_size / 1024
    print(f"[导出] {out_path}  ({size_kb:.1f} KB)  [{scaler_kind}]")

# ──────────────────────────────────────────────────────────────
# 可视化
# ──────────────────────────────────────────────────────────────
def plot_curves(tr_losses, te_losses, tr_accs, te_accs, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
 
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
        ep = range(1, len(tr_losses) + 1)
        a1.plot(ep, tr_losses, label="Train",      color="#4f7cff")
        a1.plot(ep, te_losses, label="Validation", color="#ff6b6b")
        a1.set(xlabel="Epoch", ylabel="Loss", title="Loss")
        a1.legend(); a1.grid(alpha=0.3)
 
        a2.plot(ep, [a*100 for a in tr_accs], label="Train",      color="#4f7cff")
        a2.plot(ep, [a*100 for a in te_accs], label="Validation", color="#00e5c3")
        a2.set(xlabel="Epoch", ylabel="Accuracy (%)", title="Accuracy")
        a2.legend(); a2.grid(alpha=0.3)
 
        plt.tight_layout()
        path = out_dir / "training_curve.png"
        plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()
        print(f"[曲线] {path}")
    except ImportError:
        print("[提示] matplotlib 未安装，跳过曲线绘制")

def plot_confusion(y_true, y_pred, class_names, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
 
        cm = confusion_matrix(y_true, y_pred)
        cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
 
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm_n, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title("Confusion Matrix (row-normalized)")
 
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                txt = f"{cm[i, j]}"
                color = "white" if cm_n[i, j] > 0.5 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        color=color, fontsize=9)
 
        plt.colorbar(im); plt.tight_layout()
        path = out_dir / "confusion_matrix.png"
        plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()
        print(f"[混淆矩阵] {path}")
    except ImportError:
        pass
 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="/home/ych/yfem-p3/ai_solver_selector/data_collector/heat_results.csv")
    parser.add_argument("--output",     default="models/")
    parser.add_argument("--epochs",     type=int,   default=300)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout",    type=float, default=0.15)
    parser.add_argument("--hidden",     nargs="+", type=int,
                         default=[256, 128, 64])
    parser.add_argument("--patience",   type=int,   default=40)
    parser.add_argument("--test_size",  type=float, default=0.2)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--scaler",     default="standard",
                         choices=["standard", "minmax", "robust"],
                         help="归一化策略 (默认 standard)")
    parser.add_argument("--no_cuda",    action="store_true")
    args = parser.parse_args()
 
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(
        "cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu"
    )
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
 
    print("=" * 62)
    print("  热传导求解器选择器 — PyTorch 训练")
    print("=" * 62)
    print(f"  设备: {device}")
 
 
    # 1. 准备数据
    data = prepare_data(args.data, args.test_size, args.seed,
                         scaler_kind=args.scaler)
    n_features = data["X_train"].shape[1]
    n_classes  = len(data["class_names"])
    print(f"\n[网络] {n_features} → {args.hidden} → {n_classes}")
 
    # 2. 类别平衡采样
    class_counts = np.bincount(data["y_train"])
    w = 1.0 / class_counts[data["y_train"]]
    sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
 
    tr_loader = DataLoader(
        HeatDataset(data["X_train"], data["y_train"]),
        batch_size=args.batch_size, sampler=sampler, num_workers=0
    )
    te_loader = DataLoader(
        HeatDataset(data["X_test"], data["y_test"]),
        batch_size=256, shuffle=False, num_workers=0
    )
    # 3. 模型 + 优化器
    model = HeatSolverNet(n_features, n_classes, args.hidden, args.dropout)
    model.to(device)
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[模型] 参数量: {n_p:,}")
 
    criterion = LabelSmoothingCE(n_classes, smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
 
    # 4. 训练循环
    best_acc, best_state, patience_ctr = 0.0, None, 0
    tr_losses, te_losses, tr_accs, te_accs = [], [], [], []
 
    print(f"\n{'Epoch':>6} {'TrLoss':>9} {'TrAcc':>8} "
          f"{'VlLoss':>9} {'VlAcc':>7} {'LR':>9} {'Time':>6}")
    print("  " + "─" * 60)
 
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tl, ta = train_one_epoch(model, tr_loader, criterion, optimizer, device)
        vl, va = evaluate(model, te_loader, criterion, device)
        scheduler.step()
        dt = time.time() - t0
 
        tr_losses.append(tl); te_losses.append(vl)
        tr_accs.append(ta);   te_accs.append(va)
 
        if va > best_acc:
            best_acc = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
            mark = " ◀"
        else:
            patience_ctr += 1
            mark = ""
 
        if ep % 10 == 0 or ep == 1 or ep == args.epochs or mark:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"{ep:>6d} {tl:>9.4f} {ta*100:>7.2f}% "
                  f"{vl:>9.4f} {va*100:>6.2f}% {lr_now:>9.2e} "
                  f"{dt:>5.2f}s{mark}")
 
        if patience_ctr >= args.patience:
            print(f"\n  [早停] Epoch {ep}, 最优 Val Acc = {best_acc*100:.2f}%")
            break
 
    model.load_state_dict(best_state)
 
    # 5. 最终评估
    print("\n" + "=" * 62)
    print("  最终评估")
    print("=" * 62)
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for X, y in te_loader:
            out = model(X.to(device))
            all_pred.extend(out.argmax(1).cpu().numpy())
            all_true.extend(y.numpy())
    all_pred = np.array(all_pred); all_true = np.array(all_true)
    final_acc = (all_pred == all_true).mean()
 
    classes_present = sorted(np.unique(all_true).tolist())
    target_names    = [data["class_names"][c] for c in classes_present]
 
    print(f"  测试准确率: {final_acc*100:.2f}%")
    print("\n  分类报告:\n")
    print(classification_report(all_true, all_pred,
                                 labels=classes_present,
                                 target_names=target_names,
                                 zero_division=0))
 
    # 6. 导出模型
    print("\n" + "=" * 62)
    print("  导出模型")
    print("=" * 62)
    export_model_json(
        model, data["scaler"], data["feature_names"],
        data["class_names"], data["n_numeric"], final_acc,
        args.scaler,
        out_dir / "solver_heat_model.json"
    )
    # PyTorch checkpoint: 保存 sklearn scaler 对象本身（可通过 pickle 复用）
    import pickle
    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(data["scaler"], f)
    torch.save({
        "state_dict":    model.state_dict(),
        "feature_names": data["feature_names"],
        "class_names":   data["class_names"],
        "scaler_kind":   args.scaler,
        "hidden":        args.hidden,
        "dropout":       args.dropout,
        "n_numeric":     data["n_numeric"],
        "accuracy":      final_acc,
    }, out_dir / "solver_heat_model.pth")
    print(f"[导出] {out_dir / 'solver_heat_model.pth'}")
 
    # 7. 可视化
    plot_curves(tr_losses, te_losses, tr_accs, te_accs, out_dir)
    plot_confusion(all_true, all_pred, data["class_names"], out_dir)
 
    # 8. 训练摘要
    summary = {
        "accuracy":        float(final_acc),
        "best_val_acc":    float(best_acc),
        "n_features":      n_features,
        "n_classes":       n_classes,
        "class_names":     data["class_names"],
        "feature_names":   data["feature_names"],
        "epochs_trained":  len(tr_losses),
        "hyperparams": {
            "hidden":       args.hidden,
            "dropout":      args.dropout,
            "lr":           args.lr,
            "weight_decay": args.weight_decay,
            "batch_size":   args.batch_size,
        },
    }
    with open(out_dir / "heat_train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
 
    print(f"\n✅ 完成 — 测试准确率 {final_acc*100:.2f}%")
    print(f"   模型: {out_dir / 'solver_heat_model.json'}")
    print(f"   下一步: 在 C++ 中使用 heat_solver_inference.hpp 加载")


if __name__ == "__main__":
    main()