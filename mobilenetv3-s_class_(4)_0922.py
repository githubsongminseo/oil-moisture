# -*- coding: utf-8 -*-

# 기대값→bin 매핑 (한 줄로 효과)

# TTA (테스트/검증만)

# 경계 가중치 (훈련 루프에 몇 줄)

# Ordinal 학습 (가장 강력, 구조 변경 수반)

import os, sys, json, csv, random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torchvision.models as tvm
from tqdm import tqdm

print("[RUNNING FILE]", Path(__file__).resolve())

# ===== Paths (EDIT) =====
LABEL_TRAIN = Path(r"C:\\Users\\win\\Desktop\\project\\label_mois\\label\\mapped_train_cleaned_aug.json")
LABEL_VAL   = Path(r"C:\\Users\\win\\Desktop\\project\\label_mois\\label\\mapped_val_cleaned.json")
LABEL_TEST  = Path(r"C:\\Users\\win\\Desktop\\project\\label_mois\\label\\mapped_test_cleaned.json")

NPY_DIR_TRAIN = Path(r"C:\\Users\\win\\Desktop\\project\\Data Augmentation\\train\\npy")
NPY_DIR_VAL   = Path(r"C:\\Users\\win\\Desktop\\project\\testval_npy\\val")
NPY_DIR_TEST  = Path(r"C:\\Users\\win\\Desktop\\project\\testval_npy\\test")

base = Path(r"C:\Users\win\Desktop\project\model\Aug_models\Re\classification")
OUT_DIR = base / "mobilenetv3-s_ep80_bat32_lr2e-4_tta"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_LAST = OUT_DIR / "last.pt"
CKPT_BEST = OUT_DIR / "best.pt"
LOG_CSV   = OUT_DIR / "metrics_log.csv"
VIZ_DIR   = OUT_DIR / "viz"

# ===== Settings =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_WORKERS = 4     # Windows 문제 있으면 0
EPOCHS = 80
BASE_LR = 2e-4
WEIGHT_DECAY = 1e-4     ##수정사항
EARLY_STOP_PATIENCE = 8 ##수정사항
SEED = 2025
PRETRAINED = True

TARGET_SIZE = (256, 256)
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

# ===== Feature toggles =====
# 채널 순서 (OpenCV -> BGR 저장이면 True 권장)
NPY_IS_BGR = True

# CE/샘플링 관련
CLASS_WEIGHTS = True
USE_WEIGHTED_SAMPLER = True
LABEL_SMOOTHING = 0.05 ##수정사항

# LR 스케줄
WARMUP_EPOCHS = 5
FREEZE_EPOCHS = 5

# ---- 경계 가중치 (train only)
USE_BOUNDARY_WEIGHT = True
BOUNDARY_EPS = 1.0       # 경계 ±1.0 안쪽
MIN_W = 0.3              # 경계 바로 근처 샘플 가중치

# ---- 기대값 기반 결정 (eval/test)
USE_EXPECTATION_DECISION = True

# ---- TTA at eval/test
USE_TTA = True

# ---- (Optional) Ordinal(CORAL) 학습
USE_ORDINAL = False  # True로 켜면 재학습 필요(헤드/로스 변경)

# ===== Bins =====
RANGE_MIN = 40.0
RANGE_MAX = 80.0
BIN_EDGES = [40.0, 50.0, 60.0, 70.0, 80.0]   # 4 bins
NUM_CLASSES = len(BIN_EDGES) - 1
BIN_CENTERS = torch.tensor([45., 55., 65., 75.], dtype=torch.float32)  # expectation용

INCLUDE_EXPANDED = False

# ===== sklearn for ROC-AUC =====
try:
    from sklearn.metrics import roc_auc_score
    SKLEARN_AVAILABLE = True
except Exception as e:
    print("[WARN] sklearn not available, ROCAUC will be NaN:", e)
    SKLEARN_AVAILABLE = False

# ===== Utils =====
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def assert_exists(p: Path, tag: str):
    print(f"[CHECK-{tag}]", p)
    if not p.exists():
        print(f"[ERROR] {tag} 경로가 존재하지 않습니다: {p}")
        sys.exit(2)

for p, t in [(LABEL_TRAIN, "TRAIN"),(LABEL_VAL,"VAL"),(LABEL_TEST,"TEST"),
             (NPY_DIR_TRAIN,"NPY_TRAIN"),(NPY_DIR_VAL,"NPY_VAL"),(NPY_DIR_TEST,"NPY_TEST")]:
    assert_exists(p, t)

# ===== I/O =====
def load_label_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    out = []
    for r in rows:
        stem = Path(r["filename"]).stem
        out.append({"stem": stem, "moisture": float(r["moisture"])})
    print(f"[LOAD] {path} -> {len(out)} rows")
    return out

def filter_by_range(rows: List[dict], lo=RANGE_MIN, hi=RANGE_MAX) -> List[dict]:
    f = [r for r in rows if lo <= float(r["moisture"]) <= hi]
    print(f"[FILTER] keep {len(f)} / {len(rows)} rows in [{lo},{hi}]")
    return f

def moisture_to_bin_index(v: float) -> int:
    v = float(v)
    for i in range(len(BIN_EDGES) - 1):
        lo, hi = BIN_EDGES[i], BIN_EDGES[i+1]
        if i < len(BIN_EDGES) - 2:
            if lo <= v < hi: return i
        else:
            if lo <= v <= hi: return i
    return 0 if v < BIN_EDGES[0] else (len(BIN_EDGES) - 2)

def rows_to_items(rows: List[dict], npy_root: Path, include_expanded=INCLUDE_EXPANDED) -> List[Tuple[Path,int,float]]:
    items, miss, seen = [], [], set()
    for r in rows:
        stem = r["stem"]; y = float(r["moisture"]); cls = moisture_to_bin_index(y)
        matched = False
        exact = npy_root / f"{stem}.npy"
        if exact.exists():
            if exact not in seen:
                items.append((exact, cls, y)); seen.add(exact)
            matched = True
        for p in npy_root.glob(f"{stem}_aug*.npy"):
            if p not in seen:
                items.append((p, cls, y)); seen.add(p)
            matched = True
        if include_expanded:
            for p in npy_root.glob(f"{stem}_expanded*.npy"):
                if p not in seen:
                    items.append((p, cls, y)); seen.add(p)
                matched = True
        if not matched:
            miss.append(stem)
    if miss[:10]:
        print(f"[MISS] first 10 missing under {npy_root}:", miss[:10])
    print(f"[MAP] rows={len(rows)} -> matched_npy={len(items)} under {npy_root}")
    return items

# ===== Preprocess =====
import torch.nn.functional as Fnn

def preprocess_npy(npy_path: Path):
    arr = np.load(str(npy_path))
    x = torch.from_numpy(arr).to(torch.float32)
    if x.ndim == 2:
        x = x.unsqueeze(-1).repeat(1,1,3).permute(2,0,1)  # C H W
    elif x.ndim == 3:
        if x.shape[-1] == 3:              # H W 3
            x = x.permute(2,0,1)          # C H W
            if NPY_IS_BGR: x = x[[2,1,0], ...]  # BGR->RGB
        elif x.shape[0] == 3:             # C H W
            if NPY_IS_BGR: x = x[[2,1,0], ...]
        else:
            if x.shape[0] == 1:
                x = x.repeat(3,1,1)
            elif x.shape[-1] >= 3:
                x = x[...,:3].permute(2,0,1)
                if NPY_IS_BGR: x = x[[2,1,0], ...]
            elif x.shape[0] >= 3:
                x = x[:3,...]
                if NPY_IS_BGR: x = x[[2,1,0], ...]
            else:
                x = x.unsqueeze(0).repeat(3,1,1)
    else:
        raise ValueError(f"Unsupported npy shape: {tuple(x.shape)} for {npy_path}")

    if x.max() > 1.5: x = x/255.0
    x = Fnn.interpolate(x.unsqueeze(0), size=TARGET_SIZE,
                        mode='bilinear', align_corners=False).squeeze(0)
    m = torch.tensor(MEAN).view(3,1,1); s = torch.tensor(STD).view(3,1,1)
    x = (x - m)/s
    return x

# ===== Dataset/Loader =====
class NPYClsDataset(Dataset):
    def __init__(self, items: List[Tuple[Path,int,float]]):
        self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        p, cls, mois = self.items[idx]
        x = preprocess_npy(p)
        return x, torch.tensor(cls, dtype=torch.long), float(mois), str(p)

def make_train_loader(items, batch_size=BATCH_SIZE, use_weighted_sampler=False, class_weights=None):
    g = torch.Generator(); g.manual_seed(SEED)
    if use_weighted_sampler and class_weights is not None:
        labels = [cls for _, cls, _ in items]
        sample_weights = torch.tensor([class_weights[cls].item() for cls in labels], dtype=torch.float)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True
    return DataLoader(
        NPYClsDataset(items),
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=g
    )

def make_valtest_loader(items, batch_size=BATCH_SIZE):
    return DataLoader(
        NPYClsDataset(items),
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

# ===== Model =====
def build_mobilenet_v3_small_ce(pretrained=PRETRAINED):
    m = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
    in_f = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_f, NUM_CLASSES)
    return m

# Ordinal(CORAL): K-1 로짓을 예측(경계 ≥50, ≥60, ≥70)
class CoralHead(nn.Module):
    def __init__(self, in_f, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_f, num_classes-1)
    def forward(self, x):
        return self.fc(x)

def build_mobilenet_v3_small_ordinal(pretrained=PRETRAINED):
    m = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
    in_f = m.classifier[-1].in_features
    # 분류기 교체: features -> pooling -> classifier[:-1] -> CoralHead
    # MobileNetV3-Small의 classifier는 [Linear->Hardswish->Dropout->Linear]
    # 마지막 Linear 대신 CoralHead로 교체
    cls = list(m.classifier)
    in_f = cls[-1].in_features
    m.classifier[-1] = CoralHead(in_f, NUM_CLASSES)
    return m

# ===== Ordinal helpers =====
def ordinal_targets_from_class(y_cls: torch.Tensor, K: int) -> torch.Tensor:
    # y>=edge? 형태의 K-1 타겟 (B, K-1)
    # y in [0..K-1]
    B = y_cls.size(0)
    t = torch.zeros(B, K-1, device=y_cls.device, dtype=torch.float32)
    for k in range(K-1):
        t[:, k] = (y_cls > k).float()  # CORAL: P(y > k)
    return t

def probs_from_ordinal_logits(logits_ord: torch.Tensor, K: int) -> torch.Tensor:
    # logits_ord: [B, K-1], q_k = sigmoid(logit_k) ~= P(y>k)
    q = torch.sigmoid(logits_ord)                     # [B, K-1]
    # class probs from q: p0 = 1-q0; pk = q_{k-1}-q_k; p_{K-1}=q_{K-2}
    B = q.size(0)
    p = torch.zeros(B, K, device=q.device, dtype=torch.float32)
    p[:, 0] = 1.0 - q[:, 0]
    for k in range(1, K-1):
        p[:, k] = q[:, k-1] - q[:, k]
    p[:, K-1] = q[:, K-2]
    # 수치 오차 보정
    p = torch.clamp(p, min=0.0)
    p = p / (p.sum(dim=1, keepdim=True) + 1e-12)
    return p

# ===== Evaluate =====
@torch.no_grad()
def tta_probabilities(model, x, ce_mode=True):
    # views: original, hflip, mild scale-up center crop
    views = [x, torch.flip(x, dims=[-1])]
    H, W = x.shape[-2:]
    up = Fnn.interpolate(x, size=(int(H*1.1), int(W*1.1)), mode='bilinear', align_corners=False)
    dh, dw = (up.shape[-2]-H)//2, (up.shape[-1]-W)//2
    crop = up[..., dh:dh+H, dw:dw+W]
    views.append(crop)

    probs = []
    for v in views:
        out = model(v)
        if ce_mode:
            pr = F.softmax(out, dim=1)                # [B,K]
        else:
            pr = probs_from_ordinal_logits(out, NUM_CLASSES)  # [B,K]
        probs.append(pr)
    return torch.stack(probs, dim=0).mean(dim=0)       # [B,K]

def to_bin_index_from_value(v: float, edges=BIN_EDGES):
    for i in range(len(edges)-1):
        lo, hi = edges[i], edges[i+1]
        if (i < len(edges)-2 and lo <= v < hi) or (i==len(edges)-2 and lo <= v <= hi):
            return i
    return 0

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    C = NUM_CLASSES
    cm = torch.zeros((C,C), dtype=torch.long)
    total, correct1, correct2 = 0, 0, 0
    total_loss = 0.0
    y_true_list, y_pred_list, prob_list = [], [], []

    ce_mode = not USE_ORDINAL
    centers = BIN_CENTERS.to(DEVICE).view(1, -1)

    for x, y, _, _ in loader:
        x = x.to(DEVICE); y = y.to(DEVICE)

        # loss는 단일뷰로 계산(일관성 위해)
        logits_single = model(x)
        if ce_mode:
            loss = F.cross_entropy(logits_single, y, label_smoothing=LABEL_SMOOTHING if LABEL_SMOOTHING>0 else 0.0)
        else:
            t = ordinal_targets_from_class(y, C)        # [B,K-1]
            bce = F.binary_cross_entropy_with_logits(logits_single, t, reduction='none')  # [B,K-1]
            loss = bce.mean()

        total_loss += loss.item() * x.size(0)

        # 확률은 TTA로 얻을 수 있음
        if USE_TTA:
            prob = tta_probabilities(model, x, ce_mode=ce_mode)
        else:
            if ce_mode:
                prob = F.softmax(logits_single, dim=1)
            else:
                prob = probs_from_ordinal_logits(logits_single, C)

        # 결정: 기대값->bin 재매핑 or argmax
        if USE_EXPECTATION_DECISION:
            y_hat = (prob * centers).sum(dim=1)  # [B]
            pred1 = torch.tensor([to_bin_index_from_value(float(v)) for v in y_hat.detach().cpu().tolist()],
                                 device=DEVICE)
        else:
            pred1 = prob.argmax(dim=1)

        # top2 계산
        _, top2 = prob.topk(2, dim=1)

        correct1 += (pred1 == y).sum().item()
        correct2 += (top2 == y.unsqueeze(1)).any(dim=1).sum().item()
        total += y.numel()

        for t_i, p_i in zip(y.view(-1), pred1.view(-1)):
            cm[t_i.long(), p_i.long()] += 1

        y_true_list.append(y.cpu().numpy())
        y_pred_list.append(pred1.cpu().numpy())
        prob_list.append(prob.cpu().numpy())

    y_true = np.concatenate(y_true_list) if y_true_list else np.array([])
    y_pred = np.concatenate(y_pred_list) if y_pred_list else np.array([])
    probs  = np.vstack(prob_list)        if prob_list else np.zeros((0, C))

    # per-class metrics
    recalls, precisions, f1s = [], [], []
    for k in range(C):
        tp = cm[k,k].item()
        fn = cm[k,:].sum().item() - tp
        fp = cm[:,k].sum().item() - tp
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        f1   = 2 * prec * rec / (prec + rec + 1e-12)
        precisions.append(prec); recalls.append(rec); f1s.append(f1)

    acc             = correct1 / max(1, total)
    top2            = correct2 / max(1, total)
    macro_f1        = float(sum(f1s)/C)
    precision_macro = float(sum(precisions)/C)
    recall_macro    = float(sum(recalls)/C)
    balanced_acc    = float(sum(recalls)/C)
    val_loss        = total_loss / max(1, total)

    # ROC-AUC OvR macro
    if SKLEARN_AVAILABLE and y_true.size and probs.shape[0] == y_true.shape[0]:
        try:
            roc_auc_macro_ovr = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
        except Exception:
            roc_auc_macro_ovr = float("nan")
    else:
        roc_auc_macro_ovr = float("nan")

    return {
        "ValLoss": val_loss,
        "Acc": acc,
        "Top2": top2,
        "MacroF1": macro_f1,
        "BalancedAcc": balanced_acc,
        "PrecisionMacro": precision_macro,
        "RecallMacro": recall_macro,
        "ROCAUC_MacroOvR": roc_auc_macro_ovr,
        "CM": cm.tolist(),
    }

# ===== Log/CKPT/Vis =====
def current_lr(optim):
    return float(optim.param_groups[0]["lr"]) if optim.param_groups else BASE_LR

def log_csv_row(path: Path, epoch, phase, metrics: dict, loss=None, lr=None):
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["epoch","phase","loss","acc","precision_macro","recall_macro","f1_macro","roc_auc_macro_ovr","lr"])
        w.writerow([
            epoch,
            phase,
            f"{(0.0 if loss is None else loss):.6f}",
            f"{metrics.get('Acc',0.0):.6f}",
            f"{metrics.get('PrecisionMacro',0.0):.6f}",
            f"{metrics.get('RecallMacro',0.0):.6f}",
            f"{metrics.get('MacroF1',0.0):.6f}",
            f"{metrics.get('ROCAUC_MacroOvR',0.0):.6f}",
            f"{(0.0 if lr is None else lr):.8f}",
        ])

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def denormalize_to_uint8(x: torch.Tensor):
    if isinstance(x, torch.Tensor): x = x.detach().cpu()
    m = torch.tensor(MEAN).view(3,1,1)
    s = torch.tensor(STD).view(3,1,1)
    img = (x * s + m).clamp(0,1).permute(1,2,0).numpy()
    img = (img * 255.0).round().astype(np.uint8)
    return img

@torch.no_grad()
def visualize_random_samples(model, items: List[Tuple[Path,int,float]], k=3, out_dir: Path = VIZ_DIR, title_prefix="val"):
    if len(items) == 0: return
    out_dir.mkdir(parents=True, exist_ok=True)
    picks = random.sample(items, k=min(k, len(items)))
    model.eval()
    centers = BIN_CENTERS.to(DEVICE).view(1,-1)
    for i, (path, gt_cls, _) in enumerate(picks, 1):
        x = preprocess_npy(path).unsqueeze(0).to(DEVICE)
        out = model(x)
        if USE_ORDINAL:
            prob = probs_from_ordinal_logits(out, NUM_CLASSES)
        else:
            prob = F.softmax(out, dim=1)
        # 기대값 기반 표시
        y_hat = float((prob * centers).sum(dim=1).item())
        def bin_label(i):
            lo, hi = BIN_EDGES[i], BIN_EDGES[i+1]
            return f"{int(lo)}–{int(hi)}"
        pred_cls = int(prob.argmax(dim=1).item())
        img = denormalize_to_uint8(x.squeeze(0))
        plt.figure(figsize=(8,5)); plt.imshow(img); plt.axis('off')
        plt.title(f"{title_prefix}: pred={bin_label(pred_cls)} (exp={y_hat:.1f}) / gt={bin_label(gt_cls)}", fontsize=14)
        plt.savefig(out_dir / f"{title_prefix}_sample_{i}.png", bbox_inches='tight', pad_inches=0.1)
        plt.close()

# ===== Helper: compute class weights from train set =====
def compute_class_weights_from_items(items: List[Tuple[Path,int,float]]) -> torch.Tensor:
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for _, cls, _ in items:
        counts[cls] += 1
    total = counts.sum()
    inv = total / (counts + 1e-6)
    weights = inv / inv.sum()
    print("[CLASS WEIGHTS]", {i: float(w) for i, w in enumerate(weights)})
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)

# ===== Train =====
def boundary_weight(mois_t: torch.Tensor) -> torch.Tensor:
    # mois_t: [B] float tensor
    if not USE_BOUNDARY_WEIGHT:
        return torch.ones_like(mois_t)
    edges = torch.tensor(BIN_EDGES[1:-1], device=mois_t.device).view(1,-1)  # [50,60,70]
    d = (mois_t.view(-1,1) - edges).abs().min(dim=1).values                 # [B] 경계까지 최소거리
    w = torch.ones_like(d)
    mask = d < BOUNDARY_EPS
    w[mask] = MIN_W + (1 - MIN_W) * (d[mask] / BOUNDARY_EPS)               # 경계일수록 downweight
    return w

def train(model, train_loader, val_loader, class_weights=None):
    model = model.to(DEVICE)

    # Freeze backbone
    if FREEZE_EPOCHS > 0:
        for p in model.features.parameters():
            p.requires_grad = False

    optim = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, EPOCHS - max(0, WARMUP_EPOCHS)))

    best_score = -1.0
    best_state = None
    wait = 0

    # CE loss kwargs
    ce_kwargs = {}
    if (not USE_ORDINAL) and class_weights is not None:
        ce_kwargs["weight"] = class_weights
    if (not USE_ORDINAL) and LABEL_SMOOTHING and LABEL_SMOOTHING > 0:
        ce_kwargs["label_smoothing"] = float(LABEL_SMOOTHING)

    for ep in range(1, EPOCHS + 1):
        # unfreeze
        if FREEZE_EPOCHS > 0 and ep == FREEZE_EPOCHS + 1:
            for p in model.features.parameters():
                p.requires_grad = True
            print(f"[UNFREEZE] Backbone unfrozen at epoch {ep}")

        # warmup
        if WARMUP_EPOCHS > 0 and ep <= WARMUP_EPOCHS:
            warmup_lr = BASE_LR * ep / WARMUP_EPOCHS
            for g in optim.param_groups:
                g["lr"] = warmup_lr
        elif WARMUP_EPOCHS > 0 and ep == WARMUP_EPOCHS + 1:
            for g in optim.param_groups:
                g["lr"] = BASE_LR

        model.train()
        run_loss, seen = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{EPOCHS}", leave=False)
        for x, y_cls, y_mois, _ in pbar:
            x = x.to(DEVICE); y_cls = y_cls.to(DEVICE)
            logits = model(x)

            if USE_ORDINAL:
                t = ordinal_targets_from_class(y_cls, NUM_CLASSES)          # [B,K-1]
                bce = F.binary_cross_entropy_with_logits(logits, t, reduction='none')  # [B,K-1]
                per_sample = bce.mean(dim=1)                                 # [B]
            else:
                per_sample = F.cross_entropy(logits, y_cls, reduction='none', **ce_kwargs)  # [B]

            w_b = boundary_weight(torch.tensor(y_mois, device=DEVICE, dtype=torch.float32)) # [B]
            loss = (per_sample * w_b).mean()

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            bs = x.size(0)
            run_loss += loss.item() * bs
            seen += bs
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        tr_loss = run_loss / max(1, seen)
        curr_lr = current_lr(optim)
        log_csv_row(LOG_CSV, ep, "train", metrics={}, loss=tr_loss, lr=curr_lr)

        # Validation
        val = evaluate(model, val_loader)
        print(f"[Ep {ep:02d}] TL {tr_loss:.4f} | Acc {val['Acc']:.4f} Top2 {val['Top2']:.4f} | "
              f"MacroF1 {val['MacroF1']:.4f} BalAcc {val['BalancedAcc']:.4f} | "
              f"Prec {val['PrecisionMacro']:.4f} Rec {val['RecallMacro']:.4f} | "
              f"AUC(ovr-macro) {val['ROCAUC_MacroOvR']:.4f} | ValLoss {val['ValLoss']:.4f} | lr {curr_lr:.6e}")
        log_csv_row(LOG_CSV, ep, "val", metrics=val, loss=val["ValLoss"], lr=curr_lr)

        # Early stopping on Macro-F1
        score = val["MacroF1"]
        if score > best_score + 1e-6:
            best_score = score; wait = 0
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save({"epoch": ep, "model": model.state_dict(), "best": score}, CKPT_BEST)
        else:
            wait += 1
            if wait >= EARLY_STOP_PATIENCE:
                print(f"[EarlyStop] No val improvement for {EARLY_STOP_PATIENCE} epochs. Stop at ep={ep}.")
                torch.save({"epoch": ep, "model": model.state_dict(), "best": best_score}, CKPT_LAST)
                break

        if WARMUP_EPOCHS == 0 or ep > WARMUP_EPOCHS:
            cosine.step()

        visualize_random_samples(model, val_items, k=3, out_dir=VIZ_DIR/ ("epoch_%02d" % ep),
                                 title_prefix=f"val_ep{ep:02d}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# ===== Main =====
if __name__ == "__main__":
    set_seed(SEED)

    train_rows = load_label_rows(LABEL_TRAIN)
    val_rows   = load_label_rows(LABEL_VAL)
    test_rows  = load_label_rows(LABEL_TEST)

    train_rows = filter_by_range(train_rows)
    val_rows   = filter_by_range(val_rows)
    test_rows  = filter_by_range(test_rows)

    train_items = rows_to_items(train_rows, NPY_DIR_TRAIN, include_expanded=INCLUDE_EXPANDED)
    val_items   = rows_to_items(val_rows,   NPY_DIR_VAL,   include_expanded=INCLUDE_EXPANDED)
    test_items  = rows_to_items(test_rows,  NPY_DIR_TEST,  include_expanded=INCLUDE_EXPANDED)

    print("[DEBUG] items | train:", len(train_items), "val:", len(val_items), "test:", len(test_items))
    assert len(train_items) > 0, "Train 매칭 0개입니다. stem/경로를 확인하세요."
    assert len(val_items)   > 0, "Val 매칭 0개입니다. 경로/파일명을 확인하세요."
    assert len(test_items)  > 0, "Test 매칭 0개입니다. 경로/파일명을 확인하세요."

    # class weights
    class_weights = None
    if CLASS_WEIGHTS and (not USE_ORDINAL):
        class_weights = compute_class_weights_from_items(train_items)
    else:
        print("[INFO] CLASS_WEIGHTS disabled or ORDINAL mode.")

    # loaders
    train_loader = make_train_loader(
        train_items,
        batch_size=BATCH_SIZE,
        use_weighted_sampler=USE_WEIGHTED_SAMPLER,
        class_weights=class_weights if class_weights is not None else torch.ones(NUM_CLASSES, device=DEVICE)
    )
    val_loader   = make_valtest_loader(val_items,   batch_size=BATCH_SIZE)
    test_loader  = make_valtest_loader(test_items,  batch_size=BATCH_SIZE)

    # model
    if USE_ORDINAL:
        model = build_mobilenet_v3_small_ordinal(pretrained=PRETRAINED)
    else:
        model = build_mobilenet_v3_small_ce(pretrained=PRETRAINED)

    model = train(model, train_loader, val_loader, class_weights=class_weights)

    # Final test
    m = evaluate(model, test_loader)
    print("[TEST] Acc %.4f | Top2 %.4f | MacroF1 %.4f | BalancedAcc %.4f | "
          "Prec %.4f | Rec %.4f | AUC(ovr-macro) %.4f" % (
        m['Acc'], m['Top2'], m['MacroF1'], m['BalancedAcc'], m['PrecisionMacro'], m['RecallMacro'], m['ROCAUC_MacroOvR']))
    log_csv_row(LOG_CSV, epoch=0, phase="test", metrics=m, loss=m["ValLoss"], lr=0.0)

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    visualize_random_samples(model, test_items, k=3, out_dir=VIZ_DIR/"test_examples", title_prefix="test")
