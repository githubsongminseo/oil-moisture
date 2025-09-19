# -*- coding: utf-8 -*-
"""
MobileNetV3-Small — Binned Classification (CrossEntropy only) + Training Tweaks
- Bins: [40, 50, 60, 70, 80]  -> 4 classes
- Loss: CrossEntropy (hard labels). *No* regression metrics.
- Reports/CSV columns (exact): epoch,phase,loss,acc,precision_macro,recall_macro,f1_macro,roc_auc_macro_ovr,lr
- Prints Top-2 and Balanced Accuracy to console for reference.
- Early stopping on Macro-F1

Added (toggleable):
- Class weights for CE (CLASS_WEIGHTS)
- WeightedRandomSampler for balanced mini-batches (USE_WEIGHTED_SAMPLER)
- Label smoothing for CE (LABEL_SMOOTHING)
- LR warmup + cosine annealing (WARMUP_EPOCHS)
- Backbone freeze-then-unfreeze (FREEZE_EPOCHS)
"""

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

OUT_DIR   = Path(r"C:\\Users\\win\\Desktop\\project\\model\\Aug_models\\Re\\classfication\\mobilenet_v3_small_bin10_2_0919")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_LAST = OUT_DIR / "last.pt"
CKPT_BEST = OUT_DIR / "best.pt"
LOG_CSV   = OUT_DIR / "metrics_log.csv"
VIZ_DIR   = OUT_DIR / "viz"

# ===== Settings =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64  # ↑ (VRAM 여유 없으면 32로)
NUM_WORKERS = 4  # Windows에서 이슈면 0
EPOCHS = 80      # ↑
BASE_LR = 1e-3   # ↑ (워밍업 사용)
WEIGHT_DECAY = 5e-5
EARLY_STOP_PATIENCE = 10
SEED = 2025
PRETRAINED = True

TARGET_SIZE = (256, 256)  # ↑ (필요시 224로)
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

# ===== Tweaks (toggles) =====
CLASS_WEIGHTS = True            # CE에 클래스 가중치 적용
USE_WEIGHTED_SAMPLER = True     # train 미니배치 균형화
LABEL_SMOOTHING = 0.10          # 0.0이면 끔 (0.05~0.1 권장)
WARMUP_EPOCHS = 5               # 0이면 워밍업 없음
FREEZE_EPOCHS = 5               # 0이면 동결 없음 (features unfreeze 시점)

# ===== Bins (classification only) =====
RANGE_MIN = 40.0
RANGE_MAX = 80.0
BIN_EDGES = [40.0, 50.0, 60.0, 70.0, 80.0]   # 4 bins
NUM_CLASSES = len(BIN_EDGES) - 1

INCLUDE_EXPANDED = False  # set True if *_expanded*.npy should be included

# ===== Optional: sklearn for ROC-AUC =====
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

def rows_to_items(rows: List[dict], npy_root: Path, include_expanded=INCLUDE_EXPANDED) -> List[Tuple[Path,int]]:
    items, miss, seen = [], [], set()
    for r in rows:
        stem = r["stem"]; y = r["moisture"]; cls = moisture_to_bin_index(y)
        matched = False
        exact = npy_root / f"{stem}.npy"
        if exact.exists():
            if exact not in seen:
                items.append((exact, cls)); seen.add(exact)
            matched = True
        for p in npy_root.glob(f"{stem}_aug*.npy"):
            if p not in seen:
                items.append((p, cls)); seen.add(p)
            matched = True
        if include_expanded:
            for p in npy_root.glob(f"{stem}_expanded*.npy"):
                if p not in seen:
                    items.append((p, cls)); seen.add(p)
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
        x = x.unsqueeze(-1).repeat(1,1,3)  # H W 3
        x = x.permute(2,0,1)
    elif x.ndim == 3:
        if x.shape[-1] == 3:
            x = x.permute(2,0,1)
        elif x.shape[0] == 3:
            pass
        else:
            if x.shape[0] == 1:
                x = x.repeat(3,1,1)
            elif x.shape[-1] >= 3:
                x = x[...,:3].permute(2,0,1)
            elif x.shape[0] >= 3:
                x = x[:3,...]
            else:
                x = x.unsqueeze(0).repeat(3,1,1)
    else:
        raise ValueError(f"Unsupported npy shape: {tuple(x.shape)} for {npy_path}")
    if x.max() > 1.5: x = x/255.0
    x = x.unsqueeze(0)
    x = Fnn.interpolate(x, size=TARGET_SIZE, mode='bilinear', align_corners=False)
    x = x.squeeze(0)
    m = torch.tensor(MEAN).view(3,1,1); s = torch.tensor(STD).view(3,1,1)
    x = (x - m)/s
    return x

# ===== Dataset/Loader =====
class NPYClsDataset(Dataset):
    def __init__(self, items: List[Tuple[Path,int]]):
        self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        p, cls = self.items[idx]
        x = preprocess_npy(p)
        return x, torch.tensor(cls, dtype=torch.long), str(p)

def make_train_loader(items, batch_size=BATCH_SIZE, use_weighted_sampler=False, class_weights=None):
    g = torch.Generator(); g.manual_seed(SEED)
    if use_weighted_sampler and class_weights is not None:
        # 각 샘플 weight = 해당 클래스의 weight
        labels = [cls for _, cls in items]
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
    g = torch.Generator(); g.manual_seed(SEED)
    return DataLoader(
        NPYClsDataset(items),
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=None
    )

# ===== Model =====
def build_mobilenet_v3_small(pretrained=PRETRAINED):
    m = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
    in_f = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_f, NUM_CLASSES)
    return m

# ===== Evaluate (classification only) =====
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    C = NUM_CLASSES
    cm = torch.zeros((C,C), dtype=torch.long)
    total, correct1, correct2 = 0, 0, 0
    total_loss = 0.0
    y_true_list, y_pred_list, prob_list = [], [], []

    for x, y, _ in loader:
        x = x.to(DEVICE); y = y.to(DEVICE)
        logits = model(x)
        loss = F.cross_entropy(logits, y, label_smoothing=LABEL_SMOOTHING if LABEL_SMOOTHING>0 else 0.0)
        total_loss += loss.item() * x.size(0)
        prob = F.softmax(logits, dim=1)
        pred1 = prob.argmax(dim=1)
        _, top2 = prob.topk(2, dim=1)
        correct1 += (pred1 == y).sum().item()
        correct2 += (top2 == y.unsqueeze(1)).any(dim=1).sum().item()
        total += y.numel()
        for t,pred in zip(y.view(-1), pred1.view(-1)):
            cm[t.long(), pred.long()] += 1
        y_true_list.append(y.cpu().numpy())
        y_pred_list.append(pred1.cpu().numpy())
        prob_list.append(prob.cpu().numpy())

    y_true = np.concatenate(y_true_list) if y_true_list else np.array([])
    y_pred = np.concatenate(y_pred_list) if y_pred_list else np.array([])
    probs  = np.vstack(prob_list)        if prob_list else np.zeros((0, C))

    # per-class precision/recall/f1 from confusion matrix (macro)
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

    # ROC-AUC (macro OvR) — requires sklearn
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
def visualize_random_samples(model, items: List[Tuple[Path,int]], k=3, out_dir: Path = VIZ_DIR, title_prefix="val"):
    if len(items) == 0: return
    out_dir.mkdir(parents=True, exist_ok=True)
    picks = random.sample(items, k=min(k, len(items)))
    model.eval()
    for i, (path, gt_cls) in enumerate(picks, 1):
        x = preprocess_npy(path).unsqueeze(0).to(DEVICE)
        logits = model(x); prob = F.softmax(logits, dim=1)
        pred_cls = int(prob.argmax(dim=1).item())
        def bin_label(i):
            lo, hi = BIN_EDGES[i], BIN_EDGES[i+1]
            return f"{int(lo)}–{int(hi)}"
        img = denormalize_to_uint8(x.squeeze(0))
        plt.figure(figsize=(8,5)); plt.imshow(img); plt.axis('off')
        plt.title(f"{title_prefix}: pred={bin_label(pred_cls)} / gt={bin_label(gt_cls)}", fontsize=14)
        plt.savefig(out_dir / f"{title_prefix}_sample_{i}.png", bbox_inches='tight', pad_inches=0.1)
        plt.close()

# ===== Helper: compute class weights from train set =====
def compute_class_weights_from_items(items: List[Tuple[Path,int]]) -> torch.Tensor:
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for _, cls in items:
        counts[cls] += 1
    total = counts.sum()
    inv = total / (counts + 1e-6)          # inverse frequency
    weights = inv / inv.sum()              # normalize
    print("[CLASS WEIGHTS]", {i: float(w) for i, w in enumerate(weights)})
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)

# ===== Train (CE only, with tweaks) =====
def train(model, train_loader, val_loader, class_weights=None):
    model = model.to(DEVICE)

    # --- Freeze backbone for first FREEZE_EPOCHS
    if FREEZE_EPOCHS > 0:
        for p in model.features.parameters():
            p.requires_grad = False

    # CE with class weights and label smoothing
    ce_kwargs = {}
    if class_weights is not None:
        ce_kwargs["weight"] = class_weights
    if LABEL_SMOOTHING and LABEL_SMOOTHING > 0:
        ce_kwargs["label_smoothing"] = float(LABEL_SMOOTHING)

    optim = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    # Cosine is applied *after* warmup. We'll call cosine.step() only after warmup phase.
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, EPOCHS - max(0, WARMUP_EPOCHS)))

    best_score = -1.0
    best_state = None
    wait = 0

    for ep in range(1, EPOCHS + 1):
        # Unfreeze after FREEZE_EPOCHS
        if FREEZE_EPOCHS > 0 and ep == FREEZE_EPOCHS + 1:
            for p in model.features.parameters():
                p.requires_grad = True
            print(f"[UNFREEZE] Backbone unfrozen at epoch {ep}")

        # ---- warmup lr (linear)
        if WARMUP_EPOCHS > 0 and ep <= WARMUP_EPOCHS:
            warmup_lr = BASE_LR * ep / WARMUP_EPOCHS
            for g in optim.param_groups:
                g["lr"] = warmup_lr
        elif WARMUP_EPOCHS > 0 and ep == WARMUP_EPOCHS + 1:
            # ensure exact BASE_LR handed to cosine start
            for g in optim.param_groups:
                g["lr"] = BASE_LR

        model.train()
        run_loss, seen = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{EPOCHS}", leave=False)
        for x, y, _ in pbar:
            x = x.to(DEVICE); y = y.to(DEVICE)
            logits = model(x)
            loss = F.cross_entropy(logits, y, **ce_kwargs)
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

        # step cosine after warmup window
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

    # ---- class weights from train distribution
    class_weights = None
    if CLASS_WEIGHTS:
        class_weights = compute_class_weights_from_items(train_items)
    else:
        print("[INFO] CLASS_WEIGHTS disabled.")

    # ---- train loader (optionally weighted sampler)
    train_loader = make_train_loader(
        train_items,
        batch_size=BATCH_SIZE,
        use_weighted_sampler=USE_WEIGHTED_SAMPLER,
        class_weights=class_weights if class_weights is not None else torch.ones(NUM_CLASSES, device=DEVICE)
    )
    val_loader   = make_valtest_loader(val_items,   batch_size=BATCH_SIZE)
    test_loader  = make_valtest_loader(test_items,  batch_size=BATCH_SIZE)

    model = build_mobilenet_v3_small(pretrained=PRETRAINED)
    model = train(model, train_loader, val_loader, class_weights=class_weights)

    # Final test
    m = evaluate(model, test_loader)
    print("[TEST] Acc %.4f | Top2 %.4f | MacroF1 %.4f | BalancedAcc %.4f | "
          "Prec %.4f | Rec %.4f | AUC(ovr-macro) %.4f" % (
        m['Acc'], m['Top2'], m['MacroF1'], m['BalancedAcc'], m['PrecisionMacro'], m['RecallMacro'], m['ROCAUC_MacroOvR']))
    log_csv_row(LOG_CSV, epoch=0, phase="test", metrics=m, loss=m["ValLoss"], lr=0.0)

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    visualize_random_samples(model, test_items, k=3, out_dir=VIZ_DIR/"test_examples", title_prefix="test")
