# -*- coding: utf-8 -*-
# MobileNetV3-Small 분류(10단위 bin) + 기대값 복원(옵션) 학습 스크립트
# - 라벨 범위: [40, 80], 10단위 bin -> 4 classes
# - 학습/보고: 분류 지표(Acc, Top-2, Macro-F1, BalancedAcc) 중심
# - 보조(옵션): 기대값 복원 기반 회귀 지표는 REPORT_REGRESSION_AS_AUX 로 켜/끄기

import os, sys, json, csv, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tvm
from tqdm import tqdm

print("[RUNNING FILE]", Path(__file__).resolve())

# ===== 0) cleaned 라벨 경로 =====
LABEL_TRAIN = Path(r"C:\Users\win\Desktop\project\label_mois\label\mapped_train_cleaned_aug.json")
LABEL_VAL   = Path(r"C:\Users\win\Desktop\project\label_mois\label\mapped_val_cleaned.json")
LABEL_TEST  = Path(r"C:\Users\win\Desktop\project\label_mois\label\mapped_test_cleaned.json")

# ===== 1) (선택) 옛 경로 문자열 탐지 =====
try:
    repo_root = Path(__file__).resolve().parent.parent
    hits = []
    for p in repo_root.rglob("*.py"):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "Aug_label\\NEW\\train\\mapped_train.json" in text \
           or "Aug_label\\\\NEW\\\\train\\\\mapped_train.json" in text:
            hits.append(p)
    if hits:
        print("\n[WARN] 옛 경로 문자열이 남아있는 파일 발견:")
        for h in hits[:10]:
            print(" -", h)
        if len(hits) > 10:
            print(f" ... and {len(hits)-10} more")
        print("[HINT] 위 파일에서 경로를 cleaned 경로로 수정하거나, 해당 파일이 실행/import되지 않게 하세요.\n")
except Exception as e:
    print("[SCAN-SKIP] 경로 스캔 중 오류:", e)

# ===== 2) 경로 존재 검증 =====
def assert_exists(p: Path, tag: str):
    print(f"[CHECK-{tag}]", p)
    if not p.exists():
        print(f"[ERROR] {tag} 경로가 존재하지 않습니다:", p)
        sys.exit(2)

assert_exists(LABEL_TRAIN, "TRAIN")
assert_exists(LABEL_VAL,   "VAL")
assert_exists(LABEL_TEST,  "TEST")

# ===== 3) NPY 경로 =====
NPY_DIR_TRAIN = Path(r"C:\Users\win\Desktop\project\Data Augmentation\train\npy")
NPY_DIR_VAL   = Path(r"C:\Users\win\Desktop\project\testval_npy\val")
NPY_DIR_TEST  = Path(r"C:\Users\win\Desktop\project\testval_npy\test")

for d, tag in [(NPY_DIR_TRAIN, "NPY_TRAIN"), (NPY_DIR_VAL, "NPY_VAL"), (NPY_DIR_TEST, "NPY_TEST")]:
    print(f"[CHECK-{tag}]", d, "exists?", d.exists())

# ===== 4) 학습 출력 폴더 =====
OUT_DIR   = Path(r"C:\Users\win\Desktop\project\model\Aug_models\Re\mobilenet_v3_small_bin10_cls")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_LAST = OUT_DIR / "last.pt"
CKPT_BEST = OUT_DIR / "best.pt"
LOG_CSV   = OUT_DIR / "metrics_log.csv"
VIZ_DIR   = OUT_DIR / "viz"

# ===== 5) 설정 =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 50
LR = 3e-4
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 7
SEED = 2025
PRETRAINED = True

TARGET_SIZE = (224, 224)  # (H, W)
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

# ===== 5.5) 분류 bin 정의 (10단위) =====
RANGE_MIN = 40.0
RANGE_MAX = 80.0
BIN_EDGES   = [40.0, 50.0, 60.0, 70.0, 80.0]                 # 4 bins
BIN_CENTERS = torch.tensor([45.0, 55.0, 65.0, 75.0], dtype=torch.float32)  # 기대값 복원용(보조)
NUM_CLASSES = len(BIN_EDGES) - 1

# === Classification reporting config
REPORT_REGRESSION_AS_AUX = False    # 회귀 보조지표 표시 여부(분류 중심 보고를 위해 기본 False)
CLIP_PREDICTIONS = True             # (보조) 기대값 복원 시 [40,80] 클립

# 소프트 라벨(가우시안) 옵션
USE_SOFT_LABEL = True
SOFT_SIGMA = 1.5                     # (단위: 수분값) 1.5~2.0 권장

# 증강 매칭 포함 여부
INCLUDE_EXPANDED = False

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ===== 6) 라벨 로딩/필터 =====
def load_label_rows_exact(path: Path):
    print("[LOAD_LABEL_ROWS - TRY OPEN]", path)
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    out = []
    for r in rows:
        stem = Path(r["filename"]).stem
        out.append({"stem": stem, "moisture": float(r["moisture"])})
    print(f"[LOAD] {path} -> {len(out)} rows")
    return out

def filter_rows_by_range(rows, lo=RANGE_MIN, hi=RANGE_MAX):
    f = [r for r in rows if lo <= float(r["moisture"]) <= hi]
    print(f"[FILTER] keep {len(f)} / {len(rows)} rows in [{lo}, {hi}]")
    return f

# ===== 7) rows -> items (증강 매칭 포함) =====
def rows_to_items(rows, npy_root: Path, include_expanded=INCLUDE_EXPANDED):
    items, miss, seen = [], [], set()
    for r in rows:
        stem = r["stem"]; y = r["moisture"]
        matched = False

        exact = npy_root / f"{stem}.npy"
        if exact.exists():
            if exact not in seen:
                items.append((exact, y)); seen.add(exact)
            matched = True

        for p in npy_root.glob(f"{stem}_aug*.npy"):
            if p not in seen:
                items.append((p, y)); seen.add(p)
            matched = True

        if include_expanded:
            for p in npy_root.glob(f"{stem}_expanded*.npy"):
                if p not in seen:
                    items.append((p, y)); seen.add(p)
                matched = True

        if not matched:
            miss.append(stem)

    if miss[:10]:
        print(f"[MISS] first 10 missing under {npy_root}:", miss[:10])
    print(f"[MAP] rows={len(rows)} -> matched_npy={len(items)} under {npy_root}")
    return items

# ===== 7.5) 전처리 =====
import torch.nn.functional as Fnn

def preprocess_npy(npy_path: Path, mean=MEAN, std=STD, target_size=TARGET_SIZE):
    arr = np.load(str(npy_path))
    x = torch.from_numpy(arr).to(torch.float32)

    if x.ndim == 2:
        x = x.unsqueeze(-1).repeat(1, 1, 3)      # H W 3
        x = x.permute(2, 0, 1)                   # 3 H W
    elif x.ndim == 3:
        if x.shape[-1] == 3:
            x = x.permute(2, 0, 1)               # 3 H W
        elif x.shape[0] == 3:
            pass
        else:
            if x.shape[0] == 1:
                x = x.repeat(3, 1, 1)
            elif x.shape[-1] >= 3:
                x = x[..., :3].permute(2, 0, 1)
            elif x.shape[0] >= 3:
                x = x[:3, ...]
            else:
                x = x.unsqueeze(0).repeat(3,1,1)
    else:
        raise ValueError(f"Unsupported npy shape: {tuple(x.shape)} for {npy_path}")

    if x.max() > 1.5:  # 0~255 추정
        x = x / 255.0

    x = x.unsqueeze(0)
    x = Fnn.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
    x = x.squeeze(0)

    m = torch.tensor(mean, dtype=torch.float32).view(3,1,1)
    s = torch.tensor(std,  dtype=torch.float32).view(3,1,1)
    x = (x - m) / s
    return x

# ===== 8) 분류 라벨 유틸 =====
def moisture_to_bin_index(v: float, edges=BIN_EDGES):
    v = float(v)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i+1]
        if i < len(edges) - 2:
            if lo <= v < hi:   # 마지막 bin 제외, 우끝 미포함
                return i
        else:
            if lo <= v <= hi:  # 마지막 bin 우끝 포함
                return i
    return 0 if v < edges[0] else (len(edges) - 2)

def gaussian_soft_label(y: float, centers: torch.Tensor = BIN_CENTERS, sigma: float = SOFT_SIGMA):
    # y: scalar float -> soft prob over K bins
    c = centers  # [K]
    diff = (y - c) / sigma
    prob = torch.exp(-0.5 * diff * diff)  # [K]
    prob = prob / (prob.sum() + 1e-12)
    return prob  # torch.float32 [K], sum=1

def soft_ce_loss(logits: torch.Tensor, target_prob: torch.Tensor):
    # logits: [B,K], target_prob: [B,K] (soft)
    log_prob = F.log_softmax(logits, dim=1)
    loss = -(target_prob * log_prob).sum(dim=1).mean()
    return loss

# ===== 9) Dataset/Loader =====
class NPYSupervisedDataset(Dataset):
    def __init__(self, items):
        self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        npy_path, y = self.items[idx]
        x = preprocess_npy(npy_path)
        if USE_SOFT_LABEL:
            y_prob = gaussian_soft_label(float(y))  # [K]
            return x, y_prob, float(y), str(npy_path)
        else:
            cls = moisture_to_bin_index(y)
            return x, torch.tensor(cls, dtype=torch.long), float(y), str(npy_path)

def make_loader(items, batch_size=BATCH_SIZE, train=False):
    g = torch.Generator(); g.manual_seed(SEED)
    return DataLoader(
        NPYSupervisedDataset(items),
        batch_size=batch_size,
        shuffle=train,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        generator=g if train else None
    )

# ===== 10) 모델 =====
def build_mobilenet_v3_small_cls(pretrained=PRETRAINED):
    model = tvm.mobilenet_v3_small(
        weights=tvm.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    )
    in_f = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_f, NUM_CLASSES)  # K-way
    return model

# ===== 11) 평가(분류 중심) =====
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    C = NUM_CLASSES
    cm = torch.zeros((C, C), dtype=torch.long)
    total, correct1, correct2 = 0, 0, 0
    total_loss = 0.0

    for batch in loader:
        x = batch[0].to(DEVICE)

        if USE_SOFT_LABEL:
            hard = torch.tensor([moisture_to_bin_index(v) for v in batch[2]], device=DEVICE)
            target_prob = batch[1]
            if not isinstance(target_prob, torch.Tensor):
                target_prob = torch.stack(target_prob, dim=0)
            target_prob = target_prob.to(DEVICE)  # [B,K]
        else:
            hard = batch[1].to(DEVICE)            # [B]

        logits = model(x)                          # [B,K]
        prob   = F.softmax(logits, dim=1)          # [B,K]

        # 분류 손실
        if USE_SOFT_LABEL:
            loss = soft_ce_loss(logits, target_prob)
        else:
            loss = F.cross_entropy(logits, hard)
        bs = x.size(0)
        total_loss += loss.item() * bs

        # Top-1 / Top-2
        pred1 = prob.argmax(dim=1)                 # [B]
        _, top2 = prob.topk(2, dim=1)              # [B,2]
        correct1 += (pred1 == hard).sum().item()
        correct2 += (top2 == hard.unsqueeze(1)).any(dim=1).sum().item()
        total   += hard.numel()

        # Confusion matrix
        for t, p in zip(hard.view(-1), pred1.view(-1)):
            cm[t.long(), p.long()] += 1

    # 매크로 지표
    recalls, precisions, f1s = [], [], []
    for k in range(C):
        tp = cm[k, k].item()
        fn = cm[k, :].sum().item() - tp
        fp = cm[:, k].sum().item() - tp
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        f1   = 2 * prec * rec / (prec + rec + 1e-12)
        precisions.append(prec); recalls.append(rec); f1s.append(f1)

    acc           = correct1 / max(1, total)
    top2          = correct2 / max(1, total)
    macro_f1      = float(sum(f1s) / C)
    balanced_acc  = float(sum(recalls) / C)
    val_loss      = total_loss / max(1, total)

    out = {
        "ValLoss": val_loss,
        "Acc": acc,
        "Top2": top2,
        "MacroF1": macro_f1,
        "BalancedAcc": balanced_acc,
        "CM": cm.tolist(),
    }

    if REPORT_REGRESSION_AS_AUX:
        # (보조) 기대값 복원 기반 연속지표
        centers = BIN_CENTERS.to(DEVICE).view(1, -1)
        Ys, Ps = [], []
        for batch in loader:
            x = batch[0].to(DEVICE)
            y_true_scalar = torch.tensor(batch[2], dtype=torch.float32).view(-1, 1).to(DEVICE)
            logits = model(x); prob = F.softmax(logits, dim=1)
            pred_y = (prob * centers).sum(dim=1, keepdim=True)
            if CLIP_PREDICTIONS:
                pred_y = pred_y.clamp(min=RANGE_MIN, max=RANGE_MAX)
            Ys.append(y_true_scalar); Ps.append(pred_y)
        if Ys:
            y = torch.cat(Ys, 0).cpu(); p = torch.cat(Ps, 0).cpu()
            huber = nn.SmoothL1Loss()(p, y).item()
            mae  = (p - y).abs().mean().item()
            mse  = ((p - y)**2).mean().item()
            rmse = (((p - y)**2).mean().sqrt()).item()
            y_mean = y.mean()
            ss_res = ((y - p)**2).sum()
            ss_tot = ((y - y_mean)**2).sum() + 1e-12
            r2 = (1 - ss_res/ss_tot).item()
            out.update({"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "Huber": huber})

    return out

# ===== 12) 로깅/체크포인트/시각화 =====
def log_csv_row(path: Path, epoch, split, metrics: dict, train_loss=None):
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new:
            cols = ["epoch","split","train_loss","ValLoss","Acc","Top2","MacroF1","BalancedAcc"]
            if REPORT_REGRESSION_AS_AUX:
                cols += ["MAE","MSE","RMSE","R2","Huber"]
            w.writerow(cols)
        row = [
            epoch, split,
            f"{train_loss:.6f}" if train_loss is not None else "",
            f"{metrics.get('ValLoss',0.0):.6f}",
            f"{metrics.get('Acc',0.0):.6f}",
            f"{metrics.get('Top2',0.0):.6f}",
            f"{metrics.get('MacroF1',0.0):.6f}",
            f"{metrics.get('BalancedAcc',0.0):.6f}",
        ]
        if REPORT_REGRESSION_AS_AUX:
            row += [
                f"{metrics.get('MAE',0.0):.6f}",
                f"{metrics.get('MSE',0.0):.6f}",
                f"{metrics.get('RMSE',0.0):.6f}",
                f"{metrics.get('R2',0.0):.6f}",
                f"{metrics.get('Huber',0.0):.6f}",
            ]
        w.writerow(row)

def save_ckpt(path: Path, model, optim, sched, epoch, best_key_value):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "sched": sched.state_dict() if sched is not None else None,
        "best_key_value": best_key_value,
    }, path)

def denormalize_to_uint8(x: torch.Tensor, mean=MEAN, std=STD):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
    m = torch.tensor(mean).view(3,1,1)
    s = torch.tensor(std).view(3,1,1)
    img = (x * s + m).clamp(0,1).permute(1,2,0).numpy()
    img = (img * 255.0).round().astype(np.uint8)
    return img

@torch.no_grad()
def visualize_random_samples(model, items, k=3, out_dir: Path = VIZ_DIR, title_prefix="val"):
    if len(items) == 0: return
    out_dir.mkdir(parents=True, exist_ok=True)
    picks = random.sample(items, k=min(k, len(items)))
    model.eval()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers = BIN_CENTERS.to(DEVICE).view(1,-1)
    for i, (path, gt) in enumerate(picks, 1):
        x = preprocess_npy(path).unsqueeze(0).to(DEVICE)
        logits = model(x); prob = F.softmax(logits, dim=1)
        pred = (prob * centers).sum(dim=1, keepdim=True).squeeze()
        if CLIP_PREDICTIONS:
            pred = pred.clamp(min=RANGE_MIN, max=RANGE_MAX)
        pred_val = float(pred.detach().cpu().item())

        img = denormalize_to_uint8(x.squeeze(0))
        plt.figure(figsize=(8,5)); plt.imshow(img); plt.axis('off')
        plt.title(f"{title_prefix}: pred={pred_val:.2f}, gt={gt:.2f}", fontsize=14)
        save_path = out_dir / f"{title_prefix}_sample_{i}.png"
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1); plt.close()

# ===== 13) 학습 루프 (분류 중심 얼리스탑: Macro-F1) =====
def train(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, wd=WEIGHT_DECAY, patience=EARLY_STOP_PATIENCE):
    model = model.to(DEVICE)

    if USE_SOFT_LABEL:
        def criterion(logits, target_prob): return soft_ce_loss(logits, target_prob)
    else:
        def criterion(logits, target_cls):   return F.cross_entropy(logits, target_cls)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    best_score = -1.0
    best_state = None
    wait = 0

    for ep in range(1, epochs + 1):
        model.train()
        run_loss, seen = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs}", leave=False)

        for batch in pbar:
            x = batch[0].to(DEVICE)
            if USE_SOFT_LABEL:
                y_target = batch[1]
                if not isinstance(y_target, torch.Tensor):
                    y_target = torch.stack(y_target, dim=0)
                y_target = y_target.to(DEVICE)                   # [B,K]
            else:
                y_target = batch[1].to(DEVICE)                   # [B]

            logits = model(x)
            loss = criterion(logits, y_target)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            bs = x.size(0)
            run_loss += loss.item() * bs
            seen += bs
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        sched.step()
        tr_loss = run_loss / max(1, seen)

        # 검증(분류 중심)
        val = evaluate(model, val_loader)
        print(f"[Ep {ep:02d}] TL {tr_loss:.4f} | "
              f"Acc {val['Acc']:.4f} Top2 {val['Top2']:.4f} | "
              f"MacroF1 {val['MacroF1']:.4f} BalAcc {val['BalancedAcc']:.4f} | "
              f"ValLoss {val['ValLoss']:.4f}")
        if REPORT_REGRESSION_AS_AUX and 'MAE' in val:
            print(f"          (aux) MAE {val['MAE']:.3f} RMSE {val['RMSE']:.3f} R2 {val['R2']:.3f}")

        log_csv_row(LOG_CSV, ep, "val", val, train_loss=tr_loss)

        score = val["MacroF1"]  # 얼리스탑 기준
        if score > best_score + 1e-6:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_ckpt(CKPT_BEST, model, optim, sched, ep, best_key_value=score)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"[EarlyStop] No val improvement for {patience} epochs. Stop at ep={ep}.")
                save_ckpt(CKPT_LAST, model, optim, sched, ep, best_key_value=best_score)
                break

        visualize_random_samples(model, val_items, k=3, out_dir=VIZ_DIR/("epoch_%02d" % ep),
                                 title_prefix=f"val_ep{ep:02d}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# ===== 14) 메인 =====
if __name__ == "__main__":
    set_seed(SEED)

    train_rows = load_label_rows_exact(LABEL_TRAIN)
    val_rows   = load_label_rows_exact(LABEL_VAL)
    test_rows  = load_label_rows_exact(LABEL_TEST)

    # 학습/평가 범위 제한 (40~80)
    train_rows = filter_rows_by_range(train_rows, RANGE_MIN, RANGE_MAX)
    val_rows   = filter_rows_by_range(val_rows,   RANGE_MIN, RANGE_MAX)
    test_rows  = filter_rows_by_range(test_rows,  RANGE_MIN, RANGE_MAX)

    train_items = rows_to_items(train_rows, NPY_DIR_TRAIN, include_expanded=INCLUDE_EXPANDED)
    val_items   = rows_to_items(val_rows,   NPY_DIR_VAL,   include_expanded=INCLUDE_EXPANDED)
    test_items  = rows_to_items(test_rows,  NPY_DIR_TEST,  include_expanded=INCLUDE_EXPANDED)

    print("[DEBUG] items | train:", len(train_items), "val:", len(val_items), "test:", len(test_items))
    assert len(train_items) > 0, "Train 매칭 0개입니다. stem 패턴/경로를 확인하세요."

    train_loader = make_loader(train_items, train=True)
    val_loader   = make_loader(val_items,   train=False)
    test_loader  = make_loader(test_items,  train=False)

    model = build_mobilenet_v3_small_cls(pretrained=PRETRAINED)
    model = train(model, train_loader, val_loader)

    m = evaluate(model, test_loader)
    print("[TEST] "
          f"Acc {m['Acc']:.4f} | Top2 {m['Top2']:.4f} | MacroF1 {m['MacroF1']:.4f} | "
          f"BalancedAcc {m['BalancedAcc']:.4f}")
    if REPORT_REGRESSION_AS_AUX and 'MAE' in m:
        print("       (aux) MAE %.4f | RMSE %.4f | R2 %.4f | Huber %.4f"
              % (m['MAE'], m['RMSE'], m['R2'], m['Huber']))

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    visualize_random_samples(model, test_items, k=3, out_dir=VIZ_DIR/("test_examples"),
                             title_prefix="test")
