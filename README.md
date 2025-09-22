# oil-moisture

### ■ 연구계획
<img width="1722" height="920" alt="image" src="https://github.com/user-attachments/assets/6151136d-afec-4642-a261-aff1c9d8071f" />






##### ※ Regression -> classfication 학습으로 변경 ( 40-80 범주 제한, bin = 10, class = 4)

## ■ 변경 사항 및 성능 개선 과정
### ◆ MobileNetV3-Small

##### [9/15] (1)

--------------------------------------------------------------------------

bin = 10 

BATCH_SIZE = 32

NUM_WORKERS = 4 

EPOCHS = 50

LR = 3e-4

WEIGHT_DECAY = 1e-4

EARLY_STOP_PATIENCE = 7

SEED = 2025


- LOSS FUNCTION : SmoothL1Loss()

- OPTIMIZER : Adam

- SCHEDULER : CosineAnnealingLR
--------------------------------------------------------------------------





##### [9/19] (2)

--------------------------------------------------------------------------
bin = 10 

BATCH_SIZE = 32

NUM_WORKERS = 4

EPOCHS = 50

LR = 3e-4

WEIGHT_DECAY = 1e-4

EARLY_STOP_PATIENCE = 7

SEED = 2025


- LOSS FUNCTION : CrossEntropy()
- OPTIMIZER : Adam
- SCHEDULER : CosineAnnealingLR

--------------------------------------------------------------------------
=> [Ep 24] TL 0.1088 | Acc 0.3917 Top2 0.7119 | MacroF1 0.3461 BalAcc 0.3464 | ValLoss 2.4290
=> [TEST] Acc 0.3955 | Top2 0.7170 | MacroF1 0.3513 | BalancedAcc 0.3507

<img width="944" height="967" alt="image" src="https://github.com/user-attachments/assets/11227722-d0a2-45af-8e48-7fb44b2ca649" />



---------




### * 1.과적합 우려 ep 50 -> 80 변경
### * 2. LR 3e-4 -> 2e-4 변경 /  weight_decay 1e-4 -> 5e-4 


##### [9/21] (3)

--------------------------------------------------------------------------
bin = 10 
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 80
LR = 2e-4
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE =10
SEED = 2025

- LOSS FUNCTION : CrossEntropy()
- OPTIMIZER : Adam
- SCHEDULER : CosineAnnealingLR

--------------------------------------------------------------------------
=> early stopping x

=> [Ep 80] TL 1.4430 | Acc 0.4413 Top2 0.6223 | MacroF1 0.3702 BalAcc 0.3637 | ValLoss 1.4430

=> [TEST] Acc 0.4143 | Top2 0.7027 | MacroF1 0.3272 | BalancedAcc 0.3302 | Prec 0.3485 | Rec 0.3302 | AUC(ovr-macro) 0.6214

<img width="1232" height="793" alt="plot_loss" src="https://github.com/user-attachments/assets/a139fe9d-57c1-4bc6-94b5-6d899910678c" />
=> 과적합 징후 살짝 보임 (의심)


### [수정사항]---------------------
### * 1. 과적합 방지  : EARLY_STOP_PATIENCE = 10 -> 8 
### * 2.**LABEL_SMOOTHING = 0.10 -> 0.05  **
### * 3. WEIGHT_DECAY = 5e-5 ->1e-4
### * 4.스케줄러 교체 코사인(CosineAnnealingLR) -> ReduceLROnPlateau
###  5. 기대값→bin 매핑 (한 줄로 효과)
###  6. TTA (테스트/검증만)
### 7. 경계 가중치 (훈련 루프에 몇 줄)
###  8. Ordinal 학습 (가장 강력, 구조 변경 수반)
#-------------------------------------------------




<img width="853" height="1111" alt="image" src="https://github.com/user-attachments/assets/b0084106-b4c1-4f2d-b715-2b45fc67e2e9" />



##### [9/22] (4)

--------------------------------------------------------------------------
bin = 10 

BATCH_SIZE = 32

NUM_WORKERS = 4

EPOCHS = 80

##### LR = 2e-4

WEIGHT_DECAY = 1e-4

##### EARLY_STOP_PATIENCE = 8

SEED = 2025

- LOSS FUNCTION : CrossEntropy()
- OPTIMIZER : AdamW
-  SCHEDULER : #####  ReduceLROnPlateau

--------------------------------------------------------------------------
=> 

=> 

=>
