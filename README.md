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



<img width="944" height="967" alt="image" src="https://github.com/user-attachments/assets/11227722-d0a2-45af-8e48-7fb44b2ca649" />



---------

<img width="886" height="556" alt="image" src="https://github.com/user-attachments/assets/7d408b75-fa05-4ed8-978b-2a6b765c301d" />

