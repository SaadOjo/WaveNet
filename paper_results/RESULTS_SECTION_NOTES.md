# Results Section Notes

## Scope used for the focused workbook
The focused comparison workbook is restricted to the **common-coverage IC1 datasets**:

- `data10 (IC1, 1:1, full)`
- `data11 (IC1, 1:4, full)`
- `data12 (IC1, 1:1, max5000)`
- `data13 (IC1, 1:1, max2500)`

This restriction was chosen because the selected paper-facing model set has complete results for these rows.

Workbook path:
- `paper_results/focused_ic1_results.xlsx`

## Model columns used in the focused workbook
The workbook contains only the following columns:

- `Random Forest`
- `XGBoost`
- `SmallWaveNetTL`
- `ViT`

### Important mapping note
For the focused workbook, the `SmallWaveNetTL` column is populated from the **refactored and reproducible** run named `SmallWaveNetV3_TL`.

This was done intentionally because:
1. the original legacy `SmallWaveNetTL` notebook result is not fully reproducible from the refactored codebase,
2. the refactored `SmallWaveNetV3_TL` is the closest clean rerun of the small wavelet-CNN transfer-learning idea,
3. the observed performance pattern is close to the legacy small-wavelet transfer result.

This mapping should be disclosed explicitly in the manuscript if the paper uses this focused table.

## Data representation used by the neural models
For the wavelet-based neural models in the refactored repo:

1. each sample is read from the exported `.ts` train/test files,
2. the time series is converted to a **continuous wavelet transform (CWT)** image,
3. the CWT uses:
   - wavelet: `morl`
   - scales: `1..20`
4. the refactored code caches the **raw CWT tensors** and applies later transforms at runtime.

## Exact model/training definitions

### Random Forest
Source:
- `container_workspace/scripts/random_forest.py`
- `container_workspace/src/models/classical.py`

Definition:
- `RandomForestClassifier(n_estimators=100, random_state=42)`

Input:
- flattened time-series features from the `.ts` files
- no wavelet image conversion

### XGBoost
Source:
- `container_workspace/scripts/xgboost.py`
- `container_workspace/src/models/classical.py`

Definition:
- `XGBClassifier(`
  - `objective='binary:logistic'`
  - `n_estimators=100`
  - `max_depth=4`
  - `learning_rate=0.1`
  - `subsample=0.8`
  - `colsample_bytree=0.8`
  - `random_state=42`
  - `eval_metric='logloss'`
- `)`

Input:
- flattened time-series features from the `.ts` files
- no wavelet image conversion

### SmallWaveNetTL (implemented as SmallWaveNetV3_TL)
Source:
- `container_workspace/scripts/small_wavenet_v3_tl.py`
- `container_workspace/src/models/cnn.py`

Architecture (`make_small_wavenet_v3(mode='binary')`):
- `Conv2d(1, 32, kernel_size=3, padding=1)`
- `BatchNorm2d(32)`
- `ReLU`
- `Conv2d(32, 32, kernel_size=3, padding=1)`
- `BatchNorm2d(32)`
- `ReLU`
- `MaxPool2d(2, 2)`
- `Dropout(0.2)`
- `Conv2d(32, 64, kernel_size=3, padding=1)`
- `BatchNorm2d(64)`
- `ReLU`
- `Conv2d(64, 64, kernel_size=3, padding=1)`
- `BatchNorm2d(64)`
- `ReLU`
- `MaxPool2d(2, 2)`
- `Dropout(0.2)`
- `Conv2d(64, 128, kernel_size=3, padding=1)`
- `BatchNorm2d(128)`
- `ReLU`
- `Conv2d(128, 128, kernel_size=3, padding=1)`
- `BatchNorm2d(128)`
- `ReLU`
- `Dropout(0.2)`
- `Flatten()`
- `Linear(128 * 5 * 5, 256)`
- `BatchNorm1d(256)`
- `ReLU`
- `Dropout(0.4)`
- `Linear(256, 1)`
- `Sigmoid()`

Input:
- grayscale CWT image
- shape: `1 x 20 x 20`
- no colorization
- no resize
- normalization setting: `norm_type='none'`

Transfer-learning protocol:
- pretraining dataset: `data20`
- pretraining weights path:
  - `container_workspace/weights/base_weights_small_wavenet_v3.pth`
- transfer run loads matching weights into the same architecture and fine-tunes on the target dataset

Training settings:
- batch size: `64`
- epochs: `150`
- optimizer: `Adam`
- learning rate: `1e-3`
- weight decay: `1e-4`
- scheduler: `StepLR(step_size=8, gamma=0.3)`
- patience: `40`
- label smoothing: `0.0`

### ViT (implemented as ViT_B_16_TransferLearning)
Source:
- `container_workspace/scripts/wavelet_vit.py`
- `container_workspace/src/models/vit.py`

Backbone:
- `torchvision.models.vit_b_16`
- weights: `ViT_B_16_Weights.DEFAULT`

Trainable layers:
- all parameters are frozen first,
- then only:
  - `encoder.layers[-1]`
  - the replacement classification head
  are trainable.

Classification head:
- `Dropout(0.5)`
- `Linear(num_ftrs, 1)`
- `Sigmoid()`

Input:
- CWT image
- resized to `224 x 224`
- colorized to 3 channels
- normalization setting: `norm_type='none'`

Training settings:
- batch size: `16`
- epochs: `100`
- optimizer: `Adam`
- learning rate: `1e-4`
- weight decay: `1e-4`
- scheduler: `StepLR(step_size=8, gamma=0.3)`
- patience: `20`
- label smoothing: `0.1`
- `optimizer_trainable_only=True`

Pretraining source:
- ImageNet only
- the focused workbook does **not** use the unfinished churn-pretrained ViT-TL20 runs

## Process followed to optimize the small wavelet model
The model-development path in this repo was:

1. keep the classical baselines (`Random Forest`, `XGBoost`) as non-image references,
2. preserve the old wavelet results imported from the pre-refactor notebooks,
3. re-implement a stronger small wavelet CNN based on the old notebook's **v3** architecture,
4. run the refactored no-transfer version (`SmallWaveNetV3`),
5. run a churn-to-churn transfer version (`SmallWaveNetV3_TL`) by pretraining on `data20`,
6. compare that transfer version against the classical baselines and the ViT baseline on the common IC1 rows.

## Evaluation protocol caveat (important for the paper)
The current refactored neural training code uses the provided dataset `TRAIN` / `TEST` split as follows:

- training is performed on the provided `TRAIN` split,
- the provided `TEST` split is also used during training for model selection / early stopping,
- the final reported metrics are computed from the best checkpoint under that same held-out split.

So the current neural results are **best-checkpoint metrics on the provided test split**, not last-epoch metrics.

This should be described accurately if these numbers are reported in the manuscript.

## Reproducibility note
The refactored neural scripts do not currently set a global PyTorch seed inside the run scripts. Therefore, some variation relative to older notebook runs is expected.
