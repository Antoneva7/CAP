# 🩺 CAP

This repository provides the official baseline implementation for the **CSV Challenge**, It implements a semi-supervised UniMatch pipeline for **Carotid Plaque Segmentation and Vulnerability Assessment in Ultrasound**. For official rules, dataset downloads and the evaluation server, visit the [CSV 2026 challenge github page](https://github.com/dndins/CSV-2026-Baseline)

---

## 📁 1. Prepare Data

Place the downloaded 🖼️ training archive (provided by organizers) and unzip it to the 'data/' directory. 
**🤖 Pre-trained Weights**: For Echocare model training, download the pre-trained Echocare encoder weights from [this link](https://cashkisi-my.sharepoint.com/:u:/g/personal/cares-copilot_cair-cas_org_hk/IQBgK6rK8TAtQq8IjADsgp52AbmyC03ubimwqr3qh8ZH6DI?e=ABYQzg) and place the `echocare_encoder.pth` file in the `pretrain/` directory. so the directory structure becomes:
```text
CSV2026_Baseline/
├─ data/
|  └─ train/
|     ├─ images/        # .h5 image files (long_img & trans_img)
|     └─ labels/        # _label.h5 files (long_mask, trans_mask, cls)
└─ pretrain/ 
   └─ echocare_encoder.pth    # Pre-trained Echocare encoder weights
```

## 🧰 2. Quick Start
 We recommend **Python 3.10** and **CUDA 12.1 (cu121)**.
 Minimum recommended PyTorch version: **>= 2.4.1**.

```bash
# 📥 Clone the repository
git clone https://github.com/dndins/CSV-2026-Baseline.git
cd CSV-2026-Baseline

# 🎯 Create Python Environment
conda create -n csv-baseline python=3.10 -y
conda activate csv-baseline

pip install --index-url https://download.pytorch.org/whl/cu121   torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

pip install -r requirements.txt
```

> ⚠️ **Important:**  
> If your CUDA version differs, install PyTorch following the official instructions first, then run:
> ```bash
> pip install -r requirements.txt --no-deps
> ```


## 🧩 3. Create Local Train / Validation Split

Generate a balanced validation set and JSON splits:

```bash
python split_train_valid_fold.py --root ./data --seed 2026 --val_size 50 --n_folds 5
```

This creates:
- `train_labeled.json`
- `train_unlabeled.json`
- `valid.json`

under the `data/fole_X/` directory.


## 🚀 4. Train

### 

```bash
python train.py --model Cascaded --gpu 0 --train_epochs 50 --batch_size 4 --fold 0
```

Training checkpoints:
```text
./checkpoints/best.pth
./checkpoints/latest.pth
```


## 🔍 5. Inference

Run inference on released validation images:

```bash
python inference.py \
  --val-dir ./data/val \
  --checkpoint ./checkpoints/best.pth \
  --encoder-pth ./pretrain/echocare_encoder.pth \
  --resize-target 256 \
  --gpu 0
```

Predictions are saved to:

```text
./data/val/preds/{case_name}_pred.h5
```

Each file contains:
- `long_mask`
- `trans_mask`
- `cls`




---

## 🏁 Good Luck & Happy Research!

We look forward to your participation in the CSV 2026 Challenge 🚀
