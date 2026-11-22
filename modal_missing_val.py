# # 验证横轴缺失（只用纵轴）
# python modal_missing_val.py \
#     --model ConvNeXt \
#     --fold 0 \
#     --missing_view trans
#
# # 验证纵轴缺失（只用横轴）
# python modal_missing_val.py \
#     --model ConvNeXt \
#     --fold 0 \
#     --missing_view long
#
# # 同时验证两种缺失情况
# python modal_missing_val.py \
#     --model ConvNeXt \
#     --fold 0 \
#     --missing_view both
#
# # 使用Cascaded模型
# python modal_missing_val.py \
#     --model Cascaded \
#     --fold 0 \
#     --missing_view both \
#     --save_path ./checkpoints

import argparse
import logging
import os
import sys
import numpy as np
from typing import Tuple, Dict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix

from dataset.csv import CSVSemiDataset
from util.utils import compute_nsd

from model.convnext import ConvNeXt_UniMatch
from model.CascadedUnimatchv3 import CascadedUniMatch


def build_logger(save_path: str):
    """构建日志记录器"""
    logger = logging.getLogger("Modal Missing Validation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    os.makedirs(save_path, exist_ok=True)

    fh = logging.FileHandler(os.path.join(save_path, "modal_missing_val.log"))
    fh.setFormatter(logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S"))

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def ensure_cls_shape(y_cls: torch.Tensor) -> torch.Tensor:
    """确保分类标签形状为 [B, C]"""
    if not torch.is_tensor(y_cls):
        y_cls = torch.as_tensor(y_cls)
    if y_cls.ndim == 0:
        y_cls = y_cls.view(1, 1)
    elif y_cls.ndim == 1:
        y_cls = y_cls.unsqueeze(1)
    return y_cls


def get_model(args, device):
    """
    根据args.model构建模型并加载checkpoint
    """
    if args.model == "ConvNeXt":
        model = ConvNeXt_UniMatch(
            in_chns=1,
            seg_class_num=args.seg_num_classes,
            cls_class_num=args.cls_num_classes,
            encoder_pth=args.convnext_ckpt,
        )
    elif args.model == "Cascaded":
        model = CascadedUniMatch(
            in_chns=1,
            seg_class_num=args.seg_num_classes,
            cls_class_num=args.cls_num_classes,
            encoder_pth=args.convnext_ckpt,
            unet_base_ch=32,
            attn_heads=4,
        )
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    # 加载checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    return model


@torch.no_grad()
def validate_modal_missing(
        args,
        model,
        valid_loader,
        device,
        logger,
        missing_view: str = "trans",  # "trans" or "long"
):
    """
    模态缺失验证

    Args:
        args: 命令行参数
        model: 模型
        valid_loader: 验证集dataloader
        device: 设备
        logger: 日志记录器
        missing_view: 缺失的视角 ("trans" 横轴缺失, "long" 纵轴缺失)
    """
    model.eval()

    # 分割指标累加器
    dice_long = {1: 0.0, 2: 0.0}
    dice_trans = {1: 0.0, 2: 0.0}
    nsd_long = {1: 0.0, 2: 0.0}
    nsd_trans = {1: 0.0, 2: 0.0}

    # 分类指标
    cls_pred_list = []
    cls_gt_list = []

    num_batches = len(valid_loader)

    logger.info(f"=" * 60)
    logger.info(f"Modal Missing Validation: {missing_view} view MISSING")
    logger.info(f"Model: {args.model}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Fold: {args.fold}")
    logger.info(f"=" * 60)

    for batch_idx, (x_long, x_trans, m_long, m_trans, y_cls) in enumerate(valid_loader):
        x_long = x_long.to(device)
        x_trans = x_trans.to(device)
        m_long = m_long.to(device)
        m_trans = m_trans.to(device)
        y_cls = ensure_cls_shape(y_cls).to(device)

        # 原始尺寸
        hL, wL = x_long.shape[-2:]
        hT, wT = x_trans.shape[-2:]

        # 模态缺失：将指定视角置零
        if missing_view == "trans":
            x_trans = torch.zeros_like(x_trans)
            logger.debug(f"Batch {batch_idx}: Trans view zeroed out")
        elif missing_view == "long":
            x_long = torch.zeros_like(x_long)
            logger.debug(f"Batch {batch_idx}: Long view zeroed out")
        else:
            raise ValueError(f"missing_view must be 'trans' or 'long', got {missing_view}")

        # Resize到目标尺寸
        x_long_r = F.interpolate(
            x_long, (args.resize_target, args.resize_target),
            mode="bilinear", align_corners=False
        )
        x_trans_r = F.interpolate(
            x_trans, (args.resize_target, args.resize_target),
            mode="bilinear", align_corners=False
        )

        # 前向推理
        segL, segT, cls_out = model(x_long_r, x_trans_r)

        # 分类预测
        cls_prob = torch.sigmoid(cls_out)
        cls_pred = (cls_prob >= 0.5).long().view(-1)
        cls_pred_list.extend(cls_pred.cpu().numpy().tolist())
        cls_gt_list.extend(y_cls.view(-1).cpu().numpy().tolist())

        # 分割结果上采样回原始尺寸
        segL_up = F.interpolate(segL, (hL, wL), mode="bilinear", align_corners=False)
        segT_up = F.interpolate(segT, (hT, wT), mode="bilinear", align_corners=False)

        predL = torch.argmax(segL_up, dim=1)
        predT = torch.argmax(segT_up, dim=1)

        # 计算Dice和NSD
        for cls_id in [1, 2]:
            # Long view
            interL = ((predL == cls_id) & (m_long == cls_id)).sum().item()
            unionL = (predL == cls_id).sum().item() + (m_long == cls_id).sum().item()
            diceL = 2.0 * interL / (unionL + 1e-8)
            dice_long[cls_id] += diceL

            # Trans view
            interT = ((predT == cls_id) & (m_trans == cls_id)).sum().item()
            unionT = (predT == cls_id).sum().item() + (m_trans == cls_id).sum().item()
            diceT = 2.0 * interT / (unionT + 1e-8)
            dice_trans[cls_id] += diceT

            # NSD
            predL_np = (predL[0] == cls_id).cpu().numpy()
            gtL_np = (m_long[0] == cls_id).cpu().numpy()
            nsd_long[cls_id] += compute_nsd(predL_np, gtL_np, tolerance=3.0)

            predT_np = (predT[0] == cls_id).cpu().numpy()
            gtT_np = (m_trans[0] == cls_id).cpu().numpy()
            nsd_trans[cls_id] += compute_nsd(predT_np, gtT_np, tolerance=3.0)

        # 进度显示
        if (batch_idx + 1) % max(1, num_batches // 10) == 0:
            logger.info(f"Progress: {batch_idx + 1}/{num_batches}")

    # ========== 汇总结果 ==========
    idx_to_name = {1: "Plaque", 2: "Vessel"}

    # 平均Dice
    for cls_id in [1, 2]:
        dice_long[cls_id] /= max(1, num_batches)
        dice_trans[cls_id] /= max(1, num_batches)
        nsd_long[cls_id] /= max(1, num_batches)
        nsd_trans[cls_id] /= max(1, num_batches)

    # 分类F1
    cls_gt = np.array(cls_gt_list)
    cls_pred = np.array(cls_pred_list)
    f1 = f1_score(cls_gt, cls_pred)

    # 混淆矩阵
    cm = confusion_matrix(cls_gt, cls_pred)
    labels = list(range(cm.shape[0]))

    # ========== 打印结果 ==========
    logger.info("\n" + "=" * 60)
    logger.info(f"RESULTS - {missing_view.upper()} view MISSING")
    logger.info("=" * 60)

    logger.info("\n--- Segmentation Dice ---")
    for cls_id in [1, 2]:
        logger.info(
            f"{idx_to_name[cls_id]:<8} | Long: {dice_long[cls_id]:.4f} | Trans: {dice_trans[cls_id]:.4f}"
        )

    logger.info("\n--- Segmentation NSD ---")
    for cls_id in [1, 2]:
        logger.info(
            f"{idx_to_name[cls_id]:<8} | Long: {nsd_long[cls_id]:.4f} | Trans: {nsd_trans[cls_id]:.4f}"
        )

    mean_dice = (dice_long[1] + dice_long[2] + dice_trans[1] + dice_trans[2]) / 4.0
    mean_nsd = (nsd_long[1] + nsd_long[2] + nsd_trans[1] + nsd_trans[2]) / 4.0

    logger.info(f"\nMean Foreground Dice: {mean_dice:.4f}")
    logger.info(f"Mean Foreground NSD: {mean_nsd:.4f}")

    logger.info(f"\n--- Classification ---")
    logger.info(f"F1 Score: {f1:.4f}")

    # 格式化混淆矩阵
    logger.info("Confusion Matrix (rows=true, cols=pred):")
    header = "      " + "".join([f"  Pred:{l}  " for l in labels])
    logger.info(header)
    for i, l in enumerate(labels):
        row_total = cm[i].sum()
        if row_total > 0:
            row_str = f"True:{l} " + "".join(
                [f"  {cm[i, j]:3d} ({cm[i, j] / row_total * 100:5.1f}%)" for j in range(cm.shape[1])]
            )
        else:
            row_str = f"True:{l} " + "".join([f"  {cm[i, j]:3d} ( 0.0%)" for j in range(cm.shape[1])])
        logger.info(row_str)

    # 返回结果字典
    return {
        "missing_view": missing_view,
        "dice_long_vessel": dice_long[2],
        "dice_long_plaque": dice_long[1],
        "dice_trans_vessel": dice_trans[2],
        "dice_trans_plaque": dice_trans[1],
        "nsd_long_vessel": nsd_long[2],
        "nsd_long_plaque": nsd_long[1],
        "nsd_trans_vessel": nsd_trans[2],
        "nsd_trans_plaque": nsd_trans[1],
        "mean_dice": mean_dice,
        "mean_nsd": mean_nsd,
        "cls_f1": f1,
        "confusion_matrix": cm,
    }


def main():
    parser = argparse.ArgumentParser("Modal Missing Validation for Two-View Models")

    # 数据相关
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory containing fold_X folders")
    parser.add_argument("--fold", type=int, default=0,
                        help="Which fold to validate")
    parser.add_argument("--resize_target", type=int, default=256)
    parser.add_argument("--seg_num_classes", type=int, default=3)
    parser.add_argument("--cls_num_classes", type=int, default=1)

    # 模型相关
    parser.add_argument("--model", type=str, default="ConvNeXt",
                        choices=["ConvNeXt", "Cascaded"],
                        help="Model architecture")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (if None, auto-generate from save_path/fold_X/best.pth)")
    parser.add_argument("--save_path", type=str, default="./checkpoints",
                        help="Base checkpoint directory")
    parser.add_argument("--convnext_ckpt", type=str, default="./pretrain/convnext_tiny_22k_1k_224.pth",
                        help="Path to ConvNeXt pretrained weights")

    # 验证配置
    parser.add_argument("--missing_view", type=str, default="both",
                        choices=["trans", "long", "both"],
                        help="Which view to miss: 'trans', 'long', or 'both'")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (recommend 1 for reliable metrics)")

    args = parser.parse_args()

    # GPU设置
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 构建数据路径
    fold_dir = os.path.join(args.data_root, f"fold_{args.fold}")
    valid_json = os.path.join(fold_dir, "valid.json")

    if not os.path.exists(valid_json):
        raise FileNotFoundError(f"Validation file not found: {valid_json}")

    # 如果checkpoint未指定，自动构建
    if args.checkpoint is None:
        args.checkpoint = os.path.join(args.save_path, f"fold_{args.fold}", "best.pth")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # 构建日志目录
    log_dir = os.path.join(args.save_path, f"fold_{args.fold}", "modal_missing_logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = build_logger(log_dir)

    logger.info("=" * 60)
    logger.info("Modal Missing Validation Configuration:")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Checkpoint: {args.checkpoint}")
    logger.info(f"  Fold: {args.fold}")
    logger.info(f"  Missing View: {args.missing_view}")
    logger.info(f"  Device: {device}")
    logger.info("=" * 60)

    # 加载模型
    model = get_model(args, device)
    logger.info(f"Model loaded successfully")

    # 加载数据
    db_valid = CSVSemiDataset(valid_json, "valid")
    valid_loader = DataLoader(
        db_valid,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True
    )
    logger.info(f"Validation set size: {len(db_valid)}")

    # 执行验证
    if args.missing_view == "both":
        # 分别验证两种缺失情况
        for mv in ["trans", "long"]:
            results = validate_modal_missing(
                args, model, valid_loader, device, logger, missing_view=mv
            )

            # 保存结果到文件
            result_file = os.path.join(log_dir, f"results_{mv}_missing.txt")
            with open(result_file, "w") as f:
                f.write(f"Model: {args.model}\n")
                f.write(f"Checkpoint: {args.checkpoint}\n")
                f.write(f"Fold: {args.fold}\n")
                f.write(f"Missing View: {mv}\n")
                f.write(f"Mean Dice: {results['mean_dice']:.4f}\n")
                f.write(f"Mean NSD: {results['mean_nsd']:.4f}\n")
                f.write(f"Cls F1: {results['cls_f1']:.4f}\n")
                f.write(f"Confusion Matrix:\n{results['confusion_matrix']}\n")
            logger.info(f"Results saved to: {result_file}")
    else:
        # 验证单一缺失情况
        results = validate_modal_missing(
            args, model, valid_loader, device, logger, missing_view=args.missing_view
        )

        result_file = os.path.join(log_dir, f"results_{args.missing_view}_missing.txt")
        with open(result_file, "w") as f:
            f.write(f"Model: {args.model}\n")
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"Fold: {args.fold}\n")
            f.write(f"Missing View: {args.missing_view}\n")
            f.write(f"Mean Dice: {results['mean_dice']:.4f}\n")
            f.write(f"Mean NSD: {results['mean_nsd']:.4f}\n")
            f.write(f"Cls F1: {results['cls_f1']:.4f}\n")
            f.write(f"Confusion Matrix:\n{results['confusion_matrix']}\n")
        logger.info(f"Results saved to: {result_file}")

    logger.info("\nModal missing validation completed!")


if __name__ == "__main__":
    main()