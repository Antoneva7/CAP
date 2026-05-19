import os
import json
import h5py
import random
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Generate train/valid JSON for semi-supervised CSV 2026 challenge")

    parser.add_argument("--root", type=str, default="./data", help="Dataset root path")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument("--val_size", type=int, default=50, help="Number of labeled samples to reserve for validation (will be balanced 1:1 between classes)")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--fold", type=int, default=-1,
                        help="Specific fold to generate (0 to n_folds-1). If -1, generate all folds.")
    return parser.parse_args()


def generate_fold_split(root_path, fold_idx, n_folds, val_size, seed, output_dir=None):
    """
    生成单个折的训练/验证集划分

    Args:
        root_path: 数据根目录
        fold_idx: 当前折索引 (0 ~ n_folds-1)
        n_folds: 总折数
        val_size: 每个折的验证集大小
        seed: 随机种子
        output_dir: 输出目录，默认为 root_path/fold_{fold_idx}
    """
    if output_dir is None:
        output_dir = os.path.join(root_path, f'fold_{fold_idx}')
    os.makedirs(output_dir, exist_ok=True)

    # 设置该折的随机种子（保证可复现）
    fold_seed = seed + fold_idx
    random.seed(fold_seed)

    # ===== 以下代码基本保持原样，但修改输出路径 =====
    images_dir_path = os.path.join(root_path, 'train', 'images')
    labels_dir_path = os.path.join(root_path, 'train', 'labels')

    # 收集所有图像和标签
    all_image_filenames = [name for name in os.listdir(images_dir_path) if name.endswith('.h5')]
    all_labeled_filenames = [name.replace('_label', '') for name in os.listdir(labels_dir_path) if name.endswith('.h5')]
    all_unlabeled_filenames = [name for name in all_image_filenames if name not in all_labeled_filenames]

    # 构建有标签数据集，按类别分组
    train_labeled_dataset_list = []
    class0_list = []
    class1_list = []

    for label_filenames in all_labeled_filenames:
        image_h5_file_path = os.path.abspath(os.path.join(images_dir_path, label_filenames))
        label_h5_file_path = os.path.abspath(os.path.join(labels_dir_path, label_filenames.replace('.h5', '_label.h5')))
        entry = {'image': image_h5_file_path, 'label': label_h5_file_path}

        # 读取类别
        try:
            with h5py.File(label_h5_file_path, 'r') as hf:
                cls_raw = hf['cls'][()]
                try:
                    cls_val = int(cls_raw)
                except Exception:
                    cls_val = int(cls_raw[0]) if hasattr(cls_raw, '__getitem__') else int(cls_raw)
        except Exception:
            cls_val = 0

        train_labeled_dataset_list.append(entry)
        if cls_val == 0:
            class0_list.append(entry)
        else:
            class1_list.append(entry)

    # 无标签数据（所有折共享，保持不变）
    train_unlabeled_dataset_list = []
    for label_filenames in all_unlabeled_filenames:
        image_h5_file_path = os.path.abspath(os.path.join(images_dir_path, label_filenames))
        train_unlabeled_dataset_list.append({'image': image_h5_file_path, 'label': None})

    # ===== 关键：K折划分有标签数据 =====
    # 对每个类别分别进行K折划分
    n_classes = 2
    per_class_val = val_size // 2

    # 对每个类别分别创建折的索引
    class0_indices = list(range(len(class0_list)))
    class1_indices = list(range(len(class1_list)))

    random.shuffle(class0_indices)
    random.shuffle(class1_indices)

    # 计算每个折应该分到的样本数
    n_class0_per_fold = len(class0_list) // n_folds
    n_class1_per_fold = len(class1_list) // n_folds

    # 当前折的验证集索引范围
    start0 = fold_idx * n_class0_per_fold
    end0 = (fold_idx + 1) * n_class0_per_fold if fold_idx < n_folds - 1 else len(class0_list)
    start1 = fold_idx * n_class1_per_fold
    end1 = (fold_idx + 1) * n_class1_per_fold if fold_idx < n_folds - 1 else len(class1_list)

    # 如果指定了val_size且小于每个折的样本数，则从折内再采样
    if per_class_val > 0 and per_class_val < n_class0_per_fold:
        # 从当前折中随机采样 per_class_val 个作为验证集
        fold_class0_indices = class0_indices[start0:end0]
        sampled_indices0 = random.sample(fold_class0_indices, per_class_val)
        sampled0 = [class0_list[i] for i in sampled_indices0]
        # 剩余作为训练集
        train_class0 = [class0_list[i] for i in fold_class0_indices if i not in sampled_indices0]
    else:
        # 整个折作为验证集
        sampled0 = [class0_list[i] for i in class0_indices[start0:end0]]
        train_class0 = [class0_list[i] for i in class0_indices[:start0] + class0_indices[end0:]]

    # 同样处理类别1
    if per_class_val > 0 and per_class_val < n_class1_per_fold:
        fold_class1_indices = class1_indices[start1:end1]
        sampled_indices1 = random.sample(fold_class1_indices, per_class_val)
        sampled1 = [class1_list[i] for i in sampled_indices1]
        train_class1 = [class1_list[i] for i in fold_class1_indices if i not in sampled_indices1]
    else:
        sampled1 = [class1_list[i] for i in class1_indices[start1:end1]]
        train_class1 = [class1_list[i] for i in class1_indices[:start1] + class1_indices[end1:]]

    # 构建当前折的训练集和验证集
    train_labeled_list = train_class0 + train_class1
    valid_list = sampled0 + sampled1

    # 保存JSON（输出到fold目录）
    with open(os.path.join(output_dir, 'train_labeled.json'), 'w') as f:
        json.dump(train_labeled_list, f, indent=4)

    # 无标签数据保持不变，但也要复制到每个fold目录
    with open(os.path.join(output_dir, 'train_unlabeled.json'), 'w') as f:
        json.dump(train_unlabeled_dataset_list, f, indent=4)

    with open(os.path.join(output_dir, 'valid.json'), 'w') as f:
        json.dump(valid_list, f, indent=4)

    # 打印统计信息
    print(f"\n=== Fold {fold_idx} split summary ===")
    print(f"Training labeled: {len(train_labeled_list)} (class0: {len(train_class0)}, class1: {len(train_class1)})")
    print(f"Validation: {len(valid_list)} (class0: {len(sampled0)}, class1: {len(sampled1)})")
    print(f"Unlabeled: {len(train_unlabeled_dataset_list)}")
    print(f"Output directory: {output_dir}")

    return {
        'fold': fold_idx,
        'train_labeled': len(train_labeled_list),
        'train_unlabeled': len(train_unlabeled_dataset_list),
        'valid': len(valid_list)
    }


if __name__ == "__main__":
    args = get_args()

    if args.fold == -1:
        # 生成所有折
        print(f"Generating {args.n_folds} folds...")
        all_stats = []
        for f in range(args.n_folds):
            stats = generate_fold_split(
                root_path=args.root,
                fold_idx=f,
                n_folds=args.n_folds,
                val_size=args.val_size,
                seed=args.seed
            )
            all_stats.append(stats)

        # 打印汇总
        print("\n" + "=" * 50)
        print("All folds generated successfully!")
        print(f"Output directories: {args.root}/fold_0, {args.root}/fold_1, ...")
        print("\nSummary:")
        for s in all_stats:
            print(
                f"  Fold {s['fold']}: labeled={s['train_labeled']}, unlabeled={s['train_unlabeled']}, valid={s['valid']}")

    else:
        # 只生成指定的折
        assert 0 <= args.fold < args.n_folds, f"fold must be in [0, {args.n_folds - 1}]"
        generate_fold_split(
            root_path=args.root,
            fold_idx=args.fold,
            n_folds=args.n_folds,
            val_size=args.val_size,
            seed=args.seed
        )




