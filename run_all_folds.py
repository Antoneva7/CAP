import subprocess
import sys
import time

MODEL = "Cascaded"
GPU = 0
EPOCHS = 40
BATCH_SIZE = 4
FOLDS = [0, 1, 2, 3, 4]

def run_fold(fold):
    cmd = [
        "python", "train.py",
        "--model", MODEL,
        "--gpu", str(GPU),
        "--train_epochs", str(EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--fold", str(fold),
    ]
    print(f"\n{'='*50}")
    print(f"开始训练 fold {fold}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*50}\n")

    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"[警告] fold {fold} 训练失败 (returncode={result.returncode})")
        return False

    print(f"fold {fold} 完成，耗时 {elapsed/60:.1f} 分钟")
    return True

if __name__ == "__main__":
    failed = []
    for fold in FOLDS:
        ok = run_fold(fold)
        if not ok:
            failed.append(fold)
            # 如果需要失败就停，取消下面注释
            # sys.exit(1)

    print("\n" + "="*50)
    if failed:
        print(f"以下 fold 训练失败: {failed}")
    else:
        print("所有 fold 训练完成！")