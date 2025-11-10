import os
import shutil
import random

# 設定隨機種子碼
SEED = 42
random.seed(SEED)

# 原始資料夾路徑
RAW_DATA_DIR = "raw_data"

# 訓練與測試資料夾路徑
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"

# 分割比例
TRAIN_RATIO = 0.8

# 確保目標資料夾存在
os.makedirs(os.path.join(TRAIN_DIR, "state1"), exist_ok=True)
os.makedirs(os.path.join(TRAIN_DIR, "state2"), exist_ok=True)
os.makedirs(os.path.join(TEST_DIR, "state1"), exist_ok=True)
os.makedirs(os.path.join(TEST_DIR, "state2"), exist_ok=True)

# 分割資料的函數
def split_data(state_dir, train_dir, test_dir):
    files = [f for f in os.listdir(state_dir) if f.endswith(".csv")]
    random.shuffle(files)

    train_count = int(len(files) * TRAIN_RATIO)
    train_files = files[:train_count]
    test_files = files[train_count:]

    for f in train_files:
        shutil.copy(os.path.join(state_dir, f), os.path.join(train_dir, f))

    for f in test_files:
        shutil.copy(os.path.join(state_dir, f), os.path.join(test_dir, f))

# 分割狀態1的資料
split_data(
    os.path.join(RAW_DATA_DIR, "state1"),
    os.path.join(TRAIN_DIR, "state1"),
    os.path.join(TEST_DIR, "state1"),
)

# 分割狀態2的資料
split_data(
    os.path.join(RAW_DATA_DIR, "state2"),
    os.path.join(TRAIN_DIR, "state2"),
    os.path.join(TEST_DIR, "state2"),
)

print("資料分割完成！")