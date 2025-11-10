# 建議將此檔案儲存為: src/data_loader.py

import os
import glob
import pandas as pd
import numpy as np
import joblib
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# --- 1. 原始資料讀取 ---


def load_single_csv(file_path):
    """
    載入單一個競賽用的 CSV 檔案。
    根據題目說明 [cite: 32]，第一列為欄位名，第二列為單位，第三列才
    是數據。
    同時，我們只取 C 欄位以後的特徵資料 [cite: 31]。
    """
    # header=0 表示使用第一列 (索引為0) 作為欄位名稱
    # skiprows=[1] 表示跳過第二列 (索引為1，即「單位」那一行)
    df = pd.read_csv(file_path, header=0, skiprows=[1])

    # 根據題目說明 [cite: 31]，A欄位為索引, B欄位為時間
    # 我們只取 C 欄位 (索引為2) 之後的所有感測器數據
    data_values = df.iloc[:, 2:].values
    return data_values


def load_all_data_from_raw(raw_dir):
    """
    從 raw_data 資料夾載入所有狀態1和狀態2的檔案。

    Returns:
        all_data_list (list): 包含所有檔案數據(NumPy陣列)的列表
        all_labels_list (list): 對應的標籤 (0 for 狀態1, 1 for 狀態2)
    """
    all_data_list = []
    all_labels_list = []

    # 載入狀態1 (正常) 檔案 - 支援新舊資料夾名稱
    status1_patterns = [
        os.path.join(raw_dir, "狀態1", "*.csv"),  # 舊格式
        os.path.join(raw_dir, "state1", "*.csv")  # 新格式
    ]

    for pattern in status1_patterns:
        status1_files = glob.glob(pattern)
        for f in status1_files:
            all_data_list.append(load_single_csv(f))
            all_labels_list.append(0)  # 0 代表正常

    # 載入狀態2 (異常) 檔案 - 支援新舊資料夾名稱
    status2_patterns = [
        os.path.join(raw_dir, "狀態2", "*.csv"),  # 舊格式
        os.path.join(raw_dir, "state2", "*.csv")  # 新格式
    ]

    for pattern in status2_patterns:
        status2_files = glob.glob(pattern)
        for f in status2_files:
            all_data_list.append(load_single_csv(f))
            all_labels_list.append(1)  # 1 代表異常

    print(f"總共載入 {len(all_data_list)} 個檔案。")
    return all_data_list, all_labels_list


def load_all_data_from_raw_with_names(raw_dir):
    """
    從 raw_data 資料夾載入所有狀態1和狀態2的檔案，同時回傳檔案名稱。

    Returns:
        all_data_list (list): 包含所有檔案數據(NumPy陣列)的列表
        all_labels_list (list): 對應的標籤 (0 for 狀態1, 1 for 狀態2)
        file_names (list): 對應的檔案名稱列表
    """
    all_data_list = []
    all_labels_list = []
    file_names = []

    # 載入狀態1 (正常) 檔案 - 支援新舊資料夾名稱
    status1_patterns = [
        os.path.join(raw_dir, "狀態1", "*.csv"),  # 舊格式
        os.path.join(raw_dir, "state1", "*.csv")  # 新格式
    ]

    for pattern in status1_patterns:
        status1_files = glob.glob(pattern)
        for f in sorted(status1_files):  # 排序確保順序一致
            all_data_list.append(load_single_csv(f))
            all_labels_list.append(0)  # 0 代表正常
            file_names.append(os.path.basename(f))

    # 載入狀態2 (異常) 檔案 - 支援新舊資料夾名稱
    status2_patterns = [
        os.path.join(raw_dir, "狀態2", "*.csv"),  # 舊格式
        os.path.join(raw_dir, "state2", "*.csv")  # 新格式
    ]

    for pattern in status2_patterns:
        status2_files = glob.glob(pattern)
        for f in sorted(status2_files):  # 排序確保順序一致
            all_data_list.append(load_single_csv(f))
            all_labels_list.append(1)  # 1 代表異常
            file_names.append(os.path.basename(f))

    print(f"總共載入 {len(all_data_list)} 個檔案。")
    return all_data_list, all_labels_list, file_names


def load_all_data_from_dir(data_dir):
    """
    從指定資料夾載入所有state1和state2的檔案。

    Args:
        data_dir (str): 資料夾路徑，應包含 state1 和 state2 子資料夾

    Returns:
        all_data_list (list): 包含所有檔案數據(NumPy陣列)的列表
        all_labels_list (list): 對應的標籤 (0 for state1, 1 for state2)
    """
    all_data_list = []
    all_labels_list = []

    # 載入狀態1 (正常) 檔案
    status1_files = glob.glob(os.path.join(data_dir, "state1", "*.csv"))
    for f in status1_files:
        all_data_list.append(load_single_csv(f))
        all_labels_list.append(0)  # 0 代表正常

    # 載入狀態2 (異常) 檔案
    status2_files = glob.glob(os.path.join(data_dir, "state2", "*.csv"))
    for f in status2_files:
        all_data_list.append(load_single_csv(f))
        all_labels_list.append(1)  # 1 代表異常

    print(f"從 {data_dir} 載入 {len(all_data_list)} 個檔案。")
    print(f"  - 狀態1 (正常): {len(status1_files)} 檔案")
    print(f"  - 狀態2 (異常): {len(status2_files)} 檔案")
    return all_data_list, all_labels_list


# --- 2. 標準化 (Scaling) ---


def fit_and_save_scaler(data_list_for_fitting, save_path):
    """
    使用「訓練集」的資料來擬合 StandardScaler，並儲存。

    Args:
        data_list_for_fitting (list): 用於擬合的數據列表 (numpy array)
        save_path (str): scaler 儲存路徑 (e.g., "models/scaler.joblib")
    """
    # 為了擬合 scaler，我們需要將所有訓練檔案「攤平」成 (N, num_features)
    # N = 所有訓練檔案的總時間步長
    all_timesteps_data = np.concatenate(data_list_for_fitting, axis=0)

    scaler = StandardScaler()
    scaler.fit(all_timesteps_data)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(scaler, save_path)
    print(f"Scaler 已擬合並儲存至 {save_path}")
    return scaler


def load_scaler(load_path):
    """從路徑載入儲存的 scaler"""
    print(f"從 {load_path} 載入 scaler...")
    return joblib.load(load_path)


def apply_scaling(data_list, scaler):
    """將一個「已擬合」的 scaler 套用到所有檔案上"""
    scaled_data_list = []
    for data_array in data_list:
        scaled_array = scaler.transform(data_array)
        scaled_data_list.append(scaled_array)
    return scaled_data_list


# --- 3. 滑動窗口 (Sliding Window) ---


def create_windows(data_list, label_list, window_size, step_size):
    """
    對載入的數據列表執行滑動窗口操作。

    Args:
        data_list (list): 包含(已縮放)數據陣列的列表
        label_list (list): 對應的標籤列表 (0 或 1)
        window_size (int): 窗口大小 (e.g., 500)
        step_size (int): 窗口步長 (e.g., 50)

    Returns:
        X (np.array): (N_windows, window_size, num_features)
        y (np.array): (N_windows,)
    """
    X_windows = []
    y_windows = []

    for data_array, label in zip(data_list, label_list):
        n_timesteps = data_array.shape[0]

        for start in range(0, n_timesteps - window_size + 1, step_size):
            end = start + window_size
            window = data_array[start:end, :]

            X_windows.append(window)
            y_windows.append(label)

    X = np.array(X_windows)
    y = np.array(y_windows)

    return X, y


# --- 4. PyTorch Dataset & DataLoader ---


class CNCDataset(Dataset):
    """
    自定義的 PyTorch Dataset
    """

    def __init__(self, X_data, y_data):
        """
        Args:
            X_data (np.array): 形狀為 (N_windows, window_size, num_features)
            y_data (np.array): 形狀為 (N_windows,)
        """
        # PyTorch Conv1D 需要 (Batch, Channels, Length)
        # 我們的 X_data 是 (Batch, Length, Channels)
        # 所以在這裡做 permute (維度交換)
        self.X_data = torch.tensor(X_data.transpose(0, 2, 1), dtype=torch.float32)

        # 將 y 轉為 (Batch, 1) 以符合 nn.BCEWithLogitsLoss 的需求
        self.y_data = torch.tensor(y_data, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        return self.X_data[idx], self.y_data[idx]


def get_data_loader(X_data, y_data, batch_size, shuffle=True):
    """
    一個方便的函數，直接回傳 DataLoader
    """
    dataset = CNCDataset(X_data, y_data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


if __name__ == "__main__":
    # 這是個簡單的測試，確保 data_loader 可以運作
    # 假設您的 raw_data 放在 '../raw_data/'

    RAW_DATA_DIR = "../raw_data"
    WINDOW_SIZE = 500
    STEP_SIZE = 50
    BATCH_SIZE = 32

    # 1. 載入原始檔案
    data_list, label_list = load_all_data_from_raw(RAW_DATA_DIR)

    # (在真實流程中，您會在這裡將 data_list 切分為 train/val)
    # (這裡為了演示，我們先假設全部都是 train)

    # 2. 擬合 & 套用 Scaler
    scaler = fit_and_save_scaler(data_list, "../models/temp_scaler.joblib")
    scaled_data_list = apply_scaling(data_list, scaler)

    # 3. 創建滑動窗口
    X, y = create_windows(scaled_data_list, label_list, WINDOW_SIZE, STEP_SIZE)

    print(f"\n--- 滑動窗口結果 ---")
    print(f"原始檔案數量: {len(data_list)}")
    print(f"窗口化後樣本數 (X_shape): {X.shape}")
    print(f"窗口化後標籤數 (y_shape): {y.shape}")

    # 4. 創建 DataLoader
    test_loader = get_data_loader(X, y, BATCH_SIZE, shuffle=False)

    # 5. 檢查 DataLoader 的輸出
    X_batch, y_batch = next(iter(test_loader))

    print(f"\n--- DataLoader 檢查 ---")
    print(f"DataLoader Batch X Shape (Batch, Features, Length): {X_batch.shape}")
    print(f"DataLoader Batch y Shape (Batch, 1): {y_batch.shape}")

    print("\nsrc/data_loader.py 測試完畢。")
