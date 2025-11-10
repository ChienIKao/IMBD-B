# 建議將此檔案儲存為: src/train.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式後端
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']  # 使用英文字體
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime

# 載入我們自定義的模組
from src import data_loader
from src import model

# --- 1. 集中管理設定參數 (未來可以移到 src/config.py) ---

# 資料路徑
RAW_DATA_DIR = './raw_data'  # 原始資料路徑
MODEL_BASE_DIR = './models' # 基礎模型目錄
RES_BASE_DIR = './res' # 結果圖表目錄

# 資料分割參數
TRAIN_RATIO = 0.7         # 訓練資料比例 (80%)
TEST_RATIO = 0.3          # 測試資料比例 (20%)

# K-Fold 參數
N_SPLITS = 5          # 5 折交叉驗證
RANDOM_STATE = 1018   # 隨機種子碼
# [cite_start]競賽題目要求 F1 score 為決勝標準 [cite: 97]，我們可以用它來儲存最佳模型

# 資料處理參數
WINDOW_SIZE = 500   # 滑動窗口大小 (可調整)
STEP_SIZE = 50      # 滑動步長 (可調整)

# 訓練參數
EPOCHS = 30         # 訓練回合數 (可調整)
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# --- 2. 訓練與驗證的輔助函數 ---

def train_epoch(model, data_loader, criterion, optimizer, device):
    """
    執行一個 Epoch 的訓練
    """
    model.train() # 設置為訓練模式
    total_loss = 0

    for X_batch, y_batch in data_loader:
        # 將資料移動到 GPU (如果可用)
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # 梯度歸零
        optimizer.zero_grad()

        # 前向傳播
        outputs = model(X_batch)

        # 計算損失
        loss = criterion(outputs, y_batch)

        # 反向傳播
        loss.backward()

        # 更新權重
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)

def validate_epoch(model, data_loader, criterion, device):
    """
    執行一個 Epoch 的驗證
    """
    model.eval() # 設置為評估模式
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad(): # 評估時不需要計算梯度
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # 前向傳播
            outputs = model(X_batch)

            # 計算損失
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

            # --- 計算評分指標 ---
            # 1. 將模型的原始輸出 (logits) 轉為機率 (0~1)
            probs = torch.sigmoid(outputs)
            # 2. 根據 0.5 閾值轉為預測 (0或1)
            preds = (probs > 0.5).float()

            # 收集所有預測和標籤 (移回 CPU)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    # 將所有批次的結果攤平
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # [cite_start]計算準確度 [cite: 95] [cite_start]和 F1 Score [cite: 97]
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return total_loss / len(data_loader), acc, f1

# --- 3. 繪圖功能函數 ---

def plot_training_curves(fold_histories, save_dir):
    """
    繪製所有折次的訓練曲線

    Args:
        fold_histories: list of dict，每個 dict 包含一折的訓練歷史
        save_dir: 圖表儲存目錄
    """

    # 創建子圖
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('K-Fold Cross-Validation Training History', fontsize=16)

    # 顏色列表
    colors = ['blue', 'red', 'green', 'orange', 'purple']

    # 繪製每一折的曲線
    for fold_idx, history in enumerate(fold_histories):
        color = colors[fold_idx % len(colors)]
        label = f'Fold {fold_idx + 1}'

        epochs = list(range(1, len(history['train_loss']) + 1))

        # 訓練損失
        axes[0, 0].plot(epochs, history['train_loss'], color=color, alpha=0.7, label=label)

        # 驗證損失
        axes[0, 1].plot(epochs, history['val_loss'], color=color, alpha=0.7, label=label)

        # 驗證準確率
        axes[1, 0].plot(epochs, history['val_acc'], color=color, alpha=0.7, label=label)

        # 驗證 F1 分數
        axes[1, 1].plot(epochs, history['val_f1'], color=color, alpha=0.7, label=label)

    # 設置子圖標題和標籤
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title('Validation Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title('Validation F1 Score')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 調整佈局
    plt.tight_layout()

    # 儲存圖片
    save_path = os.path.join(save_dir, 'kfold_training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"K-Fold training curves saved to: {save_path}")

def plot_final_training_curve(final_history, save_dir):
    """
    繪製最終模型的訓練曲線

    Args:
        final_history: dict，包含最終訓練的歷史
        save_dir: 圖表儲存目錄
    """

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    epochs = list(range(1, len(final_history['train_loss']) + 1))

    ax.plot(epochs, final_history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax.set_title('Final Model Training Loss', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 儲存圖片
    save_path = os.path.join(save_dir, 'final_training_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Final model training curve saved to: {save_path}")

def plot_cv_summary(cv_f1_scores, save_dir):
    """
    繪製交叉驗證結果摘要

    Args:
        cv_f1_scores: list，每一折的 F1 分數
        save_dir: 圖表儲存目錄
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 折次 F1 分數條形圖
    folds = [f'Fold {i+1}' for i in range(len(cv_f1_scores))]
    ax1.bar(folds, cv_f1_scores, color='skyblue', alpha=0.7)
    ax1.set_title('F1 Score by Fold', fontsize=14)
    ax1.set_ylabel('F1 Score')
    ax1.grid(True, alpha=0.3)

    # 在每個條形圖上顯示數值
    for i, score in enumerate(cv_f1_scores):
        ax1.text(i, score + 0.01, f'{score:.4f}', ha='center', va='bottom')

    # F1 分數箱線圖
    ax2.boxplot([cv_f1_scores], labels=['All Folds'])
    ax2.set_title('F1 Score Distribution', fontsize=14)
    ax2.set_ylabel('F1 Score')
    ax2.grid(True, alpha=0.3)

    # 添加統計資訊
    mean_f1 = np.mean(cv_f1_scores)
    std_f1 = np.std(cv_f1_scores)
    ax2.text(1.1, mean_f1, f'Mean: {mean_f1:.4f}\nStd: {std_f1:.4f}',
             transform=ax2.transData, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # 儲存圖片
    save_path = os.path.join(save_dir, 'cv_summary.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Cross-validation summary saved to: {save_path}")

# --- 4. 主要訓練執行函數 ---

def run_training():

    # 創建時間戳記的訓練目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    MODEL_DIR = os.path.join(MODEL_BASE_DIR, f"training_{timestamp}")
    RES_DIR = os.path.join(RES_BASE_DIR, f"training_{timestamp}")

    # 確保模型儲存目錄存在
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)
    print(f"--- 訓練模型將儲存至: {MODEL_DIR} ---")
    print(f"--- 訓練圖表將儲存至: {RES_DIR} ---")

    # 設置裝置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 將使用 {device} 進行訓練 ---")
    print(f"--- 隨機種子碼: {RANDOM_STATE} ---")

    # --- A. 載入所有原始資料並重新分割 ---
    print("--- 載入原始資料並重新分割 ---")

    # 載入所有原始資料並取得檔案名稱資訊
    all_data_list, all_labels_list, file_names = data_loader.load_all_data_from_raw_with_names(RAW_DATA_DIR)

    print(f"載入了 {len(all_data_list)} 個檔案")
    print(f"  - 狀態1 (正常): {sum(1 for label in all_labels_list if label == 0)} 個檔案")
    print(f"  - 狀態2 (異常): {sum(1 for label in all_labels_list if label == 1)} 個檔案")

    # 使用 train_test_split 重新分割資料
    train_data_list, test_data_list, train_labels_list, test_labels_list, train_indices, test_indices = train_test_split(
        all_data_list,
        all_labels_list,
        list(range(len(file_names))),  # 加入索引以追蹤檔案名稱
        test_size=TEST_RATIO,
        random_state=RANDOM_STATE,
        stratify=all_labels_list  # 保持類別比例
    )

    # 取得對應的檔案名稱
    train_file_names = [file_names[i] for i in train_indices]
    test_file_names = [file_names[i] for i in test_indices]

    print(f"\n重新分割後:")
    print(f"  訓練資料: {len(train_data_list)} 個檔案")
    print(f"    - 狀態1: {sum(1 for label in train_labels_list if label == 0)} 個")
    print(f"    - 狀態2: {sum(1 for label in train_labels_list if label == 1)} 個")
    print(f"  測試資料: {len(test_data_list)} 個檔案")
    print(f"    - 狀態1: {sum(1 for label in test_labels_list if label == 0)} 個")
    print(f"    - 狀態2: {sum(1 for label in test_labels_list if label == 1)} 個")

    # 顯示詳細的檔案分割資訊
    print(f"\n=== 詳細分割資訊 (種子碼: {RANDOM_STATE}) ===")
    train_state1_files = [name for name, label in zip(train_file_names, train_labels_list) if label == 0]
    train_state2_files = [name for name, label in zip(train_file_names, train_labels_list) if label == 1]
    test_state1_files = [name for name, label in zip(test_file_names, test_labels_list) if label == 0]
    test_state2_files = [name for name, label in zip(test_file_names, test_labels_list) if label == 1]

    print(f"訓練集檔案:")
    print(f"  狀態1: {sorted(train_state1_files)}")
    print(f"  狀態2: {sorted(train_state2_files)}")
    print(f"測試集檔案:")
    print(f"  狀態1: {sorted(test_state1_files)}")
    print(f"  狀態2: {sorted(test_state2_files)}")

    # 將測試資料儲存到模型目錄中，供後續評估使用
    test_data_path = os.path.join(MODEL_DIR, "test_data.joblib")
    joblib.dump({
        'test_data_list': test_data_list,
        'test_labels_list': test_labels_list,
        'test_file_names': test_file_names,  # 同時儲存檔案名稱
        'random_state': RANDOM_STATE  # 儲存種子碼
    }, test_data_path)
    print(f"\n測試資料已儲存至: {test_data_path}")
    print(f"(包含檔案名稱和種子碼資訊)")

    # 轉為 NumPy 陣列，以利 K-Fold 切分
    # 注意：K-Fold 是在「檔案」層級上操作的
    all_data_np = np.array(train_data_list, dtype=object)  # 只使用訓練資料
    all_labels_np = np.array(train_labels_list)

    # 從資料中動態獲取特徵數量
    NUM_FEATURES = all_data_np[0].shape[1]

    # --- B. K-Fold 交叉驗證迴圈 ---

    kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_f1_scores = []
    fold_histories = []  # 記錄每一折的訓練歷史

    print(f"\n--- 開始 {N_SPLITS} 折交叉驗證 ---")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(all_data_np, all_labels_np)):
        print(f"\n--- 第 {fold+1} / {N_SPLITS} 折 ---")

        # 1. 切分「檔案列表」
        train_files_list = all_data_np[train_idx]
        train_labels_list = all_labels_np[train_idx]
        val_files_list = all_data_np[val_idx]
        val_labels_list = all_labels_np[val_idx]

        # 2. Fit Scaler (關鍵：只在「訓練集檔案」上 Fit)
        scaler_path = os.path.join(MODEL_DIR, f"scaler_fold_{fold+1}.joblib")
        scaler = data_loader.fit_and_save_scaler(train_files_list, scaler_path)

        # 3. Apply Scaler
        scaled_train_list = data_loader.apply_scaling(train_files_list, scaler)
        scaled_val_list = data_loader.apply_scaling(val_files_list, scaler)

        # 4. Sliding Window
        X_train, y_train = data_loader.create_windows(scaled_train_list, train_labels_list, WINDOW_SIZE, STEP_SIZE)
        X_val, y_val = data_loader.create_windows(scaled_val_list, val_labels_list, WINDOW_SIZE, STEP_SIZE)

        print(f"  訓練樣本數 (窗口): {len(X_train)}, 驗證樣本數 (窗口): {len(X_val)}")

        # 5. 創建 DataLoaders
        train_loader = data_loader.get_data_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
        val_loader = data_loader.get_data_loader(X_val, y_val, BATCH_SIZE, shuffle=False)

        # 6. 準備模型 (每折都重新初始化一個新模型)
        model_instance = model.CNC_1D_CNN(num_features=NUM_FEATURES, window_size=WINDOW_SIZE).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model_instance.parameters(), lr=LEARNING_RATE)

        best_val_f1 = -1

        # 記錄這一折的訓練歷史
        fold_history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }

        # 7. Epoch 迴圈
        for epoch in range(EPOCHS):
            train_loss = train_epoch(model_instance, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_f1 = validate_epoch(model_instance, val_loader, criterion, device)

            # 記錄歷史
            fold_history['train_loss'].append(train_loss)
            fold_history['val_loss'].append(val_loss)
            fold_history['val_acc'].append(val_acc)
            fold_history['val_f1'].append(val_f1)

            print(f"  Epoch {epoch+1}/{EPOCHS} -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

            # [cite_start]儲存 F1 [cite: 97] 最高的模型
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                model_save_path = os.path.join(MODEL_DIR, f"model_fold_{fold+1}_best.pth")
                torch.save(model_instance.state_dict(), model_save_path)

        cv_f1_scores.append(best_val_f1)
        fold_histories.append(fold_history)
        print(f"  第 {fold+1} 折 最佳 F1: {best_val_f1:.4f}")

    # --- C. 顯示 K-Fold 總結 ---
    print("\n--- K-Fold 交叉驗證總結 ---")
    print(f"所有 F1 分數: {cv_f1_scores}")
    print(f"平均 F1: {np.mean(cv_f1_scores):.4f} +/- {np.std(cv_f1_scores):.4f}")

    # 繪製 K-Fold 訓練曲線和摘要
    plot_training_curves(fold_histories, RES_DIR)
    plot_cv_summary(cv_f1_scores, RES_DIR)

    # --- D. 訓練最終模型 (使用所有資料) ---
    print("\n--- 開始訓練最終模型 (使用全部資料) ---")

    # 1. Fit Scaler (在「所有」資料上 Fit)
    final_scaler_path = os.path.join(MODEL_DIR, "final_scaler.joblib")
    final_scaler = data_loader.fit_and_save_scaler(all_data_np, final_scaler_path)

    # 2. Apply Scaler
    scaled_all_list = data_loader.apply_scaling(all_data_np, final_scaler)

    # 3. Sliding Window
    X_all, y_all = data_loader.create_windows(scaled_all_list, all_labels_np, WINDOW_SIZE, STEP_SIZE)
    print(f"  最終訓練樣本數 (窗口): {len(X_all)}")

    # 4. 創建 DataLoader
    final_train_loader = data_loader.get_data_loader(X_all, y_all, BATCH_SIZE, shuffle=True)

    # 5. 準備模型
    final_model = model.CNC_1D_CNN(num_features=NUM_FEATURES, window_size=WINDOW_SIZE).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(final_model.parameters(), lr=LEARNING_RATE)

    # 記錄最終模型的訓練歷史
    final_history = {
        'train_loss': []
    }

    # 6. Epoch 迴圈 (只訓練，不驗證)
    for epoch in range(EPOCHS):
        train_loss = train_epoch(final_model, final_train_loader, criterion, optimizer, device)
        final_history['train_loss'].append(train_loss)
        print(f"  Epoch {epoch+1}/{EPOCHS} -> Train Loss: {train_loss:.4f}")

    # 7. 儲存最終模型
    final_model_path = os.path.join(MODEL_DIR, "final_model.pth")
    torch.save(final_model.state_dict(), final_model_path)

    # 繪製最終模型訓練曲線
    plot_final_training_curve(final_history, RES_DIR)

    print(f"\n--- 訓練完畢 ---")
    print(f"訓練模型資料夾: {MODEL_DIR}")
    print(f"訓練圖表資料夾: {RES_DIR}")
    print(f"最終模型儲存至: {final_model_path}")
    print(f"最終 Scaler 儲存至: {final_scaler_path}")
    print(f"K-Fold 模型儲存在同一資料夾中")

    return MODEL_DIR  # 回傳模型目錄路徑，供其他函數使用

if __name__ == "__main__":
    # 這個腳本可以透過 python -m src.train 來執行
    # 或者您在 main.py 中呼叫 run_training()
    run_training()
