# 建議將此檔案儲存為: src/evaluate.py

import os
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式後端
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns

# 載入我們自定義的模組
from src import data_loader
from src import model

# --- 設定參數 (必須與 train.py 一致) ---

TEST_DATA_DIR = "./data/test"
MODEL_BASE_DIR = "./models"
DEFAULT_MODEL_DIR = "./models"  # 為了向後兼容，預設使用舊的路徑
FINAL_MODEL_NAME = "final_model.pth"
FINAL_SCALER_NAME = "final_scaler.joblib"

# 這些參數必須與訓練時一致
WINDOW_SIZE = 500
STEP_SIZE = 50
BATCH_SIZE = 32

# --- 視覺化函數 ---

def plot_confusion_matrix(cm, save_dir, title='Confusion Matrix'):
    """
    繪製混淆矩陣

    Args:
        cm: 混淆矩陣
        save_dir: 儲存目錄
        title: 圖表標題
    """
    plt.figure(figsize=(8, 6))

    # 使用 seaborn 的熱力圖
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['State1 (Normal)', 'State2 (Abnormal)'],
                yticklabels=['State1 (Normal)', 'State2 (Abnormal)'])

    plt.title(title, fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)

    # 儲存圖片
    save_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")

def plot_probability_distribution(probabilities, true_labels, save_dir):
    """
    繪製預測機率分佈

    Args:
        probabilities: 預測機率
        true_labels: 真實標籤
        save_dir: 儲存目錄
    """
    plt.figure(figsize=(12, 5))

    # 分離正常和異常樣本的機率
    normal_probs = probabilities[true_labels == 0]
    abnormal_probs = probabilities[true_labels == 1]

    # 左子圖：機率分佈直方圖
    plt.subplot(1, 2, 1)
    bins = np.linspace(0, 1, 50)
    plt.hist(normal_probs, bins=bins, alpha=0.7, label='State1 (Normal)', color='blue')
    plt.hist(abnormal_probs, bins=bins, alpha=0.7, label='State2 (Abnormal)', color='red')
    plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold (0.5)')
    plt.xlabel('Prediction Probability')
    plt.ylabel('Sample Count')
    plt.title('Prediction Probability Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 右子圖：箱線圖
    plt.subplot(1, 2, 2)
    data_to_plot = [normal_probs, abnormal_probs]
    box = plt.boxplot(data_to_plot, labels=['State1', 'State2'], patch_artist=True)
    box['boxes'][0].set_facecolor('blue')
    box['boxes'][1].set_facecolor('red')
    plt.axhline(y=0.5, color='black', linestyle='--', label='Decision Threshold (0.5)')
    plt.ylabel('Prediction Probability')
    plt.title('Prediction Probability Distribution (Box Plot)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # 儲存圖片
    save_path = os.path.join(save_dir, 'probability_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Probability distribution plot saved to: {save_path}")

def plot_threshold_analysis(test_data_list, test_labels_list, model_instance, scaler, device, save_dir):
    """
    進行閾值分析並繪製結果

    Args:
        test_data_list: 測試數據列表
        test_labels_list: 測試標籤列表
        model_instance: 模型實例
        scaler: 標準化器
        device: 計算設備
        save_dir: 儲存目錄
    """
    thresholds = np.arange(0.0, 1.01, 0.05)
    accuracies = []
    f1_scores = []

    print("進行閾值分析...")

    for threshold in thresholds:
        # 進行檔案級別預測
        file_predictions = []

        for i, (data_array, true_label) in enumerate(zip(test_data_list, test_labels_list)):
            # 對單一檔案進行預測
            scaled_data = scaler.transform(data_array)

            # 創建滑動窗口
            X_windows, _ = data_loader.create_windows([scaled_data], [0], WINDOW_SIZE, STEP_SIZE)

            if len(X_windows) == 0:
                file_predictions.append(0)  # 預設為正常
                continue

            # 轉換為 PyTorch 張量
            X_tensor = torch.tensor(X_windows.transpose(0, 2, 1), dtype=torch.float32).to(device)

            with torch.no_grad():
                outputs = model_instance(X_tensor)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            # 使用投票法決定檔案級別的預測
            abnormal_ratio = np.mean(probs > 0.5)
            file_prediction = 1 if abnormal_ratio > threshold else 0
            file_predictions.append(file_prediction)

        # 計算指標
        accuracy = accuracy_score(test_labels_list, file_predictions)
        f1 = f1_score(test_labels_list, file_predictions, zero_division=0)

        accuracies.append(accuracy)
        f1_scores.append(f1)

    # 繪製閾值分析圖
    plt.figure(figsize=(12, 5))

    # 左子圖：準確率和 F1 分數
    plt.subplot(1, 2, 1)
    plt.plot(thresholds, accuracies, 'b-', label='Accuracy', linewidth=2, marker='o', markersize=4)
    plt.plot(thresholds, f1_scores, 'r-', label='F1 Score', linewidth=2, marker='s', markersize=4)
    plt.xlabel('Threshold')
    plt.ylabel('Performance Metrics')
    plt.title('Threshold Analysis - Performance Metrics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)

    # 找出最佳閾值
    best_f1_idx = np.argmax(f1_scores)
    best_acc_idx = np.argmax(accuracies)

    plt.axvline(x=thresholds[best_f1_idx], color='red', linestyle='--', alpha=0.7,
                label=f'Best F1 Threshold: {thresholds[best_f1_idx]:.2f}')
    plt.axvline(x=thresholds[best_acc_idx], color='blue', linestyle='--', alpha=0.7,
                label=f'Best Accuracy Threshold: {thresholds[best_acc_idx]:.2f}')

    # 右子圖：指標差異
    plt.subplot(1, 2, 2)
    diff = np.array(accuracies) - np.array(f1_scores)
    plt.plot(thresholds, diff, 'g-', label='Accuracy - F1 Score', linewidth=2, marker='^', markersize=4)
    plt.xlabel('Threshold')
    plt.ylabel('Metrics Difference')
    plt.title('Threshold Analysis - Metrics Difference')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.xlim(0, 1)

    plt.tight_layout()

    # 儲存圖片
    save_path = os.path.join(save_dir, 'threshold_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Threshold analysis plot saved to: {save_path}")

    # 返回最佳閾值資訊
    return {
        'thresholds': thresholds,
        'accuracies': accuracies,
        'f1_scores': f1_scores,
        'best_f1_threshold': thresholds[best_f1_idx],
        'best_acc_threshold': thresholds[best_acc_idx]
    }

def evaluate_on_test_set(threshold=None, model_dir=None):
    """
    在測試集上評估最終模型的性能

    Args:
        threshold (float): 檔案級別投票閾值，預設為 0.1
        model_dir (str): 模型目錄路徑，如果為 None 則使用預設目錄
    """

    # 如果沒有指定模型目錄，使用預設目錄
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR

    print(f"--- 開始測試集評估 (初始檔案級閾值: {threshold}) ---")
    print(f"--- 使用模型目錄: {model_dir} ---")

    # 創建對應的結果目錄
    import re
    if re.search(r'training_\d{8}_\d{6}', model_dir):
        # 如果是時間戳記格式的目錄，創建對應的 res 目錄
        timestamp = re.search(r'training_(\d{8}_\d{6})', model_dir).group(1)
        res_dir = os.path.join('./res', f"training_{timestamp}")
        os.makedirs(res_dir, exist_ok=True)
        print(f"--- 評估圖表將儲存至: {res_dir} ---")
    else:
        # 舊格式或其他格式，在模型目錄下創建 plots 子目錄
        res_dir = os.path.join(model_dir, "plots")
        os.makedirs(res_dir, exist_ok=True)
        print(f"--- 評估圖表將儲存至: {res_dir} ---")

    # 嘗試從模型目錄載入 golden threshold 設定
    threshold_info_path = os.path.join(model_dir, "threshold_info.joblib")
    golden_file_threshold = None
    golden_window_threshold = 0.5
    if os.path.exists(threshold_info_path):
        try:
            threshold_info = joblib.load(threshold_info_path)
            golden_file_threshold = float(threshold_info.get("file_level_threshold", 0.3))
            golden_window_threshold = float(threshold_info.get("window_level_threshold", 0.5))
            print(f"偵測到 golden threshold 設定檔: {threshold_info_path}")
            print(f"  golden file_level_threshold = {golden_file_threshold:.4f}")
            print(f"  golden window_level_threshold = {golden_window_threshold:.4f}")

            # 規則：若 threshold 為 None (代表使用者沒有在 CLI 指定)，則自動改用 golden threshold
            if threshold is None:
                threshold = golden_file_threshold
                print(f"  檔案級閾值自動改用 golden threshold: {threshold:.4f}")
            else:
                print(f"  使用者自訂 threshold={threshold:.4f}，保留此設定")
        except Exception as e:
            print(f"警告: 讀取 threshold_info.joblib 失敗: {e}")

    # 若仍未設定 threshold，使用預設 0.3
    if threshold is None:
        threshold = 0.3
        print(f"  未找到 golden threshold，使用預設檔案級閾值: {threshold:.4f}")

    # 設置裝置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    # --- 1. 載入測試資料 ---
    # 先嘗試從模型目錄載入對應的測試資料
    test_data_path = os.path.join(model_dir, "test_data.joblib")
    if os.path.exists(test_data_path):
        print(f"從模型目錄載入測試資料: {test_data_path}")
        test_data_dict = joblib.load(test_data_path)
        test_data_list = test_data_dict['test_data_list']
        test_labels_list = test_data_dict['test_labels_list']

        # 檢查是否有檔案名稱和種子碼資訊
        if 'test_file_names' in test_data_dict and 'random_state' in test_data_dict:
            test_file_names = test_data_dict['test_file_names']
            random_state = test_data_dict['random_state']

            print(f"\n=== 測試集資訊 (種子碼: {random_state}) ===")
            test_state1_files = [name for name, label in zip(test_file_names, test_labels_list) if label == 0]
            test_state2_files = [name for name, label in zip(test_file_names, test_labels_list) if label == 1]

            print(f"測試檔案:")
            print(f"  狀態1: {sorted(test_state1_files)}")
            print(f"  狀態2: {sorted(test_state2_files)}")
            print(f"總計: {len(test_data_list)} 個檔案 (狀態1: {len(test_state1_files)}, 狀態2: {len(test_state2_files)})")
        else:
            print(f"載入了 {len(test_data_list)} 個測試檔案")
            print(f"  - 狀態1 (正常): {sum(1 for label in test_labels_list if label == 0)} 個")
            print(f"  - 狀態2 (異常): {sum(1 for label in test_labels_list if label == 1)} 個")
            print(f"  (舊格式資料，無檔案名稱和種子碼資訊)")
    else:
        print(f"未找到對應的測試資料，使用預設測試目錄: {TEST_DATA_DIR}")
        test_data_list, test_labels_list = data_loader.load_all_data_from_dir(TEST_DATA_DIR)
        print(f"使用固定測試集: {len(test_data_list)} 個檔案")

    if len(test_data_list) == 0:
        print("錯誤: 找不到測試資料")
        return

    # --- 2. 載入 Scaler ---
    scaler_path = os.path.join(model_dir, FINAL_SCALER_NAME)
    if not os.path.exists(scaler_path):
        print(f"錯誤: 找不到 Scaler: {scaler_path}")
        return
    scaler = data_loader.load_scaler(scaler_path)

    # --- 3. 載入模型 ---
    NUM_FEATURES = scaler.n_features_in_
    model_instance = model.CNC_1D_CNN(
        num_features=NUM_FEATURES, window_size=WINDOW_SIZE
    ).to(device)

    model_path = os.path.join(model_dir, FINAL_MODEL_NAME)
    if not os.path.exists(model_path):
        print(f"錯誤: 找不到模型: {model_path}")
        return

    model_instance.load_state_dict(torch.load(model_path, map_location=device))
    model_instance.eval()
    print("模型和 Scaler 載入成功")

    # --- 4. 資料前處理 ---
    # 套用標準化
    scaled_test_list = data_loader.apply_scaling(test_data_list, scaler)

    # 創建滑動窗口
    X_test, y_test = data_loader.create_windows(
        scaled_test_list, test_labels_list, WINDOW_SIZE, STEP_SIZE
    )

    print(f"測試樣本數 (窗口): {len(X_test)}")

    # --- 5. 建立 DataLoader ---
    test_loader = data_loader.get_data_loader(X_test, y_test, BATCH_SIZE, shuffle=False)

    # --- 6. 模型預測 ---
    all_preds = []
    all_labels = []
    all_probs = []

    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # 前向傳播
            outputs = model_instance(X_batch)

            # 計算損失
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

            # 轉換為機率和預測
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            # 收集結果
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    # 攤平結果
    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    all_probs = np.concatenate(all_probs).flatten()

    # --- 7. 計算指標 ---
    test_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    # --- 8. 顯示結果 ---
    print("\n--- 測試集評估結果 ---")
    print(f"測試損失: {test_loss:.4f}")
    print(f"準確率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1 分數: {f1:.4f}")

    print(f"\n--- 詳細分類報告 ---")
    target_names = ['狀態1 (正常)', '狀態2 (異常)']
    print(classification_report(all_labels, all_preds, target_names=target_names))

    print(f"\n--- 混淆矩陣 ---")
    cm = confusion_matrix(all_labels, all_preds)
    print("預測\\實際  狀態1  狀態2")
    print(f"狀態1      {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"狀態2      {cm[1,0]:4d}   {cm[1,1]:4d}")

    # --- 9. 生成視覺化圖表 ---
    print(f"\n--- 生成視覺化圖表 ---")

    # 繪製混淆矩陣
    plot_confusion_matrix(cm, res_dir)

    # 繪製機率分佈
    plot_probability_distribution(all_probs, all_labels, res_dir)

    # 進行閾值分析 (此處仍固定使用 window-level threshold=0.5，file-level threshold 掃描一整個 grid)
    threshold_results = plot_threshold_analysis(test_data_list, test_labels_list, model_instance, scaler, device, res_dir)

    # --- 10. 檔案級別的評估 ---
    print(f"\n--- 檔案級別評估 ---")
    evaluate_file_level_accuracy(test_data_list, test_labels_list, model_instance, scaler, device, threshold)

    print(f"\n--- 閾值分析結果摘要 ---")
    print(f"最佳 F1 分數閾值: {threshold_results['best_f1_threshold']:.2f} (F1: {max(threshold_results['f1_scores']):.4f})")
    print(f"最佳準確率閾值: {threshold_results['best_acc_threshold']:.2f} (準確率: {max(threshold_results['accuracies']):.4f})")

    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'test_loss': test_loss,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'probabilities': all_probs,
        'true_labels': all_labels,
        'threshold_analysis': threshold_results,
        'res_dir': res_dir
    }

def evaluate_file_level_accuracy(test_data_list, test_labels_list, model_instance, scaler, device, threshold=0.3):
    """
    在檔案級別評估模型性能（而不是窗口級別）

    Args:
        test_data_list: 測試數據列表
        test_labels_list: 測試標籤列表
        model_instance: 模型實例
        scaler: 標準化器
        device: 計算設備
        threshold: 檔案級別投票閾值
    """
    file_predictions = []
    file_true_labels = test_labels_list

    for i, (data_array, true_label) in enumerate(zip(test_data_list, test_labels_list)):
        # 對單一檔案進行預測
        scaled_data = scaler.transform(data_array)

        # 創建滑動窗口
        X_windows, _ = data_loader.create_windows([scaled_data], [0], WINDOW_SIZE, STEP_SIZE)

        if len(X_windows) == 0:
            print(f"警告: 檔案 {i+1} 太短，無法創建窗口")
            file_predictions.append(0)  # 預設為正常
            continue

        # 轉換為 PyTorch 張量
        X_tensor = torch.tensor(X_windows.transpose(0, 2, 1), dtype=torch.float32).to(device)

        with torch.no_grad():
            outputs = model_instance(X_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

        # 使用投票法決定檔案級別的預測
        abnormal_ratio = np.mean(probs > 0.5)
        file_prediction = 1 if abnormal_ratio > threshold else 0  # 使用傳入的閾值
        file_predictions.append(file_prediction)

    # 計算檔案級別的指標
    file_accuracy = accuracy_score(file_true_labels, file_predictions)
    file_f1 = f1_score(file_true_labels, file_predictions)

    print(f"檔案級別準確率: {file_accuracy:.4f} ({file_accuracy*100:.2f}%)")
    print(f"檔案級別 F1 分數: {file_f1:.4f}")

    # 顯示每個檔案的結果
    state1_files = [i for i, label in enumerate(file_true_labels) if label == 0]
    state2_files = [i for i, label in enumerate(file_true_labels) if label == 1]

    print(f"\n狀態1 檔案 ({len(state1_files)} 個):")
    for i in state1_files:
        result = "正確" if file_predictions[i] == 0 else "錯誤"
        print(f"  檔案 {i+1}: 預測={file_predictions[i]} (實際=0) - {result}")

    print(f"\n狀態2 檔案 ({len(state2_files)} 個):")
    for i in state2_files:
        result = "正確" if file_predictions[i] == 1 else "錯誤"
        print(f"  檔案 {i+1}: 預測={file_predictions[i]} (實際=1) - {result}")


if __name__ == "__main__":
    evaluate_on_test_set()
