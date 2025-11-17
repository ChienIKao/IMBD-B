# 分析每個測試檔案的詳細異常比例

import os
import torch
import numpy as np
import argparse
import joblib
from src import data_loader, model

def analyze_all_test_files(model_dir="./models"):
    """
    分析所有測試檔案的異常窗口比例

    Args:
        model_dir: 模型目錄路徑
    """
    print("=== 詳細分析所有測試檔案的異常比例 ===\n")
    print(f"使用模型目錄: {model_dir}")

    # 模型參數
    FINAL_MODEL_NAME = "final_model.pth"
    FINAL_SCALER_NAME = "final_scaler.joblib"
    WINDOW_SIZE = 500
    STEP_SIZE = 50

    # 嘗試載入由 K-Fold validation 計算出的 golden threshold
    threshold_info_path = os.path.join(model_dir, "threshold_info.joblib")
    golden_file_threshold = None
    golden_window_threshold = 0.5
    if os.path.exists(threshold_info_path):
        try:
            threshold_info = joblib.load(threshold_info_path)
            golden_file_threshold = float(threshold_info.get("file_level_threshold", 0.5))
            golden_window_threshold = float(threshold_info.get("window_level_threshold", 0.5))
            print(f"載入 golden threshold 於: {threshold_info_path}")
            print(f"  file_level_threshold = {golden_file_threshold:.4f}")
            print(f"  window_level_threshold = {golden_window_threshold:.4f}")
        except Exception as e:
            print(f"警告: 讀取 threshold_info.joblib 失敗: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 載入 scaler
    scaler_path = os.path.join(model_dir, FINAL_SCALER_NAME)
    if not os.path.exists(scaler_path):
        print(f"錯誤: 找不到 Scaler 文件: {scaler_path}")
        return

    scaler = data_loader.load_scaler(scaler_path)
    NUM_FEATURES = scaler.n_features_in_  # 從 scaler 獲取特徵數量

    # 載入模型
    model_instance = model.CNC_1D_CNN(
        num_features=NUM_FEATURES, window_size=WINDOW_SIZE
    ).to(device)

    model_path = os.path.join(model_dir, FINAL_MODEL_NAME)
    if not os.path.exists(model_path):
        print(f"錯誤: 找不到模型文件: {model_path}")
        return

    model_instance.load_state_dict(torch.load(model_path, map_location=device))
    model_instance.eval()

    print(f"成功載入模型和 Scaler")

    # 檢查是否有對應的測試資料
    test_data_path = os.path.join(model_dir, "test_data.joblib")
    if os.path.exists(test_data_path):
        print(f"使用模型目錄中的測試資料: {test_data_path}")
        test_data_dict = joblib.load(test_data_path)
        test_data_list = test_data_dict['test_data_list']
        test_labels_list = test_data_dict['test_labels_list']

        # 檢查是否有檔案名稱資訊
        if 'test_file_names' in test_data_dict:
            test_file_names = test_data_dict['test_file_names']
        else:
            test_file_names = [f"file_{i+1}" for i in range(len(test_data_list))]
    else:
        print("未找到對應的測試資料，使用預設測試目錄")
        test_data_list, test_labels_list = data_loader.load_all_data_from_dir("./data/test")
        test_file_names = [f"file_{i+1}" for i in range(len(test_data_list))]

    if len(test_data_list) == 0:
        print("錯誤: 找不到測試資料")
        return

    # 分析每個檔案
    results = []

    for i, (data_array, true_label) in enumerate(zip(test_data_list, test_labels_list)):
        file_name = test_file_names[i] if i < len(test_file_names) else f"file_{i+1}"

        # 處理單一檔案
        scaled_data = scaler.transform(data_array)
        X_windows, _ = data_loader.create_windows([scaled_data], [0], WINDOW_SIZE, STEP_SIZE)

        if len(X_windows) == 0:
            print(f"警告: 檔案 {file_name} 太短，無法創建窗口")
            continue

        X_tensor = torch.tensor(X_windows.transpose(0, 2, 1), dtype=torch.float32).to(device)

        with torch.no_grad():
            outputs = model_instance(X_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

        # 計算異常比例
        # window 層級使用 threshold_info 中的設定 (若無則為 0.5)
        abnormal_count = np.sum(probs > golden_window_threshold)
        total_windows = len(probs)
        abnormal_ratio = abnormal_count / total_windows

        # 記錄結果
        file_type = "State1 (Normal)" if true_label == 0 else "State2 (Abnormal)"
        results.append({
            'file_name': file_name,
            'file_index': i + 1,
            'file_type': file_type,
            'true_label': true_label,
            'total_windows': total_windows,
            'abnormal_windows': abnormal_count,
            'abnormal_ratio': abnormal_ratio
        })

        print(f"File {i+1} [{file_name}] - {file_type}:")
        print(f"  Total windows: {total_windows}")
        print(f"  Abnormal windows: {abnormal_count}")
        print(f"  Abnormal ratio: {abnormal_ratio:.3f} ({abnormal_ratio*100:.1f}%)")

        # 不同閾值的預測結果
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        predictions = []
        for threshold in thresholds:
            pred = 1 if abnormal_ratio > threshold else 0
            correct = "✓" if pred == true_label else "✗"
            predictions.append(f"{threshold}: {pred}{correct}")
        print(f"  Threshold predictions: {' | '.join(predictions)}")
        print()

    if not results:
        print("沒有可分析的檔案")
        return

    # 總結分析
    print("=== Threshold Effect Summary ===")

    # 如果有 golden threshold，優先顯示其效果
    if golden_file_threshold is not None:
        correct_count = 0
        for result in results:
            pred = 1 if result['abnormal_ratio'] > golden_file_threshold else 0
            if pred == result['true_label']:
                correct_count += 1
        accuracy = correct_count / len(results)
        print(f"使用 golden file-level threshold {golden_file_threshold:.3f}: Accuracy {accuracy:.3f} ({correct_count}/{len(results)})")

    # 仍然可以掃一組固定 threshold 作為參考
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    best_threshold = thresholds[0]
    best_accuracy = 0.0

    for threshold in thresholds:
        correct_count = 0
        for result in results:
            pred = 1 if result['abnormal_ratio'] > threshold else 0
            if pred == result['true_label']:
                correct_count += 1
        accuracy = correct_count / len(results)
        print(f"Threshold {threshold}: Accuracy {accuracy:.3f} ({correct_count}/{len(results)})")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    print(f"\nBest threshold (grid [0.1,0.3,0.5,0.7,0.9]): {best_threshold} with accuracy {best_accuracy:.3f}")

    # 詳細分析
    print(f"\n=== Detailed Analysis ===")
    state1_ratios = [r['abnormal_ratio'] for r in results if r['true_label'] == 0]
    state2_ratios = [r['abnormal_ratio'] for r in results if r['true_label'] == 1]

    if state1_ratios:
        print(f"Normal files (State1) abnormal ratios:")
        for i, ratio in enumerate(state1_ratios):
            print(f"  File {i+1}: {ratio:.3f} ({ratio*100:.1f}%)")
        print(f"  Max: {max(state1_ratios):.3f}")

    if state2_ratios:
        print(f"\nAbnormal files (State2) abnormal ratios:")
        for i, ratio in enumerate(state2_ratios):
            print(f"  File {len(state1_ratios)+i+1}: {ratio:.3f} ({ratio*100:.1f}%)")
        print(f"  Min: {min(state2_ratios):.3f}")

    if state1_ratios and state2_ratios:
        print(f"\n💡 Key insights:")
        print(f"- Normal files max abnormal ratio: {max(state1_ratios):.3f}")
        print(f"- Abnormal files min abnormal ratio: {min(state2_ratios):.3f}")
        print(f"- Optimal threshold should be between {max(state1_ratios):.3f} and {min(state2_ratios):.3f}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Analyze threshold effects on test files")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./models",
        help="Path to model directory (default: ./models)"
    )

    args = parser.parse_args()

    analyze_all_test_files(args.model_dir)

if __name__ == "__main__":
    main()
