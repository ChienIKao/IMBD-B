# 分析每個測試檔案的詳細異常比例

import os
import torch
import numpy as np
from src import data_loader, model

def analyze_all_test_files():
    """
    分析所有測試檔案的異常窗口比例
    """
    print("=== 詳細分析所有測試檔案的異常比例 ===\n")

    # 載入模型和 scaler
    MODEL_DIR = "./models"
    FINAL_MODEL_NAME = "final_model.pth"
    FINAL_SCALER_NAME = "final_scaler.joblib"
    WINDOW_SIZE = 500
    STEP_SIZE = 50
    NUM_FEATURES = 35

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 載入 scaler
    scaler_path = os.path.join(MODEL_DIR, FINAL_SCALER_NAME)
    scaler = data_loader.load_scaler(scaler_path)

    # 載入模型
    model_instance = model.CNC_1D_CNN(
        num_features=NUM_FEATURES, window_size=WINDOW_SIZE
    ).to(device)
    model_path = os.path.join(MODEL_DIR, FINAL_MODEL_NAME)
    model_instance.load_state_dict(torch.load(model_path, map_location=device))
    model_instance.eval()

    # 載入測試資料
    test_data_list, test_labels_list = data_loader.load_all_data_from_dir("./data/test")

    # 分析每個檔案
    results = []

    for i, (data_array, true_label) in enumerate(zip(test_data_list, test_labels_list)):
        # 處理單一檔案
        scaled_data = scaler.transform(data_array)
        X_windows, _ = data_loader.create_windows([scaled_data], [0], WINDOW_SIZE, STEP_SIZE)

        if len(X_windows) == 0:
            continue

        X_tensor = torch.tensor(X_windows.transpose(0, 2, 1), dtype=torch.float32).to(device)

        with torch.no_grad():
            outputs = model_instance(X_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

        # 計算異常比例
        abnormal_count = np.sum(probs > 0.5)
        total_windows = len(probs)
        abnormal_ratio = abnormal_count / total_windows

        # 記錄結果
        file_type = "state1 (正常)" if true_label == 0 else "state2 (異常)"
        results.append({
            'file_index': i + 1,
            'file_type': file_type,
            'true_label': true_label,
            'total_windows': total_windows,
            'abnormal_windows': abnormal_count,
            'abnormal_ratio': abnormal_ratio
        })

        print(f"檔案 {i+1} [{file_type}]:")
        print(f"  總窗口數: {total_windows}")
        print(f"  異常窗口數: {abnormal_count}")
        print(f"  異常比例: {abnormal_ratio:.3f} ({abnormal_ratio*100:.1f}%)")

        # 不同閾值的預測結果
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        predictions = []
        for threshold in thresholds:
            pred = 1 if abnormal_ratio > threshold else 0
            correct = "✓" if pred == true_label else "✗"
            predictions.append(f"{threshold}: {pred}{correct}")
        print(f"  閾值預測: {' | '.join(predictions)}")
        print()

    # 總結分析
    print("=== 閾值效果總結 ===")
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    for threshold in thresholds:
        correct_count = 0
        for result in results:
            pred = 1 if result['abnormal_ratio'] > threshold else 0
            if pred == result['true_label']:
                correct_count += 1
        accuracy = correct_count / len(results)
        print(f"閾值 {threshold}: 準確率 {accuracy:.3f} ({correct_count}/{len(results)})")

    # 分析為什麼 0.9 效果最好
    print(f"\n=== 為什麼閾值 0.9 效果最好？ ===")
    state1_ratios = [r['abnormal_ratio'] for r in results if r['true_label'] == 0]
    state2_ratios = [r['abnormal_ratio'] for r in results if r['true_label'] == 1]

    print(f"正常檔案 (state1) 的異常比例:")
    for i, ratio in enumerate(state1_ratios):
        print(f"  檔案 {i+1}: {ratio:.3f} ({ratio*100:.1f}%)")
    print(f"  最大值: {max(state1_ratios):.3f}")

    print(f"\n異常檔案 (state2) 的異常比例:")
    for i, ratio in enumerate(state2_ratios):
        print(f"  檔案 {len(state1_ratios)+i+1}: {ratio:.3f} ({ratio*100:.1f}%)")
    print(f"  最小值: {min(state2_ratios):.3f}")

    print(f"\n💡 關鍵洞察:")
    print(f"- 正常檔案的最高異常比例: {max(state1_ratios):.3f}")
    print(f"- 異常檔案的最低異常比例: {min(state2_ratios):.3f}")
    print(f"- 最佳分界點應在 {max(state1_ratios):.3f} 到 {min(state2_ratios):.3f} 之間")

    return results

if __name__ == "__main__":
    analyze_all_test_files()
