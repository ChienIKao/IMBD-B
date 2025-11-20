# 建議將此檔案儲存為: src/predict.py

import os
import torch
import numpy as np
import warnings

# 載入我們自定義的模組
from src import data_loader
from src import model

# --- 1. 集中管理設定參數 (必須與 train.py 一致) ---

MODEL_BASE_DIR = "./models"
DEFAULT_MODEL_DIR = "./models"  # 為了向後兼容，預設使用舊的路徑
FINAL_MODEL_NAME = "final_model.pth"
FINAL_SCALER_NAME = "final_scaler.joblib"

# 這些參數必須與您訓練 final_model 時所用的完全一致
WINDOW_SIZE = 500
STEP_SIZE = 50  # 推論時，步長可以設得更小以獲得更精細的預測，但必須與訓練時一致才能重現 K-Fold 結果。
# 為了安全起見，我們保持與 train.py 一致。

# --- 2. 彙總投票策略 (Aggregation) ---


def aggregate_predictions(snippet_preds, strategy="majority_vote", threshold=0.1):
    """
    將一個檔案的所有片段 (snippet) 預測彙總成單一預測。

    Args:
        snippet_preds (np.array): 來自模型的原始機率 (shape: [N_snippets,])
        strategy (str): 'majority_vote' 或 'mean_prob'
        threshold (float): 'majority_vote' 策略的閾值
                           (例如：只要有 10% 的片段被判為異常，整個檔案就為異常)

    Returns:
        int: 1 (正常) 或 2 (異常)
    """

    if strategy == "mean_prob":
        # 策略 A: 平均機率法
        # 計算所有片段的平均異常機率
        mean_prob = np.mean(snippet_preds)
        final_prediction_binary = 1 if mean_prob > 0.5 else 0

    elif strategy == "majority_vote":
        # 策略 B: 投票法 (推薦用於異常檢測)
        # 1. 將機率轉為 0 或 1
        votes = (snippet_preds > 0.5).astype(int)

        # 2. 計算「異常 (1)」票的比例
        abnormal_ratio = np.mean(votes)

        # 3. 如果異常票的比例超過閾值，則判為異常
        final_prediction_binary = 1 if abnormal_ratio > threshold else 0

    else:
        raise ValueError("未知的彙總策略")

    # 將內部 0/1 標籤映射為對外 1/2:
    # 0 (正常) -> 1, 1 (異常) -> 2
    final_prediction = 1 if final_prediction_binary == 0 else 2

    return final_prediction


# --- 3. 主要推論執行函數 ---


def run_inference(test_file_path, strategy="majority_vote", threshold=0.1, model_dir=None, verbose=True):
    """
    對單一 CSV 檔案執行完整的推論流程。

    Args:
        test_file_path (str): 要預測的 CSV 檔案路徑
        strategy (str): 'majority_vote' 或 'mean_prob'
        threshold (float): 投票法閾值
        model_dir (str): 模型目錄路徑，如果為 None 則使用預設目錄
        verbose (bool): 是否顯示詳細輸出

    Returns:
        int: 最終預測 (1=正常, 2=異常)
    """

    # 如果沒有指定模型目錄，使用預設目錄
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR

    if verbose:
        print(f"--- 開始推論: {test_file_path} ---")
        print(f"--- 使用模型目錄: {model_dir} ---")

    # 設置裝置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- A. 載入模型和 Scaler ---

    # 1. 載入 Scaler
    scaler_path = os.path.join(model_dir, FINAL_SCALER_NAME)
    if not os.path.exists(scaler_path):
        print(f"錯誤: 找不到 Scaler: {scaler_path}")
        return
    scaler = data_loader.load_scaler(scaler_path)

    # 2. 載入模型
    #    我們需要先知道 num_features 才能初始化模型
    #    scaler.n_features_in_ 可以告訴我們
    NUM_FEATURES = scaler.n_features_in_

    model_instance = model.CNC_1D_CNN(
        num_features=NUM_FEATURES, window_size=WINDOW_SIZE
    ).to(device)
    model_path = os.path.join(model_dir, FINAL_MODEL_NAME)

    if not os.path.exists(model_path):
        print(f"錯誤: 找不到模型: {model_path}")
        return

    model_instance.load_state_dict(torch.load(model_path, map_location=device))
    model_instance.eval()  # **非常重要：設置為評估模式**

    if verbose:
        print(f"模型 {FINAL_MODEL_NAME} 和 {FINAL_SCALER_NAME} 載入成功。")

    # --- B. 處理測試檔案 (與訓練時完全一致) ---

    # 1. 載入單一 CSV
    try:
        test_data_array = data_loader.load_single_csv(test_file_path)
    except FileNotFoundError:
        print(f"錯誤: 測試檔案不存在: {test_file_path}")
        return

    # 2. 套用 Scaler (注意：只用 .transform())
    scaled_test_array = scaler.transform(test_data_array)

    # 3. 創建滑動窗口
    #    我們將 data 和 label 封裝成 list (即使只有一個檔案) 以符合 create_windows 函數
    #    標籤 (0) 在這裡只是佔位符，不會被使用
    X_snippets, y_snippets = data_loader.create_windows(
        [scaled_test_array], [0], WINDOW_SIZE, STEP_SIZE
    )

    if len(X_snippets) == 0:
        print(
            f"錯誤: 檔案 {test_file_path} 太短，無法製作任何窗口 (長度 < {WINDOW_SIZE})"
        )
        return

    if verbose:
        print(f"已從檔案中創建 {len(X_snippets)} 個預測片段 (snippets)。")

    # 4. 創建 PyTorch DataLoader
    #    (不需要 PyTorch Dataset，我們手動處理批次更簡單)
    #    將 (N, L, C) 轉為 (N, C, L)
    X_snippets_torch = torch.tensor(
        X_snippets.transpose(0, 2, 1), dtype=torch.float32
    ).to(device)

    # --- C. 執行模型預測 ---

    all_snippet_probs = []

    with torch.no_grad():
        # 為了防止記憶體不足，我們可以分批次 (batch) 推論
        # 但如果測試檔案不大 (例如幾百個片段)，也可以一次全部推論
        # 這裡為簡單起見，一次全部推論

        # 前向傳播
        outputs = model_instance(X_snippets_torch)

        # 將 Logits 轉為機率
        probs = torch.sigmoid(outputs)

        # 移回 CPU 並轉為 NumPy
        all_snippet_probs = probs.cpu().numpy().flatten()

    # --- D. 彙總投票 ---

    final_prediction = aggregate_predictions(
        all_snippet_probs, strategy=strategy, threshold=threshold
    )

    if verbose:
        result_str = "狀態 2 (異常)" if final_prediction == 2 else "狀態 1 (正常)"
        print(f"\n--- 最終預測結果 ---")
        print(f"檔案: {os.path.basename(test_file_path)}")
        print(f"預測: {final_prediction} ( {result_str} )")

    return final_prediction


def batch_predict(pred_dir, model_dir, threshold, output_dir, strategy="majority_vote"):
    """
    批次預測整個資料夾的 CSV 檔案

    Args:
        pred_dir: 要預測的資料夾路徑
        model_dir: 模型目錄路徑
        threshold: 預測閾值
        output_dir: 輸出結果的目錄路徑
        strategy: 彙總策略

    Returns:
        results: 預測結果列表
    """
    import pandas as pd
    from tqdm import tqdm

    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)

    # 找出所有 CSV 檔案
    csv_files = []
    for root, dirs, files in os.walk(pred_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))

    if len(csv_files) == 0:
        print(f"錯誤: 在 {pred_dir} 中找不到任何 CSV 檔案")
        return []

    print(f"\n--- 批次預測工具 ---")
    print(f"預測資料夾: {pred_dir}")
    print(f"模型目錄: {model_dir}")
    print(f"Threshold: {threshold}")
    print(f"策略: {strategy}")
    print(f"找到 {len(csv_files)} 個檔案")
    print(f"輸出目錄: {output_dir}")
    print()

    # 儲存預測結果
    results = []

    # 使用進度條顯示預測進度
    print("開始預測...")
    for file_path in tqdm(csv_files, desc="預測進度"):
        file_name = os.path.basename(file_path)

        try:
            # 執行預測 (不顯示詳細輸出)
            prediction = run_inference(
                test_file_path=file_path,
                strategy=strategy,
                threshold=threshold,
                model_dir=model_dir,
                verbose=False  # 批次預測時不顯示每個檔案的詳細輸出
            )

            results.append({
                'file_name': file_name,
                'prediction': prediction,
                'prediction_label': 'state2' if prediction == 2 else 'state1'
            })

        except Exception as e:
            print(f"\n警告: 處理 {file_name} 時發生錯誤: {e}")
            results.append({
                'file_name': file_name,
                'prediction': -1,
                'prediction_label': 'ERROR'
            })

    # 將結果儲存為 CSV
    output_file = os.path.join(output_dir, 'predictions.csv')
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"\n--- 預測完成 ---")
    print(f"總檔案數: {len(results)}")
    print(f"預測為 state1 (正常, label=1): {sum(1 for r in results if r['prediction'] == 1)} 個")
    print(f"預測為 state2 (異常, label=2): {sum(1 for r in results if r['prediction'] == 2)} 個")
    if any(r['prediction'] == -1 for r in results):
        print(f"錯誤: {sum(1 for r in results if r['prediction'] == -1)} 個")
    print(f"\n結果已儲存至: {output_file}")

    # 顯示結果摘要表格
    print("\n--- 預測結果 ---")
    print(df_results.to_string(index=False))

    return results


if __name__ == "__main__":
    # --- 執行範例 ---
    # 假設您在 raw_data/狀態2/ 中有一個檔案可用來測試

    # 警告：此處路徑為範例，請替換為您真實的檔案路徑
    TEST_FILE = "./raw_data/狀態2/循圓數據_狀態2_01.csv"

    if not os.path.exists(TEST_FILE):
        print(f"警告：找不到測試檔案 '{TEST_FILE}'。")
        print("請修改 src/predict.py 中 'TEST_FILE' 的路徑以進行測試。")
    else:
        # 使用我們推薦的「投票法」，閾值設為 10%
        # 意思是只要有 10% 的片段被判為異常，整個檔案就算異常
        run_inference(TEST_FILE, strategy="majority_vote", threshold=0.1)
