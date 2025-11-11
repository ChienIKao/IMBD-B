#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
檔案異常比例分析工具 - 可用於任何 CSV 檔案 (不需要知道答案)

用途:
1. 分析單一檔案或整個資料夾的異常窗口比例
2. 查看每個檔案在不同 threshold 下會被預測為什麼
3. 比賽時用來檢查預測的合理性

使用方式:
  # 分析單一檔案
  python analyze_files.py --file 檔案.csv --model-dir ./models/training_XXX

  # 分析整個資料夾
  python analyze_files.py --dir 資料夾/ --model-dir ./models/training_XXX

  # 指定要測試的 threshold
  python analyze_files.py --dir 資料夾/ --thresholds 0.05 0.1 0.15 0.2
"""

import os
import torch
import numpy as np
import argparse
import pandas as pd
from src import data_loader, model

def analyze_file(file_path, model_instance, scaler, device, window_size=500, step_size=50):
    """
    分析單一檔案的異常窗口比例

    Args:
        file_path: CSV 檔案路徑
        model_instance: 訓練好的模型
        scaler: 訓練好的 scaler
        device: 計算裝置
        window_size: 窗口大小
        step_size: 步長

    Returns:
        dict: 包含分析結果的字典
    """
    try:
        # 載入並處理資料
        data_array = data_loader.load_single_csv(file_path)
        scaled_data = scaler.transform(data_array)
        X_windows, _ = data_loader.create_windows([scaled_data], [0], window_size, step_size)

        if len(X_windows) == 0:
            return {
                'file_name': os.path.basename(file_path),
                'error': 'File too short to create windows'
            }

        # 預測
        X_tensor = torch.tensor(X_windows.transpose(0, 2, 1), dtype=torch.float32).to(device)

        with torch.no_grad():
            outputs = model_instance(X_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

        # 計算統計
        abnormal_count = np.sum(probs > 0.5)
        total_windows = len(probs)
        abnormal_ratio = abnormal_count / total_windows

        # 計算異常機率的分布
        prob_stats = {
            'mean': float(np.mean(probs)),
            'std': float(np.std(probs)),
            'min': float(np.min(probs)),
            'max': float(np.max(probs)),
            'median': float(np.median(probs))
        }

        return {
            'file_name': os.path.basename(file_path),
            'file_path': file_path,
            'total_windows': total_windows,
            'abnormal_windows': abnormal_count,
            'abnormal_ratio': abnormal_ratio,
            'prob_stats': prob_stats,
            'all_probs': probs
        }

    except Exception as e:
        return {
            'file_name': os.path.basename(file_path),
            'error': str(e)
        }

def analyze_files(file_paths, model_dir, thresholds=[0.05, 0.1, 0.15, 0.2, 0.3]):
    """
    分析多個檔案的異常比例

    Args:
        file_paths: CSV 檔案路徑列表
        model_dir: 模型目錄
        thresholds: 要測試的 threshold 列表
    """
    print("=" * 70)
    print("檔案異常比例分析工具")
    print("=" * 70)
    print(f"\n模型目錄: {model_dir}")
    print(f"要分析的檔案數量: {len(file_paths)}")
    print(f"測試的 thresholds: {thresholds}\n")

    # 載入模型和 scaler
    FINAL_MODEL_NAME = "final_model.pth"
    FINAL_SCALER_NAME = "final_scaler.joblib"
    WINDOW_SIZE = 500
    STEP_SIZE = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 載入 scaler
    scaler_path = os.path.join(model_dir, FINAL_SCALER_NAME)
    if not os.path.exists(scaler_path):
        print(f"❌ 錯誤: 找不到 Scaler: {scaler_path}")
        return

    scaler = data_loader.load_scaler(scaler_path)
    NUM_FEATURES = scaler.n_features_in_

    # 載入模型
    model_instance = model.CNC_1D_CNN(
        num_features=NUM_FEATURES, window_size=WINDOW_SIZE
    ).to(device)

    model_path = os.path.join(model_dir, FINAL_MODEL_NAME)
    if not os.path.exists(model_path):
        print(f"❌ 錯誤: 找不到模型: {model_path}")
        return

    model_instance.load_state_dict(torch.load(model_path, map_location=device))
    model_instance.eval()

    print("✓ 成功載入模型和 Scaler\n")
    print("=" * 70)
    print("開始分析...")
    print("=" * 70)

    # 分析所有檔案
    results = []

    for i, file_path in enumerate(file_paths, 1):
        print(f"\n[{i}/{len(file_paths)}] {os.path.basename(file_path)}")

        result = analyze_file(file_path, model_instance, scaler, device, WINDOW_SIZE, STEP_SIZE)

        if 'error' in result:
            print(f"  ❌ 錯誤: {result['error']}")
            continue

        results.append(result)

        # 顯示基本資訊
        print(f"  總窗口數: {result['total_windows']}")
        print(f"  異常窗口數: {result['abnormal_windows']}")
        print(f"  異常比例: {result['abnormal_ratio']:.3f} ({result['abnormal_ratio']*100:.1f}%)")

        # 顯示機率統計
        prob_stats = result['prob_stats']
        print(f"  機率統計: mean={prob_stats['mean']:.3f}, std={prob_stats['std']:.3f}, "
              f"min={prob_stats['min']:.3f}, max={prob_stats['max']:.3f}")

        # 顯示不同 threshold 的預測結果
        predictions = []
        for th in thresholds:
            pred = 1 if result['abnormal_ratio'] > th else 0
            pred_label = 'state2' if pred == 1 else 'state1'
            predictions.append(f"{th}: {pred_label}")
        print(f"  預測結果: {' | '.join(predictions)}")

    if not results:
        print("\n沒有成功分析的檔案")
        return

    # 總結分析
    print("\n" + "=" * 70)
    print("總結分析")
    print("=" * 70)

    # 統計不同 threshold 下的預測分布
    for th in thresholds:
        state1_count = sum(1 for r in results if r['abnormal_ratio'] <= th)
        state2_count = len(results) - state1_count
        print(f"\nThreshold = {th}:")
        print(f"  預測為 state1 (正常): {state1_count} 個檔案")
        print(f"  預測為 state2 (異常): {state2_count} 個檔案")
        print(f"  異常比例: {state2_count/len(results)*100:.1f}%")

    # 異常比例分布
    ratios = [r['abnormal_ratio'] for r in results]
    print(f"\n異常比例統計:")
    print(f"  最小值: {min(ratios):.3f}")
    print(f"  最大值: {max(ratios):.3f}")
    print(f"  平均值: {np.mean(ratios):.3f}")
    print(f"  中位數: {np.median(ratios):.3f}")
    print(f"  標準差: {np.std(ratios):.3f}")

    # 儲存詳細結果到 CSV
    df_results = pd.DataFrame([
        {
            'file_name': r['file_name'],
            'total_windows': r['total_windows'],
            'abnormal_windows': r['abnormal_windows'],
            'abnormal_ratio': r['abnormal_ratio'],
            'prob_mean': r['prob_stats']['mean'],
            'prob_std': r['prob_stats']['std'],
            'prob_min': r['prob_stats']['min'],
            'prob_max': r['prob_stats']['max'],
            **{f'pred_th_{th}': 1 if r['abnormal_ratio'] > th else 0 for th in thresholds}
        }
        for r in results
    ])

    output_file = 'analysis_results.csv'
    df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ 詳細結果已儲存至: {output_file}")

    # 顯示建議
    print("\n" + "=" * 70)
    print("💡 建議")
    print("=" * 70)

    # 找出可能有問題的檔案
    mean_ratio = np.mean(ratios)
    std_ratio = np.std(ratios)

    outliers = [r for r in results if abs(r['abnormal_ratio'] - mean_ratio) > 2 * std_ratio]
    if outliers:
        print(f"\n⚠️  發現 {len(outliers)} 個異常值檔案 (異常比例與平均值相差 > 2 個標準差):")
        for r in outliers:
            print(f"  - {r['file_name']}: {r['abnormal_ratio']:.3f}")

    # Threshold 建議
    print(f"\n根據分析結果:")
    print(f"  - 如果您的資料應該大部分是正常的,考慮使用較低的 threshold (0.1-0.15)")
    print(f"  - 如果您的資料應該有較多異常,考慮使用較高的 threshold (0.2-0.3)")
    print(f"  - 當前資料的平均異常比例為 {np.mean(ratios):.3f}")

def main():
    parser = argparse.ArgumentParser(
        description="分析 CSV 檔案的異常窗口比例 (不需要知道答案)"
    )

    # 輸入選項: 單一檔案或資料夾
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--file",
        type=str,
        help="要分析的單一 CSV 檔案"
    )
    group.add_argument(
        "--dir",
        type=str,
        help="要分析的資料夾 (包含多個 CSV 檔案)"
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default="./models",
        help="模型目錄路徑 (default: ./models)"
    )

    parser.add_argument(
        "--thresholds",
        nargs='+',
        type=float,
        default=[0.05, 0.1, 0.15, 0.2, 0.3],
        help="要測試的 threshold 列表 (default: 0.05 0.1 0.15 0.2 0.3)"
    )

    args = parser.parse_args()

    # 收集要分析的檔案
    file_paths = []

    if args.file:
        if not os.path.exists(args.file):
            print(f"❌ 錯誤: 檔案不存在: {args.file}")
            return
        file_paths = [args.file]

    elif args.dir:
        if not os.path.exists(args.dir):
            print(f"❌ 錯誤: 資料夾不存在: {args.dir}")
            return

        # 找出所有 CSV 檔案
        for root, dirs, files in os.walk(args.dir):
            for file in files:
                if file.endswith('.csv'):
                    file_paths.append(os.path.join(root, file))

        if not file_paths:
            print(f"❌ 錯誤: 在 {args.dir} 中找不到任何 CSV 檔案")
            return

    # 執行分析
    analyze_files(file_paths, args.model_dir, args.thresholds)

if __name__ == "__main__":
    main()
