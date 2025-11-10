#!/usr/bin/env python3
"""
列出所有可用的訓練模型目錄
"""

import os
import glob
from datetime import datetime

def list_training_models():
    """列出所有訓練模型目錄"""
    models_base_dir = "./models"

    if not os.path.exists(models_base_dir):
        print("模型目錄不存在")
        return

    # 尋找所有 training_ 開頭的目錄
    training_dirs = glob.glob(os.path.join(models_base_dir, "training_*"))

    # 檢查是否有舊的模型（直接在 models/ 目錄下）
    old_model_exists = (
        os.path.exists(os.path.join(models_base_dir, "final_model.pth")) and
        os.path.exists(os.path.join(models_base_dir, "final_scaler.joblib"))
    )

    print("=== 可用的訓練模型 ===\n")

    if old_model_exists:
        print("💡 舊格式模型 (向後兼容):")
        print(f"  路徑: {models_base_dir}")
        print("  使用方式: 不需要指定 --model-dir 參數")
        print()

    if training_dirs:
        print("📁 新格式模型 (時間戳記目錄):")
        # 按時間排序
        training_dirs.sort()

        for i, dir_path in enumerate(training_dirs, 1):
            dir_name = os.path.basename(dir_path)

            # 檢查是否包含必要的檔案
            final_model = os.path.join(dir_path, "final_model.pth")
            final_scaler = os.path.join(dir_path, "final_scaler.joblib")

            if os.path.exists(final_model) and os.path.exists(final_scaler):
                # 解析時間戳記
                try:
                    timestamp_str = dir_name.replace("training_", "")
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    readable_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    status = "✅ 完整"
                except:
                    readable_time = "時間未知"
                    status = "✅ 完整"
            else:
                readable_time = "時間未知"
                status = "❌ 不完整"

            print(f"  {i}. {dir_name}")
            print(f"     時間: {readable_time}")
            print(f"     狀態: {status}")
            print(f"     路徑: {dir_path}")
            print(f"     使用方式: --model-dir {dir_path}")
            print()
    else:
        if not old_model_exists:
            print("❌ 找不到任何訓練模型")
            print("請先執行 'python main.py train' 進行訓練")

    print("=== 使用範例 ===")
    print()
    if old_model_exists:
        print("使用舊格式模型:")
        print("  python main.py evaluate --threshold 0.9")
        print("  python main.py predict data/test/state1/state1_1.csv --threshold 0.9")
        print()

    if training_dirs:
        latest_dir = training_dirs[-1]  # 最新的目錄
        print(f"使用最新的訓練模型:")
        print(f"  python main.py evaluate --threshold 0.9 --model-dir {latest_dir}")
        print(f"  python main.py predict data/test/state1/state1_1.csv --threshold 0.9 --model-dir {latest_dir}")

if __name__ == "__main__":
    list_training_models()
