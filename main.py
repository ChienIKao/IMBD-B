# 請將此檔案儲存為: main.py (放在專案根目錄)

import argparse
import os
import sys
import shutil

# 將 src 目錄添加到 Python 的搜尋路徑中
# 這樣 main.py 才能 "import src"
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 從 src 模組導入我們寫好的執行函數
from src import train
from src import predict
from src import evaluate


def main():
    # 1. 建立一個參數解析器
    parser = argparse.ArgumentParser(
        description="IMBD 2025 競賽 - CNC 機台狀態分類專案"
    )

    # 2. 建立 "子命令" (subparsers)
    #    這能讓我們做出像 'python main.py train', 'python main.py predict' 和 'python main.py evaluate' 這樣的功能
    subparsers = parser.add_subparsers(
        dest="command", help="要執行的動作 (train, predict 或 evaluate)", required=True
    )

    # --- 3. "train" 指令 ---
    # 當使用者輸入 'python main.py train' 時:
    train_parser = subparsers.add_parser(
        "train", help="開始 K-Fold 交叉驗證並訓練最終模型"
    )
    # 'train' 指令不需要額外參數，因為設定都在 src/train.py 裡了

    # --- 4. "evaluate" 指令 ---
    # 當使用者輸入 'python main.py evaluate' 時:
    evaluate_parser = subparsers.add_parser(
        "evaluate", help="在測試集上評估訓練好的模型"
    )
    # 添加可選的閾值參數
    evaluate_parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="檔案級別投票閾值 (default: 0.1)",
    )
    # 添加可選的模型目錄參數
    evaluate_parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="模型目錄路徑 (default: ./models)",
    )

    # --- 5. "predict" 指令 ---
    # 當使用者輸入 'python main.py predict [檔案路徑]' 時:
    predict_parser = subparsers.add_parser("predict", help="對新的 CSV 檔案進行推論")
    # "predict" 指令需要一個「必要的」參數：檔案路徑
    predict_parser.add_argument("file", type=str, help="要預測的單一 CSV 檔案路徑")
    # 我們還可以加入「可選的」參數，來控制推論策略
    predict_parser.add_argument(
        "--strategy",
        type=str,
        default="majority_vote",
        choices=["majority_vote", "mean_prob"],
        help="彙總策略 (default: 'majority_vote')",
    )
    predict_parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="投票法策略的異常閾值 (default: 0.1)",
    )
    # 添加可選的模型目錄參數
    predict_parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="模型目錄路徑 (default: ./models)",
    )

    # 5. 解析使用者輸入的指令
    args = parser.parse_args()

    # --- 6. 根據指令執行對應的函數 ---

    if args.command == "train":
        print("--- 啟動 [訓練] 流程 ---")
        train.run_training()
        print("--- [訓練] 流程完畢 ---")

    elif args.command == "evaluate":
        print(f"--- 啟動 [測試集評估] 流程 (閾值: {args.threshold}) ---")
        if hasattr(args, 'model_dir'):
            model_dir = getattr(args, 'model_dir')
        else:
            model_dir = None
        evaluate.evaluate_on_test_set(threshold=args.threshold, model_dir=model_dir)
        print("--- [測試集評估] 流程完畢 ---")

    elif args.command == "predict":
        print(f"--- 啟動 [推論] 流程: {args.file} ---")
        # 將解析到的參數傳遞給 run_inference 函數
        if hasattr(args, 'model_dir'):
            model_dir = getattr(args, 'model_dir')
        else:
            model_dir = None
        prediction = predict.run_inference(
            test_file_path=args.file,
            strategy=args.strategy,
            threshold=args.threshold,
            model_dir=model_dir
        )
        print(f"--- [推論] 流程完畢 ---")


if __name__ == "__main__":
    main()
