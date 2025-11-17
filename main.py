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
    # 添加可選的隨機種子參數
    train_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="隨機種子碼 (default: 使用 train.py 中的預設值 78)",
    )
    # 添加 Early Stopping 相關參數
    train_parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early Stopping 耐心值 - 等待多少 epoch 沒有改善就停止 (default: 10)",
    )
    train_parser.add_argument(
        "--min-delta",
        type=float,
        default=None,
        help="Early Stopping 最小改善量 - 小於此值視為沒有改善 (default: 0.0001)",
    )
    train_parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="停用 Early Stopping 機制",
    )

    # --- 4. "evaluate" 指令 ---
    # 當使用者輸入 'python main.py evaluate' 時:
    evaluate_parser = subparsers.add_parser(
        "evaluate", help="在測試集上評估訓練好的模型"
    )
    # 添加可選的閾值參數
    evaluate_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="檔案級別投票閾值 (不指定則使用 golden threshold)",
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
    predict_parser = subparsers.add_parser("predict", help="對新的 CSV 檔案進行推論 (單一檔案或批次)")

    # 支援兩種模式: 單一檔案或批次預測
    predict_parser.add_argument(
        "file",
        type=str,
        nargs='?',  # 使 file 變成可選參數
        default=None,
        help="要預測的單一 CSV 檔案路徑 (與 --pred-dir 擇一使用)"
    )

    # 批次預測參數
    predict_parser.add_argument(
        "--pred-dir",
        type=str,
        default=None,
        help="批次預測:要預測的資料夾路徑 (包含多個 CSV 檔案)",
    )

    predict_parser.add_argument(
        "--output-dir",
        type=str,
        default="./predictions",
        help="批次預測:輸出結果的目錄路徑 (default: ./predictions)",
    )

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

        # 準備訓練參數
        train_kwargs = {}

        if args.seed is not None:
            print(f"--- 使用自訂隨機種子碼: {args.seed} ---")
            train_kwargs['random_state'] = args.seed

        if args.no_early_stopping:
            print("--- Early Stopping 已停用 ---")
            train_kwargs['use_early_stopping'] = False
        else:
            if args.patience is not None:
                print(f"--- Early Stopping Patience: {args.patience} ---")
                train_kwargs['early_stopping_patience'] = args.patience
            if args.min_delta is not None:
                print(f"--- Early Stopping Min Delta: {args.min_delta} ---")
                train_kwargs['early_stopping_delta'] = args.min_delta

        train.run_training(**train_kwargs)
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
        # 檢查是單一檔案預測還是批次預測
        if args.pred_dir:
            # 批次預測模式
            print(f"--- 啟動 [批次預測] 流程 ---")
            print(f"預測資料夾: {args.pred_dir}")
            print(f"輸出目錄: {args.output_dir}")

            if hasattr(args, 'model_dir'):
                model_dir = getattr(args, 'model_dir')
            else:
                model_dir = None

            predict.batch_predict(
                pred_dir=args.pred_dir,
                model_dir=model_dir,
                threshold=args.threshold,
                output_dir=args.output_dir,
                strategy=args.strategy
            )
            print(f"--- [批次預測] 流程完畢 ---")

        elif args.file:
            # 單一檔案預測模式
            print(f"--- 啟動 [單一預測] 流程: {args.file} ---")

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
            print(f"--- [單一預測] 流程完畢 ---")

        else:
            print("錯誤: 請指定要預測的檔案 (file) 或資料夾 (--pred-dir)")
            print("範例:")
            print("  單一檔案: python main.py predict 檔案.csv --threshold 0.1")
            print("  批次預測: python main.py predict --pred-dir ./test_data/ --output-dir ./results/ --threshold 0.1")


if __name__ == "__main__":
    main()
