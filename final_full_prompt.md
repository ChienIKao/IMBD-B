# 最終整合版 LLM Prompt（含所有復現一致性要求）

你是一位熟悉 PyTorch、1D-CNN、Grad-CAM
與時間序列可視化的資深機器學習工程師。

## 【重要專案限制】

-   請「以當前資料夾架構為主」撰寫程式，不得任意新增 package
    或更改路徑結構。
-   必須沿用現有專案的資料讀取、前處理、scaler 載入、模型定義方式。
-   若需要假設類別名稱或路徑，請在程式碼內標註「TODO：依實際專案修改」。

我需要你撰寫一個完整的 `grad_cam_visualization.py`（單一 Python
檔），功能包含： - 從單一 CSV 產生 Grad-CAM 時間注意力 - 推導
channel-wise attention - 產生與專案一致的線圖與熱力圖 - 計算 feature
importance（Max / Ratio / Mean） - 避免復現不一致問題（非常重要）

------------------------------------------------------------------------

## 一、資料前處理（必須完全一致）

1.  跳過前兩列（欄位名稱與單位）
2.  從 C 欄開始取 35 個 features\
3.  套用訓練時保存的 StandardScaler（不可重新 fit）\
4.  sliding window：
    -   window_length = 500\
    -   stride = 50\
    -   window shape = `(1, 35, 500)`\
    -   回傳 global start index

------------------------------------------------------------------------

## 二、模型架構（hook 層必須一致）

請使用專案內定義的 CNN：

Input: (Batch, 35, 500) → Conv1D(35→32,k=5), ReLU, MaxPool(2) → (32,250)
→ Conv1D(32→64,k=5), ReLU, MaxPool(2) → (64,125) → Conv1D(64→128,k=5),
ReLU, MaxPool(2) → (128,62) → Flatten → Linear(7936→100) → Dropout →
Linear → logit

**Grad-CAM 的 hook 層必須是使用者指定的最後一個 Conv1D（通常是
conv3）。**

------------------------------------------------------------------------

## 三、Grad-CAM 計算（標準流程）

對每個 window：

1.  forward → output logit：

    -   二元分類只有一個 logit → `output[0,0]`
    -   不得先過 sigmoid

2.  backward w.r.t target layer output\

3.  α = GAP(gradient)\

4.  CAM：

        cam = ReLU( Σ_c α_c * A_c )

5.  interpolate → 長度 500\

6.  per-window min-max normalize → `[0,1]`

------------------------------------------------------------------------

## 四、多 window 合併（必須使用平均）

    global_cam = zeros(T)
    count = zeros(T)

    for window k:
        for i in range(500):
            global_cam[start+i] += cam[k][i]
            count[start+i] += 1

    global_cam[t] /= count[t]
    global_cam = minmax(global_cam)

**不得改用 max / sum / softmax / weighted 版本。**

------------------------------------------------------------------------

## 五、channel-wise attention（必須使用本方法）

    importance[t,c] = |X[t,c]| * A[t]
    channel_attention[c,:] = minmax(importance[:,c])

不得改用 input gradient / IG / SHAP / CAM++ 等方法。

------------------------------------------------------------------------

## 六、繪圖（需與提供的圖格式完全相同）

### 1. 上半部（line + red attention background）

-   藍線：X\[:, c\]
-   紅線：channel_attention\[c, :\]
-   紅色背景：fill_between(0..T, 0, channel_attention\[c,:\],
    alpha≈0.25)

### 2. 下半部（1×T Heatmap）

-   colormap：Reds
-   高度固定 1 行

------------------------------------------------------------------------

## 七、Feature Importance（必須這樣算）

### 1. Maximum Single Feature Attention

    max_score[c] = max_t(channel_attention[c,t])

### 2. Single/Overall Ratio

    ratio[c] = max_score[c] / mean(max_score[:])

### 3. Mean Single Feature Attention

    mean_score[c] = mean_t(channel_attention[c,t])

輸出各榜單前 10 名。

------------------------------------------------------------------------

## 八、避免復現失敗（必須遵守）

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    numpy.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model.eval()

-   不可重新 fit scaler\
-   不可改動 window 分割方法\
-   不可改變 hook 層\
-   不可改變 CAM 公式\
-   不可改變 normalization 步驟\
-   不可對尾端不足 500 的段落補 window\
-   不可 batch 多個 window 一起做 backward

------------------------------------------------------------------------

## 九、程式架構（請拆分成以下函式）

-   `load_model()`
-   `load_scaler()`
-   `read_and_preprocess_csv()`
-   `extract_windows()`
-   `compute_grad_cam_for_window()`
-   `aggregate_global_cam()`
-   `compute_channel_attention()`
-   `plot_line_with_attention()`
-   `plot_attention_heatmap()`
-   `compute_feature_importance()`
-   `main()`

------------------------------------------------------------------------

## 十、輸出要求

請輸出完整、可直接執行的 Python 檔：

    if __name__ == "__main__":
        main()

不得省略 import 或任何必要函式。
