# 建議將此檔案儲存為: src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNC_1D_CNN(nn.Module):

    def __init__(self, num_features, window_size):
        """
        初始化 1D-CNN 模型架構。

        Args:
            num_features (int): 輸入特徵的數量 (例如 15 個欄位)。
            window_size (int): 滑動窗口的長度 (例如 500 個時間步)。
        """
        super(CNC_1D_CNN, self).__init__()

        # --- 1. CNN 特徵提取器 ---

        # 卷積層 1
        # in_channels = num_features (15)
        # out_channels = 32 (自訂的濾鏡數量)
        # kernel_size = 5 (濾鏡大小)
        self.conv1 = nn.Conv1d(
            in_channels=num_features, out_channels=32, kernel_size=5, padding="same"
        )
        # 池化層 1
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # 卷積層 2
        # in_channels = 32 (上一層的輸出)
        # out_channels = 64
        # kernel_size = 5
        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=5, padding="same"
        )
        # 池化層 2
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # 卷積層 3
        self.conv3 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=5, padding="same"
        )
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # --- 2. 壓平層 (Flatten) ---
        # 為了計算壓平後的特徵維度，我們需要動態計算
        # 經過 3 次 pool_size=2 的池化，時間維度會變為 window_size / 2 / 2 / 2
        self.flattened_dim = (window_size // 8) * 128

        # --- 3. MLP 分類器 ---

        # 全連接層 1 (MLP 隱藏層)
        self.fc1 = nn.Linear(in_features=self.flattened_dim, out_features=100)
        # Dropout 層 (防止過擬合)
        self.dropout = nn.Dropout(0.5)

        # 輸出層
        # 輸出 1 個值 (機率)，所以 out_features=1
        self.fc2 = nn.Linear(in_features=100, out_features=1)

    def forward(self, x):
        """
        定義模型的前向傳播路徑。

        Args:
            x (Tensor): 輸入資料，形狀為 (batch_size, num_features, window_size)
        """

        # PyTorch 的 Conv1D 需要 (Batch, Channels, Length)
        # 而您載入的資料可能是 (Batch, Length, Channels)，您需要在訓練迴圈中用 .permute(0, 2, 1) 轉換

        # 區塊 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # 區塊 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # 區塊 3
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # 壓平
        # view(batch_size, -1) 中的 -1 會自動計算維度
        x = x.view(-1, self.flattened_dim)

        # MLP 分類器
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # 輸出層
        # 我們不在這裡使用 sigmoid，因為 PyTorch 的 nn.BCEWithLogitsLoss 會自動處理
        # 這能增加數值穩定性
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    # 這是個簡單的測試，確保模型架構可以運作

    # 假設參數
    BATCH_SIZE = 16
    NUM_FEATURES = 15  # 假設您有 15 個感測器欄位
    WINDOW_SIZE = 500  # 滑動窗口大小

    # 1. 建立一個假的輸入張量 (Batch, Length, Features)
    #    這通常是您用 numpy 載入資料時的格式
    fake_data_numpy_style = torch.randn(BATCH_SIZE, WINDOW_SIZE, NUM_FEATURES)

    # 2. 轉換為 PyTorch Conv1D 需要的格式 (Batch, Features, Length)
    fake_data_torch_style = fake_data_numpy_style.permute(0, 2, 1)

    # 3. 實例化模型
    # 注意：我們需要動態計算 flattened_dim，所以要把 window_size 傳入
    model = CNC_1D_CNN(num_features=NUM_FEATURES, window_size=WINDOW_SIZE)

    # 4. 執行前向傳播
    output = model(fake_data_torch_style)

    # 5. 檢查輸出形狀
    # 應該是 (Batch_Size, 1)
    print(f"輸入形狀 (PyTorch 格式): {fake_data_torch_style.shape}")
    print(f"模型輸出形狀: {output.shape}")
    print("\n模型架構:")
    print(model)
