# Grad-CAM Technical Implementation Details

## 🔬 **內部技術細節與演算法實現**

### 📋 **目錄**
1. [核心架構設計](#核心架構設計)
2. [演算法實現細節](#演算法實現細節)
3. [數據處理管線](#數據處理管線)
4. [視覺化渲染引擎](#視覺化渲染引擎)
5. [效能優化策略](#效能優化策略)
6. [程式碼架構分析](#程式碼架構分析)

---

## 🏗️ **核心架構設計**

### **整體架構圖**
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Input    │───▶│  Preprocessing   │───▶│ Model Loading   │
│  - CSV Reader   │    │  - Standardizer  │    │ - 1D-CNN Model  │
│  - Validation   │    │  - Window Split  │    │ - Scaler Load   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Grad-CAM Engine │    │ Attention Calc   │    │ Visualization   │
│ - Hook Register │    │ - Regional Enh   │    │ - Plot Engine   │
│ - Gradient Flow │    │ - Downsampling   │    │ - Export System │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### **模組相依關係**
```python
grad_cam_visualization.py
├── torch (PyTorch 深度學習框架)
├── numpy (數值計算)
├── matplotlib (視覺化)
├── src.model (自定義 1D-CNN 模型)
├── src.data_loader (資料載入器)
└── joblib (模型序列化)
```

---

## ⚙️ **演算法實現細節**

### **1. Grad-CAM 核心演算法**

#### **Hook 機制實現**
```python
def compute_grad_cam_for_window(model_instance, window, target_layer_name="conv3"):
    # 1. 建立梯度和激活值容器
    activations = None
    gradients = None
    
    # 2. 定義前向 Hook (捕獲激活值)
    def forward_hook(module, input, output):
        nonlocal activations
        activations = output  # Shape: (1, channels, length)
    
    # 3. 定義反向 Hook (捕獲梯度)
    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]  # Shape: (1, channels, length)
    
    # 4. 註冊 Hook 到目標層
    target_layer = getattr(model_instance, target_layer_name)
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)
```

#### **CAM 計算數學公式**
```python
# 1. 全域平均池化計算權重
α_c = (1/Z) * Σ(∂y_c/∂A_ij)  # Z = width × height

# 2. 加權組合生成 CAM
L_CAM = Σ(α_c × A_c)

# 3. ReLU 激活保留正值
L_CAM = ReLU(L_CAM)

# 4. Min-Max 歸一化到 [0,1]
L_CAM_norm = (L_CAM - min) / (max - min)
```

#### **實際程式碼實現**
```python
# 計算全域平均池化權重
alpha = torch.mean(gradients, dim=2, keepdim=True)  # (1, channels, 1)

# 加權組合
cam = torch.sum(alpha * activations, dim=1)  # (1, length)

# ReLU 激活
cam = F.relu(cam)  # 只保留正值

# 插值到標準長度
if cam.size(0) != WINDOW_SIZE:
    cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), 
                       size=WINDOW_SIZE, mode='linear')

# 歸一化
cam_np = cam.detach().cpu().numpy()
cam_normalized = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())
```

### **2. 滑動窗口聚合演算法**

#### **重疊窗口處理**
```python
def aggregate_global_cam(windows, start_indices, model_instance, total_length):
    global_cam = np.zeros(total_length)
    count = np.zeros(total_length)
    
    for k, (window, start) in enumerate(zip(windows, start_indices)):
        # 計算單個窗口的 CAM
        cam_k = compute_grad_cam_for_window(model_instance, window)
        
        # 累積到全域 CAM
        for i in range(WINDOW_SIZE):
            global_idx = start + i
            if global_idx < total_length:
                global_cam[global_idx] += cam_k[i]
                count[global_idx] += 1
    
    # 平均化處理
    mask = count > 0
    global_cam[mask] /= count[mask]
    
    return global_cam
```

#### **覆蓋度分析**
```
窗口設定:
- 窗口大小: 500 時間步
- 滑動步長: 50 時間步  
- 重疊比例: 90% (450/500)

覆蓋模式:
- 起始區域 (0-499): 覆蓋度 1-10
- 中心區域 (500-T-500): 覆蓋度恆定 10
- 結束區域 (T-499-T): 覆蓋度 10-1
```

### **3. 區域增強演算法**

#### **滑動區域增強**
```python
def enhance_attention_regionally(attention_values, window_size=500):
    T = len(attention_values)
    enhanced_attention = np.zeros_like(attention_values)
    
    for center_idx in range(T):
        # 定義鄰域窗口
        half_window = window_size // 2
        start_idx = max(0, center_idx - half_window)
        end_idx = min(T, center_idx + half_window)
        
        region = attention_values[start_idx:end_idx]
        max_attention = np.max(region)
        
        if max_attention > 0.05:  # 閾值過濾
            # 找到區域峰值位置
            max_idx = np.argmax(region)
            center_relative = center_idx - start_idx
            
            # 計算距離峰值的距離
            distance_to_peak = abs(center_relative - max_idx)
            
            # 高斯增強因子
            enhancement_factor = np.exp(-distance_to_peak**2 / (2 * (window_size/8)**2))
            
            # 增強公式
            enhancement = 0.3 * enhancement_factor * max_attention
            enhanced_value = attention_values[center_idx] * (1 + enhancement)
            
            enhanced_attention[center_idx] = np.clip(enhanced_value, 0, 1)
        else:
            enhanced_attention[center_idx] = attention_values[center_idx]
    
    return enhanced_attention
```

### **4. 局部最大值取樣演算法**

#### **取樣策略**
```python
def downsample_and_interpolate(values, sample_step=200, local_window=200):
    T = len(values)
    half_window = local_window // 2
    
    # 建立取樣點索引
    sample_indices = np.arange(0, T, sample_step)
    if sample_indices[-1] != T - 1:
        sample_indices = np.append(sample_indices, T - 1)
    
    sampled_values = []
    
    # 對每個取樣點提取局部最大值
    for idx in sample_indices:
        start_idx = max(0, idx - half_window)
        end_idx = min(T, idx + half_window + 1)
        
        local_window_values = values[start_idx:end_idx]
        max_value = np.max(local_window_values)
        
        sampled_values.append(max_value)
    
    # 線性插值回到原始長度
    interpolated = np.interp(np.arange(T), sample_indices, sampled_values)
    
    return interpolated
```

---

## 🔄 **數據處理管線**

### **1. CSV 讀取與預處理**

#### **資料載入流程**
```python
def read_and_preprocess_csv(csv_path, scaler):
    # 1. 讀取 CSV (跳過單位行)
    df = pd.read_csv(csv_path, header=0, skiprows=[1])
    
    # 2. 提取特徵名稱 (從第 C 欄開始)
    feature_names = df.columns[2:2+NUM_FEATURES].tolist()
    
    # 3. 使用專案資料載入器
    data_array = data_loader.load_single_csv(csv_path)
    
    # 4. 應用訓練時的標準化器 (不能重新訓練)
    scaled_data = scaler.transform(data_array)
    
    return scaled_data, feature_names
```

#### **資料格式要求**
```
CSV 結構:
Row 1: 欄位名稱 (A, B, C, D, E, ...)
Row 2: 單位資訊 (跳過)
Row 3+: 實際數據

特徵提取:
- 起始欄位: C (索引 2)
- 特徵數量: 35 個
- 目標特徵: POSFN, POS3DC.1, TCMD, SVPOS 等
```

### **2. 滑動窗口提取**

#### **窗口生成邏輯**
```python
def extract_windows(data_array, window_length=500, stride=50):
    T, num_features = data_array.shape
    windows = []
    start_indices = []
    
    for start in range(0, T - window_length + 1, stride):
        end = start + window_length
        window = data_array[start:end, :]  # (500, 35)
        
        # 轉換為 PyTorch Conv1D 格式 (35, 500)
        window_conv1d = window.transpose(1, 0)
        
        windows.append(window_conv1d)
        start_indices.append(start)
    
    return np.array(windows), start_indices
```

### **3. 特徵重要性計算**

#### **通道注意力計算**
```python
def compute_channel_attention(data_array, global_cam):
    # 計算重要性矩陣
    importance = np.abs(data_array) * global_cam[:, np.newaxis]  # (T, 35)
    
    # 每個通道的 Min-Max 歸一化
    channel_attention = np.zeros((num_features, T))
    
    for c in range(num_features):
        importance_c = importance[:, c]
        imp_min, imp_max = importance_c.min(), importance_c.max()
        
        if imp_max > imp_min:
            channel_attention[c, :] = (importance_c - imp_min) / (imp_max - imp_min)
        else:
            channel_attention[c, :] = np.zeros_like(importance_c)
    
    return channel_attention
```

---

## 🎨 **視覺化渲染引擎**

### **1. 雙軸圖表設計**

#### **主軸與副軸設定**
```python
def plot_line_with_attention(data_array, channel_attention, channel_idx, ...):
    # 建立圖表
    fig, ax1 = plt.subplots(1, 1, figsize=(16, 8))
    
    # 主軸: 信號數據 (藍色)
    ax1.plot(time_axis, data_array[:, channel_idx], 
            color='blue', linewidth=0.8, alpha=0.9,
            label=f'Signal ({feature_name})')
    
    # 建立副軸: 注意力分數 (紅色)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(time_axis, attention_values, 
                  color='red', linewidth=2.0, alpha=0.5,
                  label='Attention Score')
```

### **2. 熱力圖背景渲染**

#### **連續熱力圖實現**
```python
# 取得 Y 軸範圍
y_min, y_max = ax1.get_ylim()

# 重塑注意力數據為 imshow 格式
attention_heatmap = attention_values.reshape(1, -1)  # (1, T)

# 繪製背景熱力圖
im_bg = ax1.imshow(attention_heatmap, 
                  cmap='Reds',              # 紅色色譜
                  aspect='auto',            # 自動縱橫比
                  interpolation='bilinear', # 雙線性插值
                  alpha=0.6,               # 60% 不透明度
                  extent=[0, T, y_min, y_max],  # 範圍對齊
                  zorder=0)                # 背景層
```

### **3. 圖表美化與配置**

#### **專業化設定**
```python
# 軸標籤與顏色
ax1.set_ylabel('Signal Value', fontsize=12, color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1_twin.set_ylabel('Attention Score', fontsize=12, color='red')
ax1_twin.tick_params(axis='y', labelcolor='red')

# 網格與範圍
ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
ax1_twin.set_ylim(0, 1.0)

# 組合圖例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# 高品質輸出
plt.savefig(save_path, dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
```

---

## 🚀 **效能優化策略**

### **1. 記憶體管理**

#### **逐窗口處理**
```python
# 避免批次處理以節省記憶體
for k, (window, start) in enumerate(zip(windows, start_indices)):
    # 單窗口處理
    window_batch = window[np.newaxis, :]  # (1, 35, 500)
    cam_k = compute_grad_cam_for_window(model_instance, window_batch)
    
    # 立即累積，不保存中間結果
    for i in range(WINDOW_SIZE):
        global_idx = start + i
        if global_idx < total_length:
            global_cam[global_idx] += cam_k[i]
            count[global_idx] += 1
```

#### **Hook 清理機制**
```python
try:
    # Grad-CAM 計算
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)
    
    # ... 計算過程 ...
    
finally:
    # 確保 Hook 被清理
    forward_handle.remove()
    backward_handle.remove()
```

### **2. 計算效率優化**

#### **GPU 自動偵測**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_instance = model_instance.to(device)
window_tensor = window_tensor.to(device)
```

#### **數據壓縮策略**
```python
# 27,722 → 140 採樣點 (99.5% 壓縮)
sample_step = 200
compression_ratio = len(original_data) / len(sampled_data)
# 大幅減少渲染複雜度
```

### **3. 視覺化效能**

#### **柵格化處理**
```python
ax1.plot(..., rasterized=True)  # 向量轉柵格
im_bg = ax1.imshow(..., rasterized=True)  # 熱力圖柵格化
```

#### **輸出優化**
```python
plt.savefig(save_path, 
           dpi=300,              # 高解析度
           bbox_inches='tight',   # 緊湊邊界
           facecolor='white',     # 白色背景
           edgecolor='none')      # 無邊框
```

---

## 📐 **程式碼架構分析**

### **1. 函數職責分工**

```python
# 模型載入層
load_model()           # 載入 PyTorch 模型
load_scaler()          # 載入標準化器

# 資料處理層  
read_and_preprocess_csv()    # CSV 讀取與預處理
extract_windows()            # 滑動窗口提取

# 核心計算層
compute_grad_cam_for_window()     # 單窗口 Grad-CAM
aggregate_global_cam()            # 多窗口聚合
enhance_attention_regionally()    # 區域增強
downsample_and_interpolate()      # 降取樣與插值

# 分析計算層
compute_channel_attention()       # 通道注意力
compute_feature_importance()      # 特徵重要性

# 視覺化層
plot_line_with_attention()        # 主視覺化函數
print_top_features()              # 重要性排名輸出

# 主控制層
main()                           # 主程式流程控制
```

### **2. 資料流向分析**

```
CSV Input → StandardScaler → Sliding Windows
    ↓
Single Window Grad-CAM → Global Aggregation
    ↓  
Regional Enhancement → Downsampling & Interpolation
    ↓
Channel Attention → Feature Importance Ranking
    ↓
Visualization Rendering → File Output (PNG + NPZ)
```

### **3. 錯誤處理機制**

```python
try:
    # 主要計算流程
    model_instance = load_model(args.model_dir)
    # ...
except FileNotFoundError as e:
    print(f"❌ File not found: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error occurred: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
```

### **4. 參數驗證系統**

```python
# 特徵名稱驗證
if args.feature_name is not None:
    try:
        target_feature = feature_names.index(args.feature_name)
    except ValueError:
        print(f"⚠️ Feature name '{args.feature_name}' not found")
        # 自動回退到最重要特徵
        target_feature = np.argmax(importance_dict['max_scores'])

# 檔案存在性檢查
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Scaler not found: {scaler_path}")
```

---

## 🔧 **可自訂參數**

### **核心演算法參數**
```python
# Grad-CAM 參數
WINDOW_SIZE = 500      # 窗口大小
STEP_SIZE = 50         # 滑動步長
NUM_FEATURES = 35      # 特徵數量

# 增強參數
window_size = 500      # 區域增強窗口
enhancement = 0.3      # 增強強度
threshold = 0.05       # 增強閾值

# 取樣參數  
sample_step = 200      # 取樣間隔
local_window = 200     # 局部最大值窗口

# 視覺化參數
alpha_heatmap = 0.6    # 熱力圖透明度
alpha_line = 0.5       # 折線透明度
figsize = (16, 8)      # 圖表尺寸
dpi = 300             # 輸出解析度
```

---

## 📊 **效能基準測試**

### **處理時間 (Intel i7-8700K + GTX 1070)**
```
資料大小: 27,722 時間步 × 35 特徵
窗口數量: 545 個窗口

階段時間分布:
- 資料載入: ~2 秒
- 窗口提取: ~1 秒  
- Grad-CAM 計算: ~45 秒 (GPU) / ~180 秒 (CPU)
- 後處理與視覺化: ~3 秒
- 總計: ~51 秒 (GPU) / ~186 秒 (CPU)
```

### **記憶體使用量**
```
峰值記憶體: ~4GB (GPU) / ~2GB (RAM)
輸出檔案: ~3MB (PNG) + ~50KB (NPZ)
```

---

*技術文檔版本: 1.0.0 | 更新日期: 2025-11-20*