# 2025全國智慧製造大數據分析競賽

## 決賽競賽環境登入說明

---

## 一、決賽平台的架構

* 國網中心及台智雲公司提供給各決賽團隊的 1 顆 32GB 的 GPU、記憶體是 64GB
* 每隊的 GPU *1，型號為 NVIDIA Tesla V100 32GB GPU，詳細介紹可參考：
  [https://www.nvidia.com/zh-tw/data-center/v100/](https://www.nvidia.com/zh-tw/data-center/v100/)
* 登入節點主機：203.145.216.211 port:22，連線來源只限台灣 IP
* 資料上傳節點：203.145.216.211 port:21，連線來源只限台灣 IP

---

## 二、SSH 連線

使用 SSH 指令或 SSH 工具（如 PuTTY）連線到競賽環境之登入節點主機（203.145.216.211 port:22），並進入各參賽團隊所分配之競賽主機，以下範例為使用 PuTTY 的連線步驟。

1. 連線至 [PuTTY 官方網站](https://putty.software) 下載。
2. 點選紅框處下載檔案（版本請依需求自行選擇）。
3. 執行下載的 putty.exe 檔案，在 Host Name(or IP address) 輸入 `203.145.216.211`、Port 輸入 `22` 後，點選 Open 再點 Accept。
4. 連線到登入節點後，請輸入競賽分配之帳號與密碼，並依提示輸入：

   ```bash
   ssh c1teamXX
   ```

   即可登入競賽主機。

> 注意：
>
> 1. 第一次登入會出現更改密碼的提示，請依指示修改密碼並妥善保管。
> 2. 登入後依畫面提示輸入 `ssh` 指令即可進入隊伍環境。

登入後可使用以下指令：

```bash
ls -l /
```

目錄說明：

* `/TOPIC/projectA`：決賽題目 A（唯讀） → 使用 `cd /TOPIC/projectA`
* `/TOPIC/projectB`：決賽題目 B（唯讀） → 使用 `cd /TOPIC/projectB`
* `/FTP`：FTP 上傳的資料（唯讀）
* `/home/114XXX`：個人資料夾（200GB，可讀寫）

---

## 三、FTP 連線

使用 FTP 指令或 FTP 工具（如 FileZilla）連線到 203.145.216.211 port:21，登入後即可上傳檔案。

> 注意：FTP 節點主機僅能上傳及刪除檔案，無法下載檔案。

### FileZilla 連線方式

1. 前往 [https://filezilla-project.org](https://filezilla-project.org) 下載 FileZilla Client。
2. **快速連線模式：**

   * 主機：203.145.216.211
   * 連接埠：21
   * 帳號：競賽帳號
   * 密碼：競賽密碼
   * 點選「快速連線」。
3. **站台管理員模式：**

   * 協定選擇 FTP - 檔案傳輸協定
   * 輸入主機、連接埠、帳號、密碼後點選「連線」。

操作說明：

* 上傳：右鍵 → 上傳
* 刪除：右鍵 → 刪除
* 在主機中查看上傳結果：

  ```bash
  ls /FTP
  ```
* 若要將上傳資料移至個人資料夾：

  ```bash
  cp /FTP/檔案名稱 /home/114XXX
  ```

---

## 四、其他指令說明

### 查 GPU 資源

```bash
nvidia-smi
```

### 更換密碼

```bash
passwd
```

輸入原密碼、新密碼、再輸入新密碼確認。

---

## 五、決賽環境版本資訊

* 作業系統：Linux Ubuntu 22.04.5 LTS Server 版
* 工具軟體：

  * CUDA 12.2
  * cuDNN 9

---

## 六、各種執行環境的使用方式

### 1. 使用 PyTorch 環境

進入方式：

```bash
cd /envs/pytorch
pipenv shell
```

已安裝的套件如下（完整清單）：

```
catboost 1.2.8
certifi 2024.2.2
charset-normalizer 3.3.2
contourpy 1.3.2
cycler 0.12.1
deap 1.4.1
filelock 3.18.0
fonttools 4.51.0
fsspec 2025.5.1
future 1.0.0
graphviz 0.21
hf-xet 1.1.5
huggingface-hub 0.33.2
idna 3.7
imageio 2.34.1
Jinja2 3.1.6
joblib 1.4.2
kiwisolver 1.4.5
lazy_loader 0.4
lightgbm 4.3.0
MarkupSafe 3.0.2
matplotlib 3.8.4
mpmath 1.3.0
narwhals 1.45.0
networkx 3.3
numpy 1.26.4
nvidia-cublas-cu12 12.1.3.1
nvidia-cuda-cupti-cu12 12.1.105
nvidia-cuda-nvrtc-cu12 12.1.105
nvidia-cuda-runtime-cu12 12.1.105
nvidia-cudnn-cu12 8.9.2.26
nvidia-cufft-cu12 11.0.2.54
nvidia-curand-cu12 10.3.2.106
nvidia-cusolver-cu12 11.4.5.107
nvidia-cusparse-cu12 12.1.0.106
nvidia-nccl-cu12 2.20.5
nvidia-nvjitlink-cu12 12.9.86
nvidia-nvtx-cu12 12.1.105
opencv-python 4.11.0.86
opencv-python-headless 4.9.0.80
packaging 24.0
pandas 2.2.2
pickleshare 0.7.5
pillow 10.3.0
pip 25.1.1
plotly 6.2.0
psutil 7.0.0
py-cpuinfo 9.0.0
pyparsing 3.1.2
python-dateutil 2.9.0.post0
pytz 2024.1
PyWavelets 1.6.0
PyYAML 6.0.2
requests 2.31.0
safetensors 0.5.3
scikit-image 0.23.2
scikit-learn 1.4.2
scipy 1.13.0
seaborn 0.13.2
setuptools 80.3.1
six 1.16.0
stopit 1.1.2
sympy 1.14.0
thop 0.1.1.post2209072238
threadpoolctl 3.5.0
tifffile 2024.5.3
timm 1.0.16
torch 2.3.0
torchvision 0.18.0
TPOT 0.12.2
tqdm 4.66.4
triton 2.3.0
typing_extensions 4.11.0
tzdata 2025.2
ultralytics 8.2.8
update-checker 0.18.0
urllib3 2.2.1
wheel 0.45.1
xgboost 2.0.3
xlrd 2.0.1
xlutils 2.0.0
xlwt 1.3.0
```

---

### 2. 使用 TensorFlow 與 Keras 環境

進入方式：

```bash
cd /envs/tf_keras
pipenv shell
```

完整套件清單如下：

```
absl-py 2.1.0
astor 0.8.1
astunparse 1.6.3
cachetools 5.3.3
catboost 1.2.8
certify 2024.2.2
chardet 5.2.0
charset-normalizer 3.3.2
contourpy 1.3.2
cycler 0.12.1
deap 1.4.1
filelock 3.18.0
flatbuffers 24.3.25
fonttools 4.51.0
fsspec 2025.5.1
gast 0.5.4
google-auth 2.29.0
google-auth-oauthlib 1.2.0
google-pasta 0.2.0
graphviz 0.21
grpcio 1.63.0
h5py 3.11.0
hf-xet 1.1.5
huggingface-hub 0.33.2
idna 3.7
imageio 2.34.1
importlib_metadata 7.1.0
Jinja2 3.1.6
joblib 1.4.2
keras 2.15.0
Keras-Applications 1.0.8
Keras-Preprocessing 1.1.2
Kiwisolver 1.4.5
lazy_loader 0.4
libclang 18.1.1
lightgbm 4.3.0
Markdown 3.6
MarkupSafe 2.1.5
matplotlib 3.8.4
ml-dtypes 0.2.0
mpmath 1.3.0
narwhals 1.45.0
network 3.3
numpy 1.26.4
nvidia-cublas-cu12 12.6.4.1
nvidia-cuda-cupti-cu12 12.6.80
nvidia-cuda-nvrtc-cu12 12.6.77
nvidia-cuda-runtime-cu12 12.6.77
nvidia-cudnn-cu12 9.5.1.17
nvidia-cufft-cu12 11.3.0.4
nvidia-cufile-cu12 1.11.1.6
nvidia-curand-cu12 10.3.7.77
nvidia-cusolver-cu12 11.7.1.2
nvidia-cusparse-cu12 12.5.4.2
nvidia-cusparselt-cu12 0.6.3
nvidia-nccl-cu12 2.26.2
nvidia-nvjitlink-cu12 12.6.85
nvidia-nvtx-cu12 12.6.77
oauthlib 3.2.2
opencv-python 4.11.0.86
opencv-python-headless 4.9.0.80
opt-einsum 3.3.0
packaging 24.0
pandas 2.2.2
pickleshare 0.7.5
pillow 10.3.0
pip 25.1.1
plotly 6.2.0
protobuf 4.23.4
psutil 7.0.0
py-cpuinfo 9.0.0
pyasn1 0.6.0
pyasn1_modules 0.4.0
pyparsing 3.2.3
python-dateutil 2.9.0.post0
pytz 2024.1
PyWavelets 1.6.0
PyYAML 6.0.1
requests 2.31.0
requests-oauthlib 2.0.0
rsa 4.9
safetensors 0.5.3
scikit-image 0.23.2
scikit-learn 1.4.2
scipy 1.13.0
seaborn 0.13.2
setuptools 80.3.1
six 1.16.0
stopit 1.1.2
sympy 1.14.0
tensorboard 2.15.0
tensorboard-data-server 0.7.2
tensorboard-plugin-wit 1.8.1
tensorflow 2.15.0
tensorflow-estimator 2.15.0
tensorflow-io-gcs-filesystem 0.37.0
tensorrt-bindings 8.6.1
tensorrt-libs 8.6.1
termcolor 2.4.0
thop 0.1.1.post2209072238
threadpoolctl 3.5.0
tifffile 2024.5.3
timm 1.0.16
torch 2.7.1
torchvision 0.22.1
TPOT 0.12.2
tqdm 4.66.2
triton 3.3.1
typing_extensions 4.11.0
tzdata 2025.2
ultralytics 8.2.8
update-checker 0.18.0
urllib3 2.2.1
Werkzeug 3.0.2
wheel 0.43.0
wrapt 1.14.1
xgboost 2.0.3
xlrd 2.0.1
xlutils 2.0.0
xlwt 1.3.0
zipp 3.18.1
torchvision 0.15.2
triton 2.0.0
tzdata 2024.1
```

---

### 3. 使用 JAVA 環境

* 版本：1.8.0_451
* JAVA 環境已設定完成，可直接使用。
* 查詢版本：

  ```bash
  java -version
  ```

---

### 4. 使用 R 環境

* 版本：4.5.1
* 無環境隔離，可直接使用。
* 查詢版本：

  ```bash
  R
  ```
