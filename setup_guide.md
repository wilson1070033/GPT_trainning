# GPT訓練程式設置指南

## 系統要求
- Python 3.8+
- CUDA 11.7+ (如果使用GPU)
- 至少16GB RAM (建議32GB+)
- 至少50GB可用硬盤空間

## 安裝步驟

### 1. 安裝PyTorch
```bash
# CPU版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# GPU版本 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. 安裝其他依賴項
```bash
pip install -r requirements.txt
```

### 3. 或者一次性安裝所有依賴
```bash
pip install torch>=2.0.0 transformers>=4.30.0 datasets>=2.12.0 peft>=0.4.0 accelerate>=0.20.0 bitsandbytes>=0.39.0 pandas>=1.5.0 numpy>=1.24.0 scikit-learn>=1.3.0 tqdm>=4.65.0 tensorboard>=2.13.0
```

## 運行程式
```bash
python3 GPT.py
```

## 注意事項
1. 首次運行會下載基礎模型(約13GB)
2. 訓練過程需要大量GPU記憶體
3. 建議在有GPU的環境下運行
4. 可以調整配置中的batch_size和max_length來適應硬體限制
