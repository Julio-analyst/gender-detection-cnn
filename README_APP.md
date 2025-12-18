# 🎤 Gender Voice Detection - Streamlit App

Aplikasi **prediksi gender dari suara** menggunakan model LSTM.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_app.txt
```

### 2. Train Model (Jika belum ada)

```bash
python train_model.py
```

Script ini akan:
- Load data dari `data/processed/`
- Train model LSTM
- Save model ke `models/lstm_production.h5`
- Generate plots training history & confusion matrix

### 3. Run Streamlit App

```bash
streamlit run app_streamlit.py
```

Aplikasi akan terbuka di browser: `http://localhost:8501`

## 🎯 Fitur Aplikasi

### 1. 🎙️ Rekam Suara dari Microphone
- Klik tombol rekam
- Bicara selama 3 detik
- Auto-save recording

### 2. 📁 Upload File Audio
- Support format: WAV, MP3, M4A, FLAC
- Drag & drop atau browse file

### 3. 🎯 Prediksi Real-time
- Klik tombol "PREDIKSI GENDER"
- Hasil muncul dengan confidence score
- Visualisasi waveform & MFCC

## 📊 Data & Model

### Dataset Location
```
data/
├── processed/
│   ├── features_latest.npy    # MFCC features
│   └── labels_latest.npy      # Labels (0=male, 1=female)
```

### Model
- **Architecture**: LSTM(64) → Dropout(0.2) → Dense(1, sigmoid)
- **Location**: `models/lstm_production.h5`
- **Target Accuracy**: >95%

## 🔧 Parameters

| Parameter | Value |
|-----------|-------|
| Sample Rate | 16000 Hz |
| MFCC Coefficients | 13 |
| FFT Size | 2048 |
| Hop Length | 512 |
| RMS Target | -20 dB |
| Epochs | 50 |
| Batch Size | 16 |

## 📝 Workflow

1. **Audio Input** (Rekam/Upload)
2. **Preprocessing**:
   - Load audio → 16kHz mono
   - Noise reduction
   - High-pass filter (preemphasis)
   - RMS normalization (-20 dB)
3. **Feature Extraction**:
   - Extract 13 MFCC coefficients
4. **Prediction**:
   - Pad sequences
   - LSTM inference
   - Output: Gender + Confidence

## 🐛 Troubleshooting

### Error: Model not found
```bash
# Train model dulu
python train_model.py
```

### Error: Module not found
```bash
# Install dependencies
pip install -r requirements_app.txt
```

### Error: Can't record audio
- Pastikan browser support audio recording (Chrome/Edge recommended)
- Allow microphone permission

## 📁 File Structure

```
mlops - Copy/
├── app_streamlit.py           # Main Streamlit app
├── train_model.py            # Training script
├── requirements_app.txt      # Dependencies
├── README_APP.md            # This file
├── data/
│   └── processed/
│       ├── features_latest.npy
│       └── labels_latest.npy
└── models/
    └── lstm_production.h5   # Trained model
```

## 🎯 Tips

1. **Audio Quality**: Rekam di tempat sepi untuk hasil terbaik
2. **Duration**: Durasi optimal 3 detik
3. **Format**: WAV memberikan kualitas terbaik

## 📱 Cara Pakai

1. Buka app: `streamlit run app_streamlit.py`
2. Pilih metode input:
   - 🎙️ Klik tombol rekam → bicara
   - 📁 Upload file audio
3. Klik **"PREDIKSI GENDER"**
4. Lihat hasil:
   - Predicted gender (👨 Laki-laki / 👩 Perempuan)
   - Confidence score
   - Visualisasi waveform & MFCC

## 🎉 Result Display

- **Gender Icon**: 👨 untuk Laki-laki, 👩 untuk Perempuan
- **Confidence**: Persentase keyakinan model
- **Breakdown**: Probabilitas masing-masing gender
- **Visualisasi**: Waveform & MFCC features

---

**Happy Predicting! 🎤**
