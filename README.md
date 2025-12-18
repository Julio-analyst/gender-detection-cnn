# 🎤 Gender Voice Detection - Streamlit App

Deteksi gender (Laki-laki/Perempuan) menggunakan Deep Learning - CNN LSTM Model

## 🚀 Deployment ke Streamlit Cloud

### Persiapan Repository
Pastikan file-file berikut ada di root repository Anda:

```
├── app.py                    # Main Streamlit app
├── requirements.txt          # Python dependencies
├── packages.txt             # System packages
├── runtime.txt              # Python version
├── models/
│   └── lstm_production.h5   # Trained model
├── .streamlit/
│   └── config.toml          # Streamlit config
└── README.md                # This file
```

### Langkah Deploy

1. **Push ke GitHub**
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Deploy ke Streamlit Cloud**
   - Buka [share.streamlit.io](https://share.streamlit.io)
   - Connect ke GitHub repository Anda
   - Pilih branch `main`
   - Set main file path: `app.py`
   - Klik Deploy!

### Dependencies yang Diperbaiki

- **TensorFlow 2.15.0**: Compatible dengan Streamlit 1.29.0
- **NumPy 1.24.3**: Compatible dengan TensorFlow 2.15.0
- **Python 3.11**: Specified di runtime.txt

### Troubleshooting

Jika deploy gagal:

1. **Protobuf Conflict**: Sudah diperbaiki dengan TensorFlow 2.15.0
2. **Model Not Found**: Pastikan `models/lstm_production.h5` ada
3. **Audio Processing**: `packages.txt` sudah include ffmpeg dan libsndfile1

### Features

- 🎙️ **Rekam Suara**: Real-time recording via microphone
- 📁 **Upload File**: Support WAV, MP3, M4A, FLAC
- 📊 **Visualisasi**: Waveform dan MFCC features
- 🎯 **Prediksi**: CNN LSTM model untuk akurasi tinggi
- 📈 **Confidence Score**: Breakdown probabilitas gender

### Model Info

- **Architecture**: CNN + LSTM
- **Input**: MFCC features (13 coefficients)
- **Sample Rate**: 16kHz
- **Framework**: TensorFlow/Keras

---

Made with ❤️ using Streamlit & TensorFlow
