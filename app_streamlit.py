
# 🎤 Gender Voice Detection - Streamlit App
# Prediksi Gender dari Suara menggunakan Logistic Regression (tanpa TensorFlow)


import os
import tempfile
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st
import noisereduce as nr
from audio_recorder_streamlit import audio_recorder
from sklearn.linear_model import LogisticRegression
import joblib

# ============================================================================
# CONFIG
# ============================================================================
st.set_page_config(
    page_title="🎤 Prediksi Gender Suara",
    page_icon="🎤",
    layout="wide",
)

# ============================================================================
# CONSTANTS
# ============================================================================
SR = 16000
N_MFCC = 13
N_FFT = 2048
HOP = 512
TARGET_DB = -20.0
MODEL_PATH = "models/sklearn_logreg_gender.joblib"

# ============================================================================
# FUNCTIONS
# ============================================================================

@st.cache_resource
def load_sklearn_model():
    """Load or train a simple sklearn model (Logistic Regression)"""
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            return model
        else:
            st.info("🔄 Model not found. Training simple Logistic Regression model...")
            X_PATH = "data/processed/features_latest.npy"
            Y_PATH = "data/processed/labels_latest.npy"
            if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
                st.error("❌ Training data not found. Cannot create model.")
                return None
            X = np.load(X_PATH, allow_pickle=True)
            y = np.load(Y_PATH)
            # Feature engineering: use mean and std of MFCCs as features
            X_feat = np.array([[np.mean(x), np.std(x)] for x in X])
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X_feat, y, test_size=0.2, random_state=42, stratify=y)
            model = LogisticRegression()
            model.fit(X_train, y_train)
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            joblib.dump(model, MODEL_PATH)
            st.success("✅ Model trained and saved!")
            return model
    except Exception as e:
        st.error(f"❌ Error loading/training model: {e}")
        return None


def rms_db(y):
    """Calculate RMS in dB"""
    rms = np.sqrt(np.mean(y**2))
    return 20 * np.log10(rms + 1e-9)


def normalize_rms(y, target_db=-20.0):
    """Normalize audio RMS"""
    cur = rms_db(y)
    gain = target_db - cur
    y_norm = y * (10 ** (gain / 20))
    
    peak = np.max(np.abs(y_norm))
    if peak > 0.999:
        y_norm = y_norm / peak * 0.999
    
    return y_norm


def process_audio(audio, sr):
    """Preprocess audio"""
    # Noise reduction
    reduced = nr.reduce_noise(y=audio, sr=sr)
    
    # High-pass filter
    filtered = librosa.effects.preemphasis(reduced)
    
    # RMS normalization
    normalized = normalize_rms(filtered, target_db=TARGET_DB)
    
    return normalized


def extract_mfcc(audio, sr):
    """Extract MFCC features"""
    mfcc = librosa.feature.mfcc(
        y=audio, 
        sr=sr, 
        n_mfcc=N_MFCC, 
        n_fft=N_FFT, 
        hop_length=HOP
    )
    return mfcc.T


def predict_gender(model, audio, sr):
    """Predict gender from audio"""
    # Process audio
    processed = process_audio(audio, sr)
    mfcc = extract_mfcc(processed, sr)
    # Feature: mean and std of MFCCs
    feat = np.array([[np.mean(mfcc), np.std(mfcc)]])
    pred_prob = model.predict_proba(feat)[0][1]
    if pred_prob > 0.5:
        label = "Perempuan"
        confidence = pred_prob
    else:
        label = "Laki-laki"
        confidence = 1 - pred_prob
    return label, confidence, mfcc, processed


def plot_waveform(audio, sr):
    """Plot waveform"""
    fig, ax = plt.subplots(figsize=(10, 3))
    time = np.arange(len(audio)) / sr
    ax.plot(time, audio, linewidth=0.8, color='#1f77b4')
    ax.set_title("Audio Waveform", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_mfcc(mfcc):
    """Plot MFCC"""
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        mfcc.T, 
        x_axis='time', 
        ax=ax, 
        cmap='viridis',
        hop_length=HOP,
        sr=SR
    )
    ax.set_title("MFCC Features", fontsize=14, fontweight='bold')
    ax.set_ylabel("MFCC Coefficients")
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN APP
# ============================================================================

# Header
st.title("🎤 Prediksi Gender dari Suara")
st.markdown("**Deteksi gender (Laki-laki/Perempuan) menggunakan Deep Learning - LSTM Model**")
st.markdown("---")

# Load model
with st.spinner("⏳ Loading model..."):
    model = load_sklearn_model()

if model is None:
    st.error(f"❌ Model tidak ditemukan di: `{MODEL_PATH}`")
    st.info("💡 Pastikan data/processed/features_latest.npy dan labels_latest.npy tersedia.")
    st.stop()

st.success("✅ Model berhasil dimuat!")

# Sidebar
with st.sidebar:
    st.header("ℹ️ Informasi Model")
    st.markdown(f"""
    **Model:** LSTM (64 units)  
    **Sample Rate:** {SR} Hz  
    **MFCC Coefficients:** {N_MFCC}  
    **FFT Size:** {N_FFT}  
    **Hop Length:** {HOP}  
    **RMS Target:** {TARGET_DB} dB
    
    ---
    
    ### 📝 Cara Pakai:
    
    1. **Pilih metode input:**
       - 🎙️ Rekam suara dari mic
       - 📁 Upload file audio
    
    2. **Klik tombol Prediksi**
    
    3. **Lihat hasil prediksi!**
    
    ---
    
    ### 📂 Format Supported:
    - WAV (.wav)
    - MP3 (.mp3)
    - M4A (.m4a)
    - FLAC (.flac)
    """)

st.markdown("---")

# Input method selection
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎙️ Rekam Suara dari Microphone")
    
    st.info("👇 Klik tombol di bawah untuk mulai merekam")
    
    # Audio recorder
    audio_bytes = audio_recorder(
        text="Klik untuk Rekam",
        recording_color="#e74c3c",
        neutral_color="#6aa84f",
        icon_name="microphone",
        icon_size="3x",
    )
    
    if audio_bytes:
        st.success("✅ Rekaman berhasil!")
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path_mic = tmp_file.name
        
        # Display audio player
        st.audio(audio_bytes, format="audio/wav")
        
        # Store in session state
        st.session_state['audio_source'] = 'mic'
        st.session_state['audio_path'] = tmp_path_mic

with col2:
    st.subheader("📁 Upload File Audio")
    
    uploaded_file = st.file_uploader(
        "Pilih file audio",
        type=['wav', 'mp3', 'm4a', 'flac'],
        help="Upload file audio dalam format WAV, MP3, M4A, atau FLAC"
    )
    
    if uploaded_file:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path_upload = tmp_file.name
        
        # Display audio player
        st.audio(uploaded_file, format=f"audio/{os.path.splitext(uploaded_file.name)[1][1:]}")
        
        # Store in session state
        st.session_state['audio_source'] = 'upload'
        st.session_state['audio_path'] = tmp_path_upload

st.markdown("---")

# Predict button
if 'audio_path' in st.session_state:
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        predict_btn = st.button(
            "🎯 PREDIKSI GENDER",
            type="primary",
            use_container_width=True
        )
    
    if predict_btn:
        audio_path = st.session_state['audio_path']
        
        with st.spinner("🔄 Processing audio..."):
            # Load audio
            audio, sr_loaded = librosa.load(audio_path, sr=SR, mono=True)
            
            # Predict
            label, confidence, mfcc, processed = predict_gender(model, audio, SR)
        
        st.markdown("---")
        
        # Results
        st.subheader("🎉 Hasil Prediksi")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if label == "Laki-laki":
                st.markdown(
                    f"""
                    <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.3);'>
                        <h1 style='color: white; font-size: 3em; margin: 0;'>👨</h1>
                        <h2 style='color: white; margin: 10px 0;'>{label}</h2>
                        <p style='color: #e0e0e0; font-size: 1.2em; margin: 0;'>Confidence: {confidence*100:.2f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.3);'>
                        <h1 style='color: white; font-size: 3em; margin: 0;'>👩</h1>
                        <h2 style='color: white; margin: 10px 0;'>{label}</h2>
                        <p style='color: #e0e0e0; font-size: 1.2em; margin: 0;'>Confidence: {confidence*100:.2f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        st.markdown("---")
        
        # Confidence breakdown
        st.subheader("📊 Confidence Breakdown")
        
        prob_male = (1 - confidence) * 100 if label == "Perempuan" else confidence * 100
        prob_female = confidence * 100 if label == "Perempuan" else (1 - confidence) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("👨 Laki-laki", f"{prob_male:.2f}%")
            st.progress(prob_male / 100)
        
        with col2:
            st.metric("👩 Perempuan", f"{prob_female:.2f}%")
            st.progress(prob_female / 100)
        
        st.markdown("---")
        
        # Visualizations
        with st.expander("📈 Visualisasi Audio & Features", expanded=True):
            
            tab1, tab2 = st.tabs(["🌊 Waveform", "🎵 MFCC Features"])
            
            with tab1:
                st.markdown("### Audio Waveform (Processed)")
                fig_wave = plot_waveform(processed, SR)
                st.pyplot(fig_wave)
                plt.close()
            
            with tab2:
                st.markdown("### MFCC Features")
                fig_mfcc = plot_mfcc(mfcc)
                st.pyplot(fig_mfcc)
                plt.close()
                
                st.info(f"📐 MFCC Shape: {mfcc.shape} (time_steps × coefficients)")
        
        # Audio info
        with st.expander("ℹ️ Audio Information"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Duration", f"{len(audio)/SR:.2f} s")
            
            with col2:
                st.metric("Sample Rate", f"{SR} Hz")
            
            with col3:
                st.metric("RMS (dB)", f"{rms_db(processed):.2f}")

else:
    st.info("👆 Silakan rekam suara atau upload file audio terlebih dahulu")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888;'>
        <p>🎤 <b>Gender Voice Detection</b> | LSTM Model | Made with ❤️ using Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
