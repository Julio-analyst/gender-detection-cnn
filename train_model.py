"""
Training Script - LSTM Gender Voice Detection
Load data yang sudah ada dan train model
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# PATHS
# ============================================================================
DATA_DIR = "data/processed"
MODEL_DIR = "models"

X_PATH = os.path.join(DATA_DIR, "features_latest.npy")
Y_PATH = os.path.join(DATA_DIR, "labels_latest.npy")
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_production.h5")

# ============================================================================
# PARAMETERS (Sesuai Notebook)
# ============================================================================
EPOCHS = 50
BATCH_SIZE = 16
TEST_SIZE = 0.2
VAL_SPLIT = 0.2
RANDOM_SEED = 42

print("="*80)
print("🎤 LSTM Gender Voice Detection - Training Script")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n📂 Loading dataset...")

if not os.path.exists(X_PATH):
    print(f"❌ Feature file not found: {X_PATH}")
    exit(1)

if not os.path.exists(Y_PATH):
    print(f"❌ Label file not found: {Y_PATH}")
    exit(1)

X = np.load(X_PATH, allow_pickle=True)
y = np.load(Y_PATH)

print(f"✅ Dataset loaded successfully!")
print(f"   Total samples: {len(X)}")
print(f"   Laki-laki (0): {np.sum(y == 0)}")
print(f"   Perempuan (1): {np.sum(y == 1)}")

# ============================================================================
# DATA PREPROCESSING
# ============================================================================
print("\n🔄 Preprocessing data...")

# Pad MFCC sequences
print("   Padding sequences...")
X_padded = pad_sequences(X, dtype='float32', padding='post')

print(f"   Padded shape: {X_padded.shape}")

# Train-test split
print(f"   Splitting data (test size: {TEST_SIZE})...")
X_train, X_test, y_train, y_test = train_test_split(
    X_padded, 
    y, 
    test_size=TEST_SIZE, 
    random_state=RANDOM_SEED, 
    stratify=y
)

print(f"   Train shape: {X_train.shape}")
print(f"   Test shape: {X_test.shape}")

# ============================================================================
# CREATE MODEL
# ============================================================================
print("\n🏗️  Building LSTM model...")

model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
], name='LSTM_Gender_Classifier')

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("✅ Model created successfully!")
print("\n" + "="*80)
model.summary()
print("="*80)

# ============================================================================
# TRAIN MODEL
# ============================================================================
print(f"\n🚀 Starting training...")
print(f"   Epochs: {EPOCHS}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Validation split: {VAL_SPLIT}")
print()

history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VAL_SPLIT,
    verbose=1
)

print("\n✅ Training completed!")

# ============================================================================
# EVALUATE MODEL
# ============================================================================
print("\n📊 Evaluating model on test set...")

y_pred_proba = model.predict(X_test, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Calculate metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "="*80)
print("EVALUATION RESULTS")
print("="*80)
print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("="*80)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
labels = ["Laki-laki", "Perempuan"]
print(classification_report(y_test, y_pred, target_names=labels))

# ============================================================================
# SAVE MODEL
# ============================================================================
print(f"\n💾 Saving model to: {MODEL_PATH}")

os.makedirs(MODEL_DIR, exist_ok=True)
model.save(MODEL_PATH)

print("✅ Model saved successfully!")

# ============================================================================
# PLOT TRAINING HISTORY
# ============================================================================
print("\n📈 Generating training history plots...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy
ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Loss
ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(MODEL_DIR, "training_history.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"✅ Training history saved to: {plot_path}")

# Confusion Matrix Plot
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=labels,
    yticklabels=labels,
    ax=ax
)
ax.set_title('Confusion Matrix - LSTM Model', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')

plt.tight_layout()
cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
print(f"✅ Confusion matrix saved to: {cm_path}")

print("\n" + "="*80)
print("🎉 TRAINING COMPLETE!")
print("="*80)
print(f"\n✅ Model ready to use: {MODEL_PATH}")
print(f"📊 Accuracy: {acc*100:.2f}%")
print(f"\n💡 Next step: Run Streamlit app untuk prediksi!")
print(f"   Command: streamlit run app_streamlit.py")
print("="*80)
