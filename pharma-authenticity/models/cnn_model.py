"""
CNN — Pharmaceutical Packaging Authenticity via Image Classification
Generates synthetic packaging images and trains a CNN to distinguish
genuine vs counterfeit based on visual packaging patterns
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Optional TensorFlow import ─────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[!] TensorFlow not installed. Run: pip install tensorflow")
    print("    CNN module loaded in DEMO mode (no training).")


IMG_SIZE   = 64   # 64×64 pixels
CHANNELS   = 3
N_CLASSES  = 2    # genuine, counterfeit


# ─────────────────────────────────────────────────────────────────────
#  Synthetic Image Generator
#  Real project: replace this with actual packaging photos
# ─────────────────────────────────────────────────────────────────────
def generate_packaging_images(n_per_class=300, img_size=IMG_SIZE, save_dir='images/packaging_samples'):
    """
    Generates synthetic packaging images:
      Genuine   → clean background, crisp text band, logo watermark
      Counterfeit → noisy background, blurry/misaligned text, faded colors
    """
    os.makedirs(os.path.join(save_dir, 'genuine'),     exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'counterfeit'), exist_ok=True)

    rng = np.random.default_rng(42)

    for i in range(n_per_class):
        # ── Genuine ────────────────────────────
        img = np.ones((img_size, img_size, 3), dtype=np.float32) * 0.96
        # Brand color band (top 15 px)
        img[:15, :] = [0.10, 0.35, 0.65]
        # Sharp text-like horizontal lines
        for row in [20, 25, 30, 35]:
            img[row, 8:img_size-8] = [0.1, 0.1, 0.1]
        # Logo circle
        cx, cy, r = img_size//2, img_size//2, 10
        Y, X = np.ogrid[:img_size, :img_size]
        mask = (X - cx)**2 + (Y - cy)**2 <= r**2
        img[mask] = [0.10, 0.35, 0.65]
        # Slight noise
        img += rng.normal(0, 0.01, img.shape).astype(np.float32)
        img = np.clip(img, 0, 1)
        plt.imsave(os.path.join(save_dir, 'genuine', f'genuine_{i:04d}.png'), img)

        # ── Counterfeit ────────────────────────
        img_f = rng.uniform(0.6, 1.0, (img_size, img_size, 3)).astype(np.float32)
        # Faded/wrong color band
        img_f[:15, :] = rng.uniform(0.4, 0.8, (15, img_size, 3)).astype(np.float32)
        # Blurry text simulation: thicker, noisy lines
        for row in [20, 25, 30, 35]:
            for dr in range(-2, 3):
                if 0 <= row + dr < img_size:
                    img_f[row+dr, 8:img_size-8] = rng.uniform(0.2, 0.5, (img_size-16, 3)).astype(np.float32)
        # Misaligned logo
        cx2 = img_size//2 + rng.integers(-8, 8)
        mask2 = (X - cx2)**2 + (Y - cy)**2 <= r**2
        img_f[mask2] = rng.uniform(0, 0.5, 3).astype(np.float32)
        img_f += rng.normal(0, 0.06, img_f.shape).astype(np.float32)
        img_f = np.clip(img_f, 0, 1)
        plt.imsave(os.path.join(save_dir, 'counterfeit', f'counterfeit_{i:04d}.png'), img_f)

    print(f"[✓] Generated {n_per_class * 2} images in {save_dir}/")
    return save_dir


# ─────────────────────────────────────────────────────────────────────
#  CNN Architecture
# ─────────────────────────────────────────────────────────────────────
def build_cnn(img_size=IMG_SIZE, channels=CHANNELS):
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required to build the CNN model.")

    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      input_shape=(img_size, img_size, channels)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # Classifier head
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')   # Binary output
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model


# ─────────────────────────────────────────────────────────────────────
#  Training Pipeline
# ─────────────────────────────────────────────────────────────────────
def train_cnn(data_dir='images/packaging_samples', img_size=IMG_SIZE,
              epochs=20, batch_size=32, save_path='models/cnn_model.keras'):

    if not TF_AVAILABLE:
        print("[!] TensorFlow not available — skipping CNN training.")
        return None, None

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        seed=42
    )

    val_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        seed=42
    )

    model = build_cnn(img_size)
    model.summary()

    cb_list = [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_auc', mode='max'),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        callbacks.ModelCheckpoint(save_path, save_best_only=True, monitor='val_auc', mode='max')
    ]

    print(f"\n[*] Training CNN for up to {epochs} epochs...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=cb_list,
        verbose=1
    )

    plot_cnn_history(history)
    print(f"[✓] CNN model saved → {save_path}")
    return model, history


def plot_cnn_history(history, save_dir='evaluation'):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#0d1117')
    for ax in axes:
        ax.set_facecolor('#161b22')

    epochs = range(1, len(history.history['accuracy']) + 1)

    axes[0].plot(epochs, history.history['accuracy'],     color='#58a6ff', label='Train Acc')
    axes[0].plot(epochs, history.history['val_accuracy'], color='#3fb950', label='Val Acc')
    axes[0].set_title('Accuracy', color='white')
    axes[0].set_xlabel('Epoch', color='white')
    axes[0].tick_params(colors='white')
    axes[0].legend(facecolor='#161b22', labelcolor='white')

    axes[1].plot(epochs, history.history['loss'],     color='#f85149', label='Train Loss')
    axes[1].plot(epochs, history.history['val_loss'], color='#e3b341', label='Val Loss')
    axes[1].set_title('Loss', color='white')
    axes[1].set_xlabel('Epoch', color='white')
    axes[1].tick_params(colors='white')
    axes[1].legend(facecolor='#161b22', labelcolor='white')

    plt.suptitle('CNN Training History — Packaging Authenticity', color='white', fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, 'cnn_training_history.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"    [✓] Plot saved → {path}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    data_dir = generate_packaging_images(n_per_class=300)
    model, history = train_cnn(data_dir=data_dir, epochs=20)

    if model:
        print("\n[✓] CNN pipeline complete.")
    else:
        print("\n[!] Install TensorFlow to run CNN training:")
        print("    pip install tensorflow")
