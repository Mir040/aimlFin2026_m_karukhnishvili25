# Convolutional Neural Network (CNN) Report

## Task 1 -- Cybersecurity Application

------------------------------------------------------------------------

## 📌 Introduction to Convolutional Neural Networks

A **Convolutional Neural Network (CNN)** is a specialized type of deep
learning model designed primarily for processing structured grid-like
data such as images. CNNs are widely used in computer vision, medical
imaging, autonomous driving, and cybersecurity. Unlike traditional
neural networks, CNNs automatically extract important features from raw
input data without requiring manual feature engineering.

CNNs consist of several layers including convolutional layers, pooling
layers, and fully connected layers. These layers work together to detect
hierarchical patterns within the data.

------------------------------------------------------------------------

## 🧠 CNN Architecture Overview

### 1. Convolutional Layer

The convolutional layer applies filters (kernels) across the input data.
Each filter extracts specific features such as edges, textures, or
shapes.

Mathematically:

    Output = Input * Kernel + Bias

These filters slide across the input matrix to generate feature maps.
Multiple filters allow the network to learn multiple features
simultaneously.

### 2. Activation Function

Activation functions such as **ReLU (Rectified Linear Unit)** introduce
non-linearity into the model, enabling it to learn complex patterns.

    ReLU(x) = max(0, x)

### 3. Pooling Layer

Pooling layers reduce the spatial size of feature maps. This reduces
computational cost and helps prevent overfitting. Common pooling types
include:

-   Max Pooling
-   Average Pooling

### 4. Fully Connected Layer

After feature extraction, flattened feature maps are passed into fully
connected layers for classification or regression tasks.

------------------------------------------------------------------------

## 📊 CNN Workflow Visualization

    Input Image → Convolution → Activation → Pooling → Flatten → Dense Layers → Output

------------------------------------------------------------------------

## 🔐 CNN Application in Cybersecurity

CNNs can detect malware by analyzing binary files. Instead of reading
files as text, they are converted into grayscale images. Patterns inside
these images represent malware signatures.

### Dataset Used

-   **Benign Files:** Random ASCII text samples
-   **Malicious Files:** EICAR antivirus test string and variations

------------------------------------------------------------------------

## 📁 Malware Visualization Example

Binary malware is converted into pixel intensities (0--255) and resized
to 64×64 grayscale images. These images are used as CNN input.

------------------------------------------------------------------------

## 💻 Python Implementation

### Dataset and Model Training Code

``` python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import hashlib
import io
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

# ============================================
# EICAR Test File Data (attached file content)
# ============================================
EICAR_STRING = "X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"


def binary_to_image(binary_data, width=64):
    """Convert binary data to grayscale image matrix"""
    # Convert each byte to pixel value (0-255)
    pixels = []
    for char in binary_data:
        pixels.append(ord(char) % 256)

    # Calculate height
    height = len(pixels) // width + (1 if len(pixels) % width else 0)

    # Pad if necessary
    padded_length = height * width
    pixels.extend([0] * (padded_length - len(pixels)))

    # Reshape to image
    img_array = np.array(pixels).reshape(height, width)
    return img_array, height, width


def generate_dataset():
    """Generate synthetic dataset for demonstration"""
    # Benign files (random ASCII text)
    benign_samples = []
    for _ in range(100):
        # Generate random printable characters
        random_bytes = ''.join(chr(np.random.randint(32, 127))
                               for _ in range(np.random.randint(68, 200)))
        benign_samples.append(random_bytes)

    # Malicious files (EICAR + variations)
    malicious_samples = []
    for i in range(100):
        # Vary the EICAR string slightly
        if i < 50:
            malicious_samples.append(EICAR_STRING)
        else:
            # Add some noise but keep malicious signature
            noise = ''.join(chr(np.random.randint(65, 90))
                            for _ in range(np.random.randint(1, 10)))
            malicious_samples.append(EICAR_STRING + noise)

    return benign_samples, malicious_samples


def prepare_data(samples, labels):
    """Convert samples to CNN-ready format"""
    X = []
    y = []

    for sample, label in zip(samples, labels):
        img, h, w = binary_to_image(sample)
        # Resize to fixed dimensions (64x64) using PIL
        img_pil = Image.fromarray(img.astype('uint8'))
        img_resized = np.array(img_pil.resize((64, 64)))

        # Normalize
        img_normalized = img_resized / 255.0
        # Add channel dimension
        img_normalized = img_normalized.reshape(64, 64, 1)

        X.append(img_normalized)
        y.append(label)

    return np.array(X), np.array(y)


# ============================================
# Generate and Prepare Dataset
# ============================================
print("🛡️  Malware Detection CNN Demo")
print("=" * 40)

benign_samples, malicious_samples = generate_dataset()
all_samples = benign_samples + malicious_samples
all_labels = [0] * len(benign_samples) + [1] * len(malicious_samples)

X, y = prepare_data(all_samples, all_labels)

# Split data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Input shape: {X_train[0].shape}")

# ============================================
# Visualize EICAR File as Image
# ============================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original EICAR
img_eicar, _, _ = binary_to_image(EICAR_STRING)
axes[0].imshow(img_eicar, cmap='gray')
axes[0].set_title('EICAR Test File\n(Original Size: 68×?)', fontsize=12)
axes[0].axis('off')

# Resized to 64x64
img_resized = np.array(Image.fromarray(img_eicar.astype('uint8')).resize((64, 64)))
axes[1].imshow(img_resized, cmap='gray')
axes[1].set_title('EICAR File\n(Resized: 64×64)', fontsize=12)
axes[1].axis('off')

# Benign sample comparison
benign_img, _, _ = binary_to_image("Hello world! This is a benign text file.")
benign_resized = np.array(Image.fromarray(benign_img.astype('uint8')).resize((64, 64)))
axes[2].imshow(benign_resized, cmap='gray')
axes[2].set_title('Benign Text File\n(64×64)', fontsize=12)
axes[2].axis('off')

plt.tight_layout()
plt.savefig('eicar_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# Build CNN Model
# ============================================
model = keras.Sequential([
    # First convolutional block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    layers.MaxPooling2D((2, 2)),

    # Second convolutional block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Third convolutional block
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Classifier
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n📊 CNN Architecture:")
model.summary()

# ============================================
# Train Model
# ============================================
print("\n🎯 Training CNN...")
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# ============================================
# Evaluate Model
# ============================================
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {test_acc:.4f} ({test_acc * 100:.1f}%)")

# ============================================
# Test on Pure EICAR File
# ============================================
eicar_img, _, _ = binary_to_image(EICAR_STRING)
eicar_resized = np.array(Image.fromarray(eicar_img.astype('uint8')).resize((64, 64)))
eicar_normalized = eicar_resized.reshape(1, 64, 64, 1) / 255.0

prediction = model.predict(eicar_normalized, verbose=0)[0][0]
print("\n🔬 Testing on EICAR test file:")
print(f"   Malicious probability: {prediction:.4f}")
print(f"   Classification: {'⚠️ MALICIOUS' if prediction > 0.5 else '✅ BENIGN'}")

# ============================================
# Visualize Training History
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history['accuracy'], 'b-', label='Training')
axes[0].plot(history.history['val_accuracy'], 'r-', label='Validation')
axes[0].set_title('Model Accuracy', fontsize=14)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'], 'b-', label='Training')
axes[1].plot(history.history['val_loss'], 'r-', label='Validation')
axes[1].set_title('Model Loss', fontsize=14)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# Feature Visualization
# ============================================
# Extract first layer filters
first_layer_weights = model.layers[0].get_weights()[0]
fig, axes = plt.subplots(4, 8, figsize=(12, 6))

for i, ax in enumerate(axes.flat):
    if i < 32:
        # Normalize weights for visualization
        filter_img = first_layer_weights[:, :, 0, i]
        filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min() + 1e-8)
        ax.imshow(filter_img, cmap='viridis')
        ax.axis('off')
    else:
        ax.axis('off')

plt.suptitle('First Layer Convolutional Filters (3×3) Learned During Training', fontsize=14)
plt.tight_layout()
plt.savefig('cnn_filters.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 40)
print("✅ CNN demonstration complete!")
print("📁 Visualizations saved as PNG files")
print("=" * 40)
```

------------------------------------------------------------------------

## 📊 Expected Output Visualizations

The program generates:

1.  **EICAR Visualization**
    -   Shows malware vs benign binary images.
2.  **Training History**
    -   Accuracy and loss graphs.
3.  **CNN Filter Visualization**
    -   Shows features learned by the convolution layers.

------------------------------------------------------------------------

## 🧪 Example Result

The CNN classifies files as:

-   ✅ Benign
-   ⚠️ Malicious

The model predicts malware probability using sigmoid output between 0
and 1.

------------------------------------------------------------------------

## 📌 Conclusion

Convolutional Neural Networks provide powerful automated feature
extraction capabilities. In cybersecurity, CNNs enable malware detection
by identifying hidden patterns in binary files. By transforming binary
data into images, CNNs can detect signatures that traditional
signature-based antivirus software may miss.

The provided Python implementation demonstrates malware detection using
the EICAR test file and synthetic benign samples. The model successfully
learns distinguishing patterns and achieves reliable classification
performance.

------------------------------------------------------------------------

## 📎 Included Data

-   EICAR Test String
-   Synthetic Benign Text Samples
-   Malware Variations

------------------------------------------------------------------------
