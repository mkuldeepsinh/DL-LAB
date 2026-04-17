# Complete Deep Learning Study Guide
## WCE Classification of Imbalanced Medical Datasets
### Covering All 4 Notebooks — Line-by-Line Explanation

---

# TABLE OF CONTENTS
1. Project Architecture & Big Picture
2. Environment Setup — Every Import Explained
3. Hyperparameters Deep Dive — Why Each Value Was Chosen
4. GPU Configuration & Mixed Precision
5. Task 1 — Dataset Exploration & Imbalance Analysis
6. Task 2 — Under-Sampling (Majority Class Control)
7. Task 3 — Augmentation-Based Over-Sampling
8. Task 4 — Data Pre-Processing & tf.data Pipeline
9. Task 5 — Transfer Learning Model Design
   - EfficientNetB0 Architecture
   - InceptionV3 Architecture
   - ResNet101V2 Architecture
   - Focal Loss & Label Smoothing
   - The `build_model()` Function — Line by Line
10. Task 6 — Intelligent Learning Rate Control
11. Task 7 — Training Under Three Settings
12. Callbacks — Phase-Aware Training Controls
13. Test-Time Augmentation (TTA)
14. Evaluation Notebook — All Metrics & Visualisations
15. Hyperparameter Comparison: EfficientNetB0 vs InceptionV3 vs ResNet101V2
16. Medical Metrics — Accuracy, Precision, Recall, F1 Explained
17. Results Interpretation & Analysis

---

# 1. PROJECT ARCHITECTURE & BIG PICTURE

## What is WCE?

**Wireless Capsule Endoscopy (WCE)** is a medical imaging technique where the patient swallows a pill-sized camera. It takes thousands of images as it travels through the gastrointestinal (GI) tract over 8–12 hours. Doctors then analyse these images to detect diseases like polyps, bleeding, inflammation, ulcers, etc.

**Why deep learning?** A gastroenterologist would need days to review 50,000+ images per patient. Deep learning can classify these images in seconds.

## Dataset — Kvasir-Capsule

The **Kvasir-Capsule** dataset is a real-world medical dataset with ~47,000 labelled WCE images across 14 GI disease classes. It is **severely imbalanced** — the most common class (Normal Clean Mucosa) has ~30,000 images, while the rarest (Blood-Fresh) may have only ~250.

This imbalance is medically realistic: most of the time the gut looks normal, rare diseases are… rare. But for a machine learning model, this is a nightmare — it learns to just predict the majority class.

## Four Notebooks and How They Connect

```
EfficientNetB0.ipynb  ──┐
InceptionV3.ipynb     ──┼──→  Saves results (.pkl) to Google Drive
ResNet101V2.ipynb     ──┘

Evaluation.ipynb  ←── Loads all 3 .pkl files → Compares all models × all settings
```

Each model notebook runs three "Settings":
- **Setting 1: Original** — raw imbalanced dataset
- **Setting 2: Under-Sampling** — majority classes capped
- **Setting 3: Under-Sampling + Augmentation** — balanced dataset

---

# 2. ENVIRONMENT SETUP — EVERY IMPORT EXPLAINED

```python
import os, glob, tarfile, shutil, random, warnings, re, time, gc
```

| Import | Why it is used |
|--------|---------------|
| `os` | File system operations — creating folders, listing directories, joining paths |
| `glob` | Finding files by pattern, e.g., `*.tar.gz` — finds all compressed class archives |
| `tarfile` | Opening and extracting `.tar.gz` compressed archives (each disease class is one archive) |
| `shutil` | Copying, moving, and deleting entire directories — used for creating balanced datasets |
| `random` | Random sampling for under-sampling, choosing images for augmentation |
| `warnings` | `warnings.filterwarnings('ignore')` suppresses TensorFlow/sklearn deprecation spam |
| `re` | Regular expressions — used to clean model/setting names for safe file names |
| `time` | Timing training runs |
| `gc` | **Garbage Collector** — critical on Colab's 12 GB RAM limit; forces Python to free memory between training runs |

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from PIL import Image
from collections import Counter
```

| Import | Why it is used |
|--------|---------------|
| `numpy` | Numerical arrays — images are stored as numpy arrays; used for averaging predictions, computing metrics |
| `pandas` | DataFrames — storing class distribution tables, comparison results |
| `matplotlib.pyplot` | All plotting — bar charts, training curves, confusion matrices |
| `matplotlib.ticker` | `ticker.ScalarFormatter()` — needed to show real numbers (not scientific notation) on log-scale axes |
| `seaborn` | Prettier statistical plots, heatmaps; `sns.set_style('whitegrid')` gives clean background |
| `PIL.Image` | Opening, resizing, displaying individual images for inspection |
| `collections.Counter` | Quickly counting class frequencies in lists |

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, save_img
```

| Import | Why it is used |
|--------|---------------|
| `tensorflow` | The core deep learning framework — builds, trains, and runs neural networks on GPU |
| `keras` | High-level API inside TensorFlow — makes building models readable |
| `layers` | All neural network layer types: `Dense`, `Dropout`, `BatchNormalization`, `GlobalAveragePooling2D` |
| `regularizers` | `L2` regularization — adds weight penalty to loss to prevent overfitting |
| `callbacks` | `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint` — control training automatically |
| `optimizers` | `Adam` and `AdamW` — the algorithms that update weights |
| `ImageDataGenerator` | **Offline augmentation** — used in Task 3 to create and save augmented images to disk |
| `load_img, img_to_array, save_img` | PIL-compatible image utilities for the augmentation loop |

```python
from tensorflow.keras.applications import EfficientNetB0, InceptionV3, ResNet101V2
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inc_preprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as res_preprocess
```

**Why three separate preprocessing functions?** Each backbone was pretrained on ImageNet with a DIFFERENT pixel normalisation scheme:
- **EfficientNet** expects pixels in `[0, 1]` (divides by 255, then applies per-channel scaling)
- **InceptionV3** expects pixels in `[-1, +1]` (subtracts 127.5, divides by 127.5)
- **ResNet101V2** expects pixels in `[-1, +1]` (similar to Inception)

Using the **wrong** `preprocess_input` with a backbone is one of the most common transfer learning mistakes. It degrades performance significantly because the backbone's learned weights expect a specific input range.

```python
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score, recall_score, f1_score)
```

| Import | Why it is used |
|--------|---------------|
| `train_test_split` | Stratified split — ensures each class has the same proportion in train/val/test |
| `compute_class_weight` | Computes inverse-frequency weights so rare classes get higher loss penalty |
| `classification_report` | Per-class Precision/Recall/F1 — essential for medical evaluation |
| `confusion_matrix` | Shows which classes are confused with each other — critical for diagnosis errors |
| `accuracy_score` etc. | Individual metric computation for the comparison table |

---

# 3. HYPERPARAMETERS DEEP DIVE — WHY EACH VALUE

## Global Hyperparameters (EfficientNetB0 — Full Training)

```python
IMG_SIZE        = 224      # works well for all three backbones
BATCH_SIZE      = 16       # 32→16: halves per-step GPU RAM (key RAM fix)
THRESHOLD       = 500      # more images per class → better generalisation
WARMUP_LR       = 5e-4    # stable head warm-up LR
FINETUNE_LR     = 1e-4    # lower LR for backbone fine-tuning
WARMUP_EPOCHS   = 10      # 10 epochs head-only warm-up
FINETUNE_EPOCHS = 20      # 20 epoch fine-tune cap → 30 total epochs
TOTAL_EPOCHS    = WARMUP_EPOCHS + FINETUNE_EPOCHS
UNFREEZE_RATIO  = 0.20    # unlock top 20 % of backbone
LABEL_SMOOTHING = 0.1     # (implied from FocalLoss call)
```

### IMG_SIZE = 224
- All three backbones (EfficientNetB0, InceptionV3, ResNet101V2) can accept 224×224 images
- 224 is the standard ImageNet training size — closest to what the pretrained weights expect
- Higher resolution (336×336) would improve accuracy but doubles memory and computation
- InceptionV3 technically prefers 299×299, but 224 works acceptably and saves RAM

### BATCH_SIZE = 16 (EfficientNetB0) vs 32 (InceptionV3, ResNet101V2)
- **Batch size** = number of images processed simultaneously before updating weights
- EfficientNetB0 uses 16 because it trains for 30 epochs (longer), needs RAM headroom
- InceptionV3 and ResNet101V2 use 32 (fewer epochs, faster to finish within Colab's 1-hr limit)
- **Why not larger?** Colab T4 GPU has 16 GB VRAM. ResNet101V2 alone has 42M parameters. With 32 images of size 224×224×3 as float16, that's already several GB. OOM (Out of Memory) crashes terminate the entire session.
- **Why not smaller (8)?** Each batch must represent the class distribution reasonably. With 14 classes and batch=8, many batches won't contain all classes, making loss estimates noisy.

### THRESHOLD = 500 (EfficientNetB0) vs 300 (InceptionV3, ResNet101V2)
- This is the **cap for under-sampling** AND the **target for augmentation over-sampling**
- EfficientNetB0 uses 500 because it trains longer (30 epochs) — more data per class is OK
- InceptionV3/ResNet101V2 use 300 to reduce total dataset size → fewer batches → faster epoch completion within Colab's 1-hour runtime limit
- **The trade-off**: Lower threshold → faster training but less data diversity → possibly lower accuracy

### WARMUP_LR = 5e-4
- During warmup, only the **new classification head** trains; the backbone is frozen
- `5e-4 = 0.0005` — moderately high learning rate so the new head layers converge quickly
- If too low (e.g., 1e-5): head won't learn fast enough in just 10 epochs
- If too high (e.g., 1e-2): loss becomes unstable ("exploding gradients")

### FINETUNE_LR = 1e-4
- During fine-tuning, the **top 20% of backbone layers** unfreeze and train
- Must be **lower than WARMUP_LR** — the backbone already has good ImageNet features; we want to gently adjust them, not destroy them
- Rule of thumb: fine-tune LR = warmup LR ÷ 5
- `1e-4 = 0.0001`

### WARMUP_EPOCHS = 10 (EfficientNetB0) vs 5 (InceptionV3/ResNet101V2)
- Head-only training first: gives the randomly initialised dense layers time to stabilise BEFORE the backbone starts changing too
- If you unfreeze the backbone immediately with a random head, the large gradients from the untrained head will damage the backbone's pretrained weights — called **"catastrophic forgetting"**
- EfficientNetB0 uses 10 epochs (more careful); InceptionV3/ResNet use 5 (faster, Colab budget)

### FINETUNE_EPOCHS = 20 (EfficientNetB0) vs 10 (InceptionV3/ResNet101V2)

### UNFREEZE_RATIO = 0.20
- Only the **top 20%** of backbone layers are unlocked for fine-tuning
- Bottom 80% stay frozen — these learn low-level features (edges, textures) that transfer perfectly from ImageNet to WCE images
- Top 20% learn high-level, dataset-specific features (polyp shapes, bleeding patterns)
- Why not 100% unfreeze? Risk of overfitting on small medical datasets; also much slower

### SEED = 42
- Setting the same random seed makes experiments **reproducible**
- `random.seed(42)` → Python's `random` module
- `np.random.seed(42)` → NumPy operations
- `tf.random.set_seed(42)` → TensorFlow's CUDA kernels
- Without this, every run gives different splits, different augmentations → can't compare fairly

---

# 4. GPU CONFIGURATION & MIXED PRECISION

## Mixed Precision Training

```python
try:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision: mixed_float16 enabled")
except Exception as e:
    print(f"Mixed precision not available ({e}) — continuing in float32")
```

**What is mixed precision?**
- Standard training uses **float32** (32-bit floating point) — each weight uses 4 bytes
- Mixed precision uses **float16** (16-bit) for most operations but keeps critical ones in float32
- **Benefits**: ~2× faster on modern NVIDIA GPUs (T4, V100, A100 have Tensor Cores optimised for float16); uses ~half the VRAM
- **Risk**: float16 has a smaller numerical range → gradients can underflow to zero ("vanishing gradients") or overflow to infinity. TensorFlow handles this automatically with **loss scaling**
- The `try/except` makes the code compatible with older TF versions where the API differs

## GPU Memory Growth

```python
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Memory growth (already initialized): {e}")
```

**Why is this important?**
- By default, TensorFlow allocates **ALL available VRAM** the moment it starts — even if it only needs 2 GB of the 16 GB available
- This immediately fails if any other process (like another Colab kernel tab) has reserved some VRAM
- `set_memory_growth(True)` tells TF to allocate VRAM **gradually as needed**
- The `RuntimeError` happens if this is called after TF has already initialised — the except handles that gracefully

---

# 5. TASK 1 — DATASET EXPLORATION & IMBALANCE ANALYSIS

## Counting Images Per Class

```python
class_counts = {}
for cls in sorted(os.listdir(IMAGE_DIR)):
    cls_path = os.path.join(IMAGE_DIR, cls)
    if os.path.isdir(cls_path):
        imgs = [f for f in os.listdir(cls_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        class_counts[cls] = len(imgs)
```

**Line-by-line:**
- `sorted(os.listdir(IMAGE_DIR))` — lists all entries in the images folder, sorted alphabetically for consistency
- `os.path.isdir(cls_path)` — skips any stray files; only processes directories (each class is a folder)
- `f.lower().endswith(...)` — `.lower()` handles uppercase extensions (.JPG on Windows); includes all common image formats
- Result: `class_counts = {'angiectasia': 866, 'blood_fresh': 446, ...}`

## The Visualisation — Log Scale Bar Chart

```python
ax.set_yscale('log')
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
```

**Why log scale?** The Kvasir-Capsule dataset has classes ranging from ~250 to ~30,000 images. On a linear scale, the minority class bars would be invisible (too small to see). A log scale compresses the range so all classes are visible.

**`ScalarFormatter()`** forces matplotlib to show real numbers (250, 1000, 10000) instead of scientific notation (2.5×10², 1×10³) which is harder to read.

## Category Classification

```python
category_map = {c: ('Majority' if n > 2000 else 'Moderate' if n > 500 else 'Minority')
                for c, n in zip(df_dist['Class'], df_dist['Count'])}
```

This classifies each class into one of three groups:
- **Majority** (>2000 images): dominant classes — will bias model if not handled
- **Moderate** (500–2000): middle ground
- **Minority** (≤500): rare diseases — hardest to learn; most medically important

## Imbalance Ratio

```python
imbalance_ratio = majority_class['Count'] / minority_class['Count']
```

In Kvasir-Capsule, this is roughly **30,000 / 250 ≈ 120:1**. This means the model sees 120 normal images for every 1 rare disease image. Without correction, the model will learn to say "it's normal" 99% of the time and still achieve 99% accuracy — but detect 0% of actual diseases. This is the **accuracy paradox** in medical AI.

## Three-Panel Imbalance Analysis Plot

The second visualisation shows three charts:
1. **Pie chart**: proportion of classes in each category (Majority/Moderate/Minority)
2. **Horizontal bar (log scale)**: imbalance ratio per class vs the majority — shows 120× worse for the rarest class
3. **Stacked bar**: total image volume by category — shows majority classes dominate 90%+ of data

---

# 6. TASK 2 — UNDER-SAMPLING

## The Problem Under-Sampling Solves

If we train on the original data, the model sees `normal_clean_mucosa` 30,000 times but `blood_fresh` only 250 times. The gradient updates from the rare class get **drowned out** by the many gradient updates from the majority class.

## Implementation

```python
THRESHOLD = 500  # EfficientNetB0 uses 500; InceptionV3/ResNet use 300

for cls in sorted(os.listdir(IMAGE_DIR)):
    src_dir = os.path.join(IMAGE_DIR, cls)
    dst_dir = os.path.join(UNDERSAMPLED_DIR, cls)
    os.makedirs(dst_dir, exist_ok=True)

    imgs = [f for f in os.listdir(src_dir) if ...]
    original_count = len(imgs)

    if original_count > THRESHOLD:
        selected = random.sample(imgs, THRESHOLD)
        data_loss[cls] = original_count - THRESHOLD
    else:
        selected = imgs   # keep ALL minority class images
        data_loss[cls] = 0

    for img_name in selected:
        shutil.copy2(os.path.join(src_dir, img_name),
                     os.path.join(dst_dir, img_name))
```

**Key decisions:**
- `random.sample(imgs, THRESHOLD)` — **random** under-sampling: picks images randomly without replacement; simple and effective
- Classes BELOW threshold keep ALL their images — we don't drop any rare disease samples
- `shutil.copy2` — copies file **with metadata** (timestamps preserved); the original dataset remains untouched

**Alternative approaches not used here:**
- **Cluster Centroids**: under-sample by removing redundant similar images — more sophisticated but much slower
- **Tomek Links**: removes samples near the decision boundary — complex but reduces noise

## Trade-off: Data Loss vs Balance

After under-sampling with threshold=500:
- Before: ~47,000 images, imbalance ratio ~120:1
- After: ~7,000 images (14 classes × 500), imbalance ratio ~1:1 to ~3:1
- Cost: we lose ~40,000 images of majority classes

Is this loss acceptable? For the **majority classes**, yes — they have so many similar frames that the model still sees sufficient variety. The deleted images are mostly near-duplicates from consecutive video frames.

---

# 7. TASK 3 — AUGMENTATION-BASED OVER-SAMPLING

## Why Augment Instead of Just Under-Sample?

Pure under-sampling loses data. Augmentation generates **synthetic but realistic** new samples by transforming existing images. This is especially valuable for rare disease classes where we only have 50–250 real samples.

## The Augmentation Configuration

```python
augmentor = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    brightness_range=[0.85, 1.15],
    fill_mode='nearest'
)
```

**Each parameter explained:**

### `horizontal_flip=True`
- Randomly flips the image left-right (50% probability)
- Why valid for WCE? The GI tract can have lesions on any side — a polyp on the left is the same disease as one on the right. Horizontal flipping is **medically valid**
- Creates 2× effective dataset size

### `vertical_flip=True`
- Randomly flips image top-bottom
- Also valid for WCE — the capsule tumbles randomly in the gut, capturing frames from any orientation

### `rotation_range=20`
- Random rotation between -20° and +20°
- WCE images are captured at arbitrary angles as the capsule rotates
- Why not 90° or 180°? Beyond ~30°, text overlays (timestamps) and circular fisheye borders look unnatural

### `width_shift_range=0.15` and `height_shift_range=0.15`
- Shifts the image horizontally/vertically by up to 15% of the image size
- Simulates the capsule being positioned slightly differently in the gut lumen
- 15% of 224 pixels = ≈33 pixels of shift

### `zoom_range=0.15`
- Randomly zooms in or out by up to 15%
- Simulates proximity differences between the capsule and the gut wall

### `brightness_range=[0.85, 1.15]`
- Randomly adjusts brightness to 85%–115% of original
- WCE LED illumination varies; different gut sections have different reflectance

### `fill_mode='nearest'`
- After rotation or shift, empty corners appear. `'nearest'` fills them by repeating the nearest edge pixel
- **Why not 'constant'?** Constant fill (e.g., black corners) would create a visible artifact that the model might learn to associate with augmented images, creating a spurious feature
- **Why not 'reflect'?** Reflect is also good but nearest is simpler and produces fewer visible seams

## The Over-Sampling Loop

```python
if current_count < THRESHOLD:
    needed = THRESHOLD - current_count
    generated = 0
    while generated < needed:
        src_img_name = random.choice(imgs)
        img = load_img(src_img_path, target_size=(224, 224))
        img_arr = img_to_array(img).reshape(1, 224, 224, 3)
        aug_iter = augmentor.flow(img_arr, batch_size=1)
        aug_img = next(aug_iter)[0].astype(np.uint8)
        aug_name = f'aug_{generated:04d}_{src_img_name}'
        save_img(os.path.join(cls_dir, aug_name), aug_img)
        generated += 1
```

**Line-by-line:**
- Only runs for classes with fewer than THRESHOLD images — majority classes are not augmented (they're already at the cap from under-sampling)
- `random.choice(imgs)` — picks a random source image each iteration (not the same image each time, so variety is maximised)
- `reshape(1, 224, 224, 3)` — ImageDataGenerator expects a 4D tensor (batch, height, width, channels), so we add the batch dimension
- `augmentor.flow(img_arr, batch_size=1)` — creates an augmentation iterator
- `next(aug_iter)[0]` — gets one augmented image; `[0]` removes the batch dimension
- `.astype(np.uint8)` — converts float back to 8-bit integers for saving as JPEG/PNG
- `aug_name = f'aug_{generated:04d}_{src_img_name}'` — names augmented files distinctly so they can be identified later (the `before_after` loop counts by prefix)
- `save_img(...)` — saves to disk alongside the original images

**Result:** Every class in `BALANCED_DIR` now has exactly `THRESHOLD` images. Imbalance ratio = 1:1.

---

# 8. TASK 4 — DATA PRE-PROCESSING & tf.data PIPELINE

## Collecting Paths and Labels

```python
all_paths = []
all_labels = []

for cls in sorted(os.listdir(BALANCED_DIR)):
    cls_dir = os.path.join(BALANCED_DIR, cls)
    for img_name in os.listdir(cls_dir):
        if img_name.lower().endswith(('.jpg','.jpeg','.png','.bmp')):
            all_paths.append(os.path.join(cls_dir, img_name))
            all_labels.append(cls)

CLASS_NAMES = sorted(list(set(all_labels)))
label_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
all_labels_idx = [label_to_idx[l] for l in all_labels]
```

- Collects file paths (strings) NOT pixels — we don't load the images into RAM yet
- `sorted(CLASS_NAMES)` ensures consistent label-to-index mapping across all three notebooks (so class 0 always means the same disease)
- `label_to_idx = {'angiectasia': 0, 'blood_fresh': 1, ...}` — integer encoding for loss computation

## Stratified Train/Val/Test Split

```python
X_train, X_temp, y_train, y_temp = train_test_split(
    all_paths, all_labels_idx, test_size=0.30,
    stratify=all_labels_idx, random_state=SEED)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50,
    stratify=y_temp, random_state=SEED)
```

**Two-step split for 70/15/15:**
- Step 1: Split 100% → 70% train + 30% temp
- Step 2: Split 30% temp → 15% val + 15% test (50% of 30% = 15%)

**`stratify=all_labels_idx`** — this is critical. Without stratification, random splits might have 80% normal cases in the train set and only 20% in the test set. Stratification ensures **each split has the same class proportions** as the full dataset, making evaluation fair.

## The tf.data Pipeline

```python
AUTOTUNE    = tf.data.AUTOTUNE
SHUFFLE_BUFFER = min(len(X_train), 2000)
PREFETCH_SIZE  = 2

def load_and_preprocess(path, label):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    return img, label
```

**Why tf.data instead of loading all images into a NumPy array?**
- 14 classes × 500 images × 224×224×3 pixels × 4 bytes = ~450 MB in float32
- With multiple splits and models, this quickly exceeds 12 GB RAM
- `tf.data` loads images **lazily** — only the current batch is in memory at any time

**`tf.io.read_file(path)`** — reads the raw bytes of an image file
**`tf.io.decode_image`** — decodes JPEG/PNG/BMP/GIF automatically; `channels=3` forces RGB (some images might be greyscale or RGBA)
**`img.set_shape([None, None, 3])`** — tells TensorFlow the shape is (height, width, 3) even before we know the exact dimensions; this is needed for the graph to compile correctly
**`tf.image.resize`** — resizes to 224×224 using bilinear interpolation (smooth, high quality)
**`tf.cast(..., tf.float32)`** — converts uint8 (0–255) to float32; keeps values in 0–255 range at this stage because preprocessing is applied later inside the model

## Online Augmentation (Training Time)

```python
@tf.function
def augment_image(img, label):
    img = img / 255.0          # normalize to [0,1] for colour space ops

    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.15)
    img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
    img = tf.image.random_saturation(img, lower=0.80, upper=1.20)
    img = tf.image.random_hue(img, max_delta=0.04)
    img = tf.clip_by_value(img, 0.0, 1.0)

    # Random crop + resize (simulates zoom)
    crop_h = tf.cast(IMG_SIZE * tf.random.uniform([], 0.85, 1.0), tf.int32)
    crop_w = tf.cast(IMG_SIZE * tf.random.uniform([], 0.85, 1.0), tf.int32)
    img = tf.image.random_crop(img, size=[crop_h, crop_w, 3])
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])

    img = img * 255.0          # scale back to [0,255] for model preprocessing
    img = tf.clip_by_value(img, 0.0, 255.0)
    return img, label
```

**`@tf.function`** — compiles the function into a TensorFlow graph (much faster than eager mode; runs entirely on GPU without Python overhead)

**Why divide by 255 before colour augmentations?**
`tf.image.random_saturation` and `tf.image.random_hue` internally convert to HSV colour space and **require values in [0, 1]**. If you pass [0, 255], they produce garbage outputs. The code divides first, augments, then multiplies back to [0, 255].

**`tf.image.random_saturation(img, lower=0.80, upper=1.20)`** — adjusts colour saturation (0.8 = slightly greyer, 1.2 = slightly more vivid); helps model generalise to colour variation in endoscopy lighting

**`tf.image.random_hue(img, max_delta=0.04)`** — very subtle hue shift (4% of full hue circle = about 14°); keeps colours medically realistic (blood should stay red, not become blue)

**Random crop + resize** — instead of just zooming, this:
1. Crops a random sub-region (85%–100% of the image)
2. Resizes it back to 224×224
This simulates the capsule being at different distances from the gut wall and provides a cheap "zoom" augmentation.

## Why Two Augmentation Passes? (Offline + Online)

- **Task 3 (offline)**: Creates augmented files on disk specifically for minority classes to balance counts
- **Task 4 (online)**: Applies live augmentation to ALL training images during loading — provides additional diversity beyond the saved files

This is best practice: offline fixes the count imbalance; online provides continuous variety.

## Dataset Construction

```python
def make_dataset(paths, labels, shuffle=False, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=SHUFFLE_BUFFER, seed=SEED)
    if augment:
        ds = ds.map(augment_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(PREFETCH_SIZE)
    return ds
```

**`from_tensor_slices((paths, labels))`** — creates a dataset where each element is `(path_string, label_int)`

**`.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)`** — applies the loading function in parallel across CPU cores; `AUTOTUNE` lets TF choose the optimal number of threads

**`.shuffle(buffer_size=SHUFFLE_BUFFER)`** — randomly shuffles the dataset. The buffer_size controls quality: a buffer of 2000 means TF picks each sample randomly from a pool of 2000. Larger = better shuffling but more RAM.
- Only applied to training (not val/test) — evaluation must be deterministic

**`.batch(BATCH_SIZE)`** — groups samples into batches; the model processes one batch per gradient step

**`.prefetch(PREFETCH_SIZE)`** — while the GPU trains on batch N, the CPU pre-loads batch N+1; eliminates "data starvation" where the GPU sits idle waiting for data. `PREFETCH_SIZE=2` means 2 batches pre-loaded.

---

# 9. TASK 5 — TRANSFER LEARNING MODEL DESIGN

## What Is Transfer Learning?

Training a deep learning model from scratch requires millions of images. The three backbones here were trained on **ImageNet** — 1.2 million images across 1000 categories. They've already learned:
- **Early layers**: edges, corners, gradients, textures
- **Middle layers**: shapes, object parts, colour patterns
- **Late layers**: high-level object representations

We take these learned features and retrain only the final layers for our specific 14-class WCE task. This is "standing on the shoulders of giants."

## EfficientNetB0 Architecture

**EfficientNet** (2019, Google Brain) is based on the insight that scaling network depth, width, and resolution together (compound scaling) is much more efficient than scaling any one dimension alone.

- **B0 = Baseline** (smallest in the B0–B7 family)
- **Parameters**: ~5.3 million (very compact)
- **Key building block**: Mobile Inverted Bottleneck Convolution (MBConv) with Squeeze-and-Excitation

**EfficientNetB0 structure:**
```
Input (224×224×3)
→ Stem Conv (3×3, stride 2) → 112×112×32
→ MBConv1 (7 blocks with squeeze-excitation)
→ GlobalAveragePooling2D → 1280-d vector
→ [Our Classification Head]
```

**Why EfficientNetB0 for medical imaging?**
- Compact (5M params) → less overfitting risk on small medical datasets
- High accuracy relative to parameter count (ImageNet top-1: 77.1%)
- Fast inference — good for clinical use

## InceptionV3 Architecture

**InceptionV3** (2016, Google) uses "Inception modules" — parallel convolutions of different sizes processed simultaneously:

```
Input (224×224×3)
→ Stem layers (Conv2D, MaxPool)
→ 3× Inception-A modules (parallel 1×1, 3×3, 5×5 convs)
→ Reduction-A
→ 4× Inception-B modules (factorised convolutions: 7×1 and 1×7)
→ Reduction-B
→ 2× Inception-C modules (highly parallel: 1×3 and 3×1)
→ GlobalAveragePooling2D → 2048-d vector
→ [Our Classification Head]
```

- **Parameters**: ~23.8 million
- **Key innovation**: Factorising n×n convolutions into 1×n and n×1 reduces computation by 33%
- **ImageNet accuracy**: 78.8% top-1

## ResNet101V2 Architecture

**ResNet** (2016, Microsoft) introduced **skip connections (residual connections)**:

```
x + F(x)  (instead of just F(x))
```

This solves the **vanishing gradient problem** — gradients can flow directly back through the skip connections, enabling very deep networks (101 layers here).

**V2** improvement: applies BatchNorm and ReLU **before** the convolution (pre-activation) instead of after, which improves gradient flow further.

- **Parameters**: ~44.6 million (largest of the three)
- **ImageNet accuracy**: 79.7% top-1
- **Trade-off**: Most accurate but slowest and most memory-hungry

## The `build_model()` Function — Line by Line

```python
def build_model(base_model_fn, model_name,
                input_shape=(IMG_SIZE, IMG_SIZE, 3),
                num_classes=NUM_CLASSES, freeze_base=True):
```

Parameters:
- `base_model_fn`: the constructor function (e.g., `EfficientNetB0`)
- `model_name`: string key to look up the correct preprocessing function
- `freeze_base=True`: during warmup, freeze ALL backbone layers

```python
    base = base_model_fn(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
```

- `include_top=False` — loads the backbone WITHOUT its original 1000-class output layer; we'll add our own 14-class head
- `weights='imagenet'` — downloads and loads ImageNet pretrained weights (downloaded once, cached)
- `input_shape=(224, 224, 3)` — tells TF the expected input tensor shape

```python
    total_layers = len(base.layers)
    freeze_until = int(total_layers * 0.80)   # freeze first 80%

    if freeze_base:
        base.trainable = False   # freeze entire backbone (warmup phase)
    else:
        for layer in base.layers[:freeze_until]:
            layer.trainable = False     # keep early layers frozen
        for layer in base.layers[freeze_until:]:
            layer.trainable = True      # train top 20%
```

**Why freeze the bottom 80%?**
The bottom layers learn universal features (edges, textures, colours) that are identical across all image domains — ImageNet's learned features work perfectly for WCE images. Only the top 20% learn dataset-specific high-level patterns.

```python
    inputs = keras.Input(shape=input_shape, name='input_image')

    preprocess_fn = _PREPROCESS[model_name]
    x = tf.keras.layers.Lambda(
        lambda img: preprocess_fn(img),
        name='preprocessing'
    )(inputs)
```

- `keras.Input` creates the input placeholder — a node in the computation graph
- The `Lambda` layer applies the backbone-specific preprocessing inside the model graph
- This means preprocessing happens on the GPU (not the CPU), and is included when saving/loading the model

```python
    x = base(x, training=not freeze_base)
```

- `training=False` during warmup: BatchNorm layers in the backbone use their **frozen** running statistics (mean/variance computed on ImageNet). This is critical — if `training=True` while frozen, BatchNorm would try to update its statistics using our small dataset, corrupting the pretrained values
- `training=True` during fine-tuning: BatchNorm layers update to adapt to the WCE data

```python
    x = layers.GlobalAveragePooling2D(name='gap')(x)
```

**GlobalAveragePooling2D (GAP)**:
- Takes the feature maps (e.g., 7×7×1280 for EfficientNetB0) and averages each channel across all spatial positions
- Output: a 1280-dimensional vector (one number per channel)
- **Why GAP instead of Flatten?** Flatten(7×7×1280) = 62,720-dimensional vector → massive parameter count in the next Dense layer. GAP(7×7×1280) = 1280-d → 100× fewer parameters → 100× less overfitting risk
- GAP also provides spatial invariance — the model is less sensitive to where exactly in the image the feature appears

```python
    x = layers.BatchNormalization(name='bn_head')(x)
```

**BatchNormalization** normalises the output of GAP so that the mean is near 0 and variance is near 1 across a batch. This:
- Stabilises training (prevents one feature from dominating)
- Acts as mild regularisation
- Allows higher learning rates

```python
    x = layers.Dense(256, activation=None, 
                     kernel_regularizer=regularizers.l2(L2_REG),
                     name='fc_256')(x)
    x = layers.BatchNormalization(name='bn_256')(x)
    x = layers.Activation('relu', name='relu_256')(x)
    x = layers.Dropout(DROPOUT_RATE, name='drop_256')(x)

    x = layers.Dense(128, activation=None,
                     kernel_regularizer=regularizers.l2(L2_REG),
                     name='fc_128')(x)
    x = layers.BatchNormalization(name='bn_128')(x)
    x = layers.Activation('relu', name='relu_128')(x)
    x = layers.Dropout(DROPOUT_RATE, name='drop_128')(x)

    outputs = layers.Dense(num_classes, activation='softmax',
                           dtype='float32', name='predictions')(x)
```

**The Classification Head:**

**Dense(256)** → **Dense(128)** → **Dense(14)** (two hidden layers + output)

Why two hidden layers? One layer maps 1280→14 directly, which is a very large jump. Two intermediate layers allow the model to learn more complex decision boundaries.

**`activation=None` then `BatchNorm` then `Activation('relu')`** — this is the "BN before activation" pattern. More numerically stable than doing Dense → ReLU → BN.

**`kernel_regularizer=regularizers.l2(L2_REG)`** — L2 regularisation adds `L2_REG × sum(weights²)` to the loss. This penalises large weights, encouraging the model to spread importance across many features rather than relying on a few → reduces overfitting.

**`Dropout(DROPOUT_RATE)`** — during training, randomly sets DROPOUT_RATE% of neurons to zero each forward pass. The model must learn redundant representations (can't rely on any single neuron). During inference (`model.predict`), dropout is automatically disabled and all neurons are used.

**`Dense(num_classes, activation='softmax', dtype='float32')`**:
- `softmax` converts raw logits to a probability distribution that sums to 1.0
- `dtype='float32'` — CRITICAL with mixed precision! The final layer must be float32 even when using float16 for training. Softmax with float16 can overflow to infinity on extreme logit values. Keras automatically upcasts the last layer when `dtype='float32'` is specified.

## Focal Loss — Why and How

```python
class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma=2.0, label_smoothing=0.1, **kw):
        super().__init__(**kw)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        num_classes = tf.shape(y_pred)[-1]
        y_true_oh = tf.one_hot(tf.cast(y_true, tf.int32), num_classes)
        # Label smoothing
        y_true_oh = (y_true_oh * (1.0 - self.label_smoothing) +
                     self.label_smoothing / tf.cast(num_classes, tf.float32))
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true_oh * tf.math.log(y_pred)
        weight = y_true_oh * tf.pow(1.0 - y_pred, self.gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * ce, axis=-1))
```

**Regular Cross-Entropy Loss:**
```
CE = -log(p_correct_class)
```
Problem: If the model is very confident on easy examples (p=0.99), CE = -log(0.99) ≈ 0.01 — tiny. But these tiny losses from thousands of easy examples dominate the gradient, crowding out the gradient from hard/rare examples.

**Focal Loss:**
```
FL = -(1 - p_correct_class)^gamma × log(p_correct_class)
```
- If the model is confident (p=0.99): weight = (1-0.99)^2 = 0.0001 — effectively zero
- If the model is uncertain (p=0.50): weight = (1-0.50)^2 = 0.25 — substantial
- The `gamma=2.0` parameter controls how aggressively easy examples are down-weighted

**Label Smoothing (0.1)**:
- Instead of hard labels [0, 0, 1, 0] (perfect certainty), uses soft labels [0.007, 0.007, 0.93, 0.007]
- Prevents the model from being overconfident — it learns "I'm quite sure it's class 2" rather than "I'm 100% certain it's class 2"
- Mathematically: `y_smooth = y_hard × (1 - ε) + ε / num_classes`
- Reduces overfitting and improves calibration (predicted probabilities match actual frequencies)

**`tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)`** — prevents `log(0) = -∞` (numerical instability)

---

# 10. TASK 6 — INTELLIGENT LEARNING RATE CONTROL

## Two-Phase Learning Rate Strategy

### Phase 1: Warmup (Constant LR)

```python
WARMUP_LR     = 5e-4
WARMUP_EPOCHS = 10  # (EfficientNetB0) / 5 (others)
```

During warmup, only the classification head trains. The LR stays constant at `5e-4`. Why constant? The head starts with random weights and needs to converge reliably. Fancy schedules would just add instability before the head is even useful.

The `ReduceLROnPlateau` callback can reduce this LR further if validation loss plateaus:

```python
callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,     # new LR = old LR × 0.3 (reduces by 70%)
    patience=3,     # wait 3 epochs with no improvement before reducing
    min_delta=MIN_DELTA,
    min_lr=1e-8    # never go below 10^-8
)
```

**`factor=0.3`**: More aggressive than the typical 0.1 or 0.5, but appropriate here because we have a short warmup (5–10 epochs) and need decisive reduction if learning stalls.

**`patience=3`**: Wait 3 epochs before reducing — avoids reacting to temporary noise in validation loss.

### Phase 2: Fine-Tuning (Cosine Decay)

```python
ft_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=FINETUNE_LR,    # 1e-4
    decay_steps=FINETUNE_EPOCHS * steps_per_epoch,
    alpha=1e-7     # minimum LR at end of decay
)
```

**Cosine Decay** smoothly reduces LR from `FINETUNE_LR` to near-zero following a cosine curve:
```
LR(t) = FINETUNE_LR × 0.5 × (1 + cos(π × t / total_steps))
```
- Early in fine-tuning: LR is close to `FINETUNE_LR` — the backbone adapts meaningfully
- Later: LR decays gracefully → model converges to a sharp minimum
- Why cosine over linear? Cosine decays slowly at first (good: backbone needs time to adapt) and then faster near the end (good: aggressive final convergence)

**Why no `ReduceLROnPlateau` during fine-tuning?**
```python
# The finetune phase uses CosineDecay (a LearningRateSchedule),
# so adding RLRP there raises:
# TypeError: optimizer was created with a LearningRateSchedule
```
Keras does not allow `ReduceLROnPlateau` when the optimizer is already using a `LearningRateSchedule` — they both try to control the LR and conflict.

## Visualisation: LR vs Epoch

The code plots three subplots:
1. Warmup phase: flat line at 5e-4
2. Fine-tune phase: cosine curve from 1e-4 to ~0
3. Combined: full 30-epoch schedule showing the "step down" at epoch 10 where phase switches

In the evaluation notebook, this plot is reconstructed from the saved `history` dict — if `'learning_rate'` was logged by Keras, it's used directly; otherwise a synthetic version is computed from the schedule formula.

---

# 11. TASK 7 — TRAINING UNDER THREE SETTINGS

## The Three Settings

```python
dataset_configs = {
    'Setting 1: Original':
        (IMAGE_DIR,        False),   # original imbalanced, no online augmentation
    'Setting 2: Under-Sampling':
        (UNDERSAMPLED_DIR, False),   # caps applied, no online augmentation
    'Setting 3: Under-Sampling + Augmentation':
        (BALANCED_DIR,     True),    # balanced + online augmentation during training
}
```

**Setting 1 (Original)** is the **baseline** — shows how badly the model performs without any imbalance handling. Expected: high overall accuracy but terrible performance on minority classes.

**Setting 2 (Under-Sampling)** shows the effect of just capping majority classes. Expected: better minority class recall at the cost of some overall accuracy.

**Setting 3 (Full Pipeline)** combines under-sampling (count balancing) + augmentation (diversity). Expected: best minority class F1, best generalisation.

## The `train_and_evaluate()` Function

```python
def train_and_evaluate(model_fn, data_dir, augment_train, setting_name):
    tf.keras.backend.clear_session()
    gc.collect()
    tf.keras.backend.clear_session()  # double-clear flushes TF graph cache
```

**Double `clear_session()`** — the second call forces TensorFlow to also release cached graph structures. Without this, memory from the previous training run can persist, causing OOM errors on the second or third setting.

```python
    # Phase 1: Warmup — freeze backbone, train only head
    model, base, tl_params, fu_params = build_model(model_fn, model_name, freeze_base=True)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=WARMUP_LR),
        loss=focal_loss,
        metrics=['accuracy']
    )
    history_warmup = model.fit(
        train_ds, validation_data=val_ds,
        epochs=WARMUP_EPOCHS,
        class_weight=class_weights,
        callbacks=get_callbacks(model_name, phase='warmup'),
        verbose=1
    )
```

**`class_weight=class_weights`** — even after under-sampling + augmentation, there may be residual imbalance (some minority classes might still have fewer than THRESHOLD samples). Class weights computed by `compute_class_weight('balanced', ...)` give rare classes a higher loss multiplier:
```
weight_class_i = total_samples / (num_classes × count_of_class_i)
```
A class with 250 samples gets a weight of ~5× compared to a class with 500 samples.

```python
    # Phase 2: Fine-tuning — unfreeze top 20% of backbone
    # Rebuild model with freeze_base=False
    model, base, tl_params, fu_params = build_model(model_fn, model_name, freeze_base=False)
    # Load warmup weights into fine-tune model
    model.set_weights(warmup_weights)
    
    finetune_optimizer = optimizers.Adam(learning_rate=ft_schedule)
    model.compile(
        optimizer=finetune_optimizer,
        loss=focal_loss,
        metrics=['accuracy']
    )
    history_finetune = model.fit(
        train_ds, validation_data=val_ds,
        epochs=FINETUNE_EPOCHS,
        class_weight=class_weights,
        callbacks=get_callbacks(model_name, phase='finetune'),
        verbose=1
    )
```

**Why rebuild the model?** When you change `layer.trainable`, you must re-`compile()` for the change to take effect. The code rebuilds from scratch with `freeze_base=False` and then copies the warmup weights — this is the cleanest approach.

## Class Weights

```python
cw = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
cwd = dict(enumerate(cw))
```

`sklearn.utils.class_weight.compute_class_weight('balanced', ...)` computes:
```
w_i = n_total / (n_classes × n_i)
```
Where `n_i` = number of samples in class i.

Example: if class "blood_fresh" has 200 samples and class "normal_clean_mucosa" has 500 (after under-sampling), blood_fresh gets weight 500/200 = 2.5× higher.

This means a misclassification of a blood_fresh image hurts the loss 2.5× more than a misclassification of normal mucosa — forcing the model to pay more attention to rare diseases.

---

# 12. CALLBACKS — PHASE-AWARE TRAINING CONTROLS

```python
def get_callbacks(model_name, phase='warmup'):
    cbs = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7 if phase == 'warmup' else 10,
            min_delta=MIN_DELTA,
            restore_best_weights=True,
            mode='min', verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=f'best_{model_name}_{phase}.keras',
            monitor='val_loss',
            save_best_only=True, mode='min', verbose=0
        ),
    ]
    if phase == 'warmup':
        cbs.append(callbacks.ReduceLROnPlateau(...))
    return cbs
```

### EarlyStopping

**`monitor='val_loss'`** — watches the validation loss (not training loss). If we monitored training loss, we'd train until we memorise the training data.

**`patience=7` (warmup) / `patience=10` (finetune)**:
- If val_loss doesn't improve for this many consecutive epochs, stop training
- Warmup is shorter so we use patience=7; fine-tuning has more epochs so we give it patience=10 before giving up
- Prevents wasting compute when the model has already converged

**`min_delta=MIN_DELTA`** — the minimum improvement that counts as "improvement." If val_loss goes from 0.5000 to 0.4999, that's noise, not real improvement. `min_delta` (e.g., 0.001) prevents false-positive early stopping.

**`restore_best_weights=True`** — when training stops, **restore the weights from the epoch with the best val_loss** (not the final epoch's weights, which may be slightly worse due to overfitting). This is crucial — without this, you get the overfit model, not the best one.

### ModelCheckpoint

**`save_best_only=True`** — only overwrites the saved file when val_loss improves. Avoids disk thrashing and always has the best model on disk even if training crashes.

### ReduceLROnPlateau (Warmup Phase Only)

```python
callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=3,
    min_delta=MIN_DELTA,
    min_lr=1e-8, verbose=1
)
```

**`factor=0.3`**: When triggered, `new_lr = current_lr × 0.3`. For example, if LR is 5e-4 and loss plateaus: 5e-4 → 1.5e-4 → 4.5e-5 → 1.35e-5...

**Why not used in fine-tuning?** CosineDecay is a `LearningRateSchedule` object passed to the optimizer. TF/Keras does not allow `ReduceLROnPlateau` to override a schedule — raises a `TypeError`. The workaround is to manually reduce LR if needed (or just rely on cosine decay to handle it naturally).

---

# 13. TEST-TIME AUGMENTATION (TTA)

```python
def test_time_augmentation(model, test_ds, n_aug=TTA_PASSES):
    preds_list = []
    for _ in range(n_aug):
        tta_ds = (test_ds.unbatch()
                  .map(lambda img, lbl: augment_image(img, lbl), num_parallel_calls=2)
                  .batch(BATCH_SIZE).prefetch(PREFETCH_SIZE))
        preds_list.append(model.predict(tta_ds, verbose=0))
        del tta_ds
    preds_list.append(model.predict(test_ds, verbose=0))  # clean pass
    avg = np.mean(preds_list, axis=0)
    return avg
```

**What is TTA?**
Instead of predicting on each test image once, we:
1. Predict on the image after each of `n_aug` different random augmentations
2. Predict on the clean original image
3. Average all predictions

**Why does this improve accuracy?**
A single flip or rotation might move a diagnostic feature to the edge of the image. By averaging predictions across multiple augmented views, we get a more robust estimate of the true class probabilities. This typically improves accuracy by 1–3% at test time with no retraining.

**`n_aug = TTA_PASSES`** (2 in the code) — 2 augmented passes + 1 clean pass = 3 total predictions averaged. Higher would be better but uses 3× the evaluation time.

**Memory management**: `del tta_ds` after each pass, followed by `gc.collect()` — each augmented dataset holds a copy of the test data in the computation graph. Deleting it releases RAM before the next pass.

---

# 14. EVALUATION NOTEBOOK — COMPLETE BREAKDOWN

## Loading Results

```python
for model_name in ['EfficientNetB0', 'InceptionV3', 'ResNet101V2']:
    fname = f'{model_name}_results.pkl'
    data = pickle.load(io.BytesIO(uploaded[fname]))
    all_results.update(data['all_results'])
    CLASS_NAMES   = data['CLASS_NAMES']
    WARMUP_EPOCHS = data['WARMUP_EPOCHS']
```

Each `.pkl` file contains:
- `all_results`: dict with keys like `"EfficientNetB0 | Setting 1: Original"`, each containing `history`, `y_true`, `y_pred`, `metrics`
- `CLASS_NAMES`, `WARMUP_EPOCHS`, `WARMUP_LR`, `FINETUNE_LR` — metadata for plots

## Training Curves Plot

```python
axes[row, 0].axvline(x=WARMUP_EPOCHS, color='gray', linestyle='--', alpha=0.5, label='Phase switch')
```

The vertical dashed line at `WARMUP_EPOCHS` marks where Phase 1 (head warmup) ends and Phase 2 (backbone fine-tuning) begins. You typically see:
- **Before the line**: rapid initial improvement as the head learns
- **After the line**: slower, more refined improvement as the backbone adapts

## Confusion Matrix

```python
cm = confusion_matrix(result['y_true'], result['y_pred'])
cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=True, fmt='.2f', ...)
```

**Normalisation**: divides each row by the total number of true samples in that class. This converts raw counts to **recall rates** per class.

**Reading the matrix:**
- The diagonal shows what proportion of each true class was correctly predicted
- Off-diagonal entries show **where the model gets confused**
- E.g., if row="blood_fresh", column="normal_mucosa" = 0.15, it means 15% of actual bleeding cases were missed and called normal — a dangerous clinical error

**Why normalised?** Raw counts are misleading when class sizes differ. A class with 50 test samples getting 5 wrong looks better (90%) than a class with 500 getting 50 wrong (also 90%) but the raw counts are very different.

## Comparison Table

```python
comparison_df = pd.DataFrame(rows)
# Columns: Model, Setting, Accuracy, Precision, Recall, F1-Score
```

This 9-row table (3 models × 3 settings) is the project's key deliverable. Expected pattern:
- Setting 1 (Original): High accuracy (~85%), low minority class recall (~30%)
- Setting 2 (Under-Sampling): Lower accuracy (~78%), higher recall (~55%)
- Setting 3 (Full Pipeline): Best F1 (~72–80%), best recall on minority classes

## Performance Heatmap

```python
metrics_pivot = comparison_df.pivot_table(
    index='Model', columns='Setting', values='F1-Score')
sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd', ...)
```

Four heatmaps (one per metric) showing model × setting performance. Warmer colour = higher score. The heatmap lets you instantly see which model+setting combination dominates.

---

# 15. HYPERPARAMETER COMPARISON: ALL THREE NOTEBOOKS

| Hyperparameter | EfficientNetB0 | InceptionV3 | ResNet101V2 |
|---|---|---|---|
| BATCH_SIZE | 16 | 32 | 32 |
| THRESHOLD | 500 | 300 | 300 |
| WARMUP_EPOCHS | 10 | 5 | 5 |
| FINETUNE_EPOCHS | 20 | 10 | 10 |
| TOTAL_EPOCHS | 30 | 15 | 15 |
| WARMUP_LR | 5e-4 | 5e-4 | 5e-4 |
| FINETUNE_LR | 1e-4 | 1e-4 | 1e-4 |
| UNFREEZE_RATIO | 0.20 | 0.20 | 0.20 |

**Why does EfficientNetB0 train longer?**
It's the smallest model (5M params). Smaller models benefit more from additional epochs since they have less capacity and need more gradient updates to use their parameters efficiently. It also uses BATCH_SIZE=16 to avoid OOM, which means more gradient steps per epoch.

**Why do InceptionV3/ResNet use THRESHOLD=300?**
These are larger models. With THRESHOLD=500, the total dataset would be ~7,000 images. Training 24M or 45M parameter models for 15 epochs on 7K images is more prone to overfitting. Using 300 (4,200 total images) and relying on strong augmentation+dropout is a better strategy for the Colab 1-hour time budget.

---

# 16. MEDICAL METRICS EXPLAINED

## Accuracy
```
Accuracy = (TP + TN) / Total
```
**Limitation**: With 14 classes where one dominates 60% of images, predicting only that class gives 60% accuracy. Useless for medical diagnosis.

## Precision (Positive Predictive Value)
```
Precision_class_i = TP_i / (TP_i + FP_i)
```
"Of all images the model said are polyps, what fraction actually are polyps?"

High precision = few false alarms. Important for triage systems where false positives cause unnecessary follow-up procedures.

## Recall (Sensitivity, True Positive Rate)
```
Recall_class_i = TP_i / (TP_i + FN_i)
```
"Of all actual polyps, what fraction did the model find?"

**For medical diagnosis, recall is usually more important than precision.** Missing a cancer (FN) is typically worse than a false alarm (FP). You want recall close to 1.0 for dangerous conditions.

## F1-Score (Harmonic Mean of Precision and Recall)
```
F1_i = 2 × (Precision_i × Recall_i) / (Precision_i + Recall_i)
```
F1 balances both. The **harmonic mean** is more conservative than the arithmetic mean — a high F1 requires BOTH precision AND recall to be high. If precision=1.0 and recall=0.0: arithmetic mean = 0.5, harmonic mean (F1) = 0.0. F1 correctly penalises this.

## Macro vs Weighted Average (in classification_report)
- **Macro average**: averages metrics across classes with equal weight → penalises poor performance on minority classes
- **Weighted average**: weights by class size → dominated by majority classes

For imbalanced medical datasets, **macro F1** is the most honest metric.

---

# 17. RESULTS INTERPRETATION & ANALYSIS

## What to Write for the Short Analysis (8–10 Lines)

Based on the project's design:

**Setting 1 (Original)**: Training on the imbalanced dataset produces high overall accuracy but catastrophically low recall for minority classes like blood_fresh, polyp, and ulcerative_colitis. The model essentially learns to predict the majority class (normal_clean_mucosa) most of the time, demonstrating the danger of using accuracy alone as a metric in medical AI.

**Setting 2 (Under-Sampling)**: Capping majority classes to 300–500 images dramatically improves minority class recall (often doubling it) at the cost of 5–15% overall accuracy. The model now has a more balanced view of the training data. However, removing majority class data discards potentially informative samples.

**Setting 3 (Full Pipeline)**: Combining under-sampling with augmentation-based over-sampling achieves the best macro F1-Score across all models. The augmented minority class samples provide synthetic diversity, reducing the risk of overfitting that would occur from simple repetition. Online augmentation during training further regularises the model.

**Model Comparison**: ResNet101V2 typically achieves the highest accuracy (more parameters, deeper features) but trains slowest and is most prone to OOM. EfficientNetB0, despite being the smallest, often achieves competitive F1 due to its efficient compound scaling. InceptionV3 sits in the middle.

**Key Insight**: The combination of (1) data balancing, (2) focal loss with label smoothing, (3) class weights, and (4) two-phase transfer learning (warmup + cosine fine-tuning) creates a robust pipeline for imbalanced medical image classification.

---

# APPENDIX: Dataset Extraction Pipeline Explained

```python
with zipfile.ZipFile(ZIP_NAME, 'r') as z:
    z.extractall(EXTRACT_DIR)

for tar_path in sorted(glob.glob(os.path.join(TAR_DIR, '*.tar.gz'))):
    class_name = os.path.basename(tar_path).replace('.tar.gz', '')
    class_dir = os.path.join(IMAGE_DIR, class_name)
    with tarfile.open(tar_path, 'r:gz') as t:
        t.extractall(class_dir)
    
    # Flatten nested directories
    for root, dirs, fnames in os.walk(class_dir):
        for f in fnames:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                src = os.path.join(root, f)
                dst = os.path.join(class_dir, f)
                if src != dst:
                    shutil.move(src, dst)
```

**Why two-stage extraction?**
The Kvasir-Capsule dataset is distributed as a ZIP file containing multiple `.tar.gz` archives (one per disease class). This is common for large medical datasets shared via Google Drive.

`glob.glob(os.path.join(TAR_DIR, '*.tar.gz'))` — finds all compressed class archives.

The **flattening loop** handles the case where tar extraction creates nested subdirectories. For example, `blood_fresh.tar.gz` might extract to `blood_fresh/labelled_images/blood_fresh/*.jpg`. The loop moves all images to the top level of `blood_fresh/` so our counting and loading code works uniformly.

`sorted(glob.glob(...))` — processes archives in alphabetical order for reproducibility.

---

*This guide covers every line of code, every hyperparameter choice, every architectural decision, and every metric across all four notebooks of the WCE Classification project. Prepared for deep learning study.*
