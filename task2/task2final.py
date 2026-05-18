import os
import pandas as pd
from pathlib import Path
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

#%%
# --- STAP 1: HET EXACTE PAD ---
# Use path relative to this script's location
BASE_DIR = Path(__file__).resolve().parent.parent
JSON_PATH = BASE_DIR / "data" / "minifigs.json"
IMAGE_DIR = BASE_DIR / "data" / "images"

#%%
# --- STAP 2: DATA INLADEN ---
if not JSON_PATH.exists():
    raise FileNotFoundError(f"❌ ERROR: Bestand niet gevonden op {JSON_PATH}")

df = pd.read_json(JSON_PATH)
df['abs_path'] = df['minifig_number'].apply(lambda x: str(IMAGE_DIR / f"{x}.jpg"))

# Check of de foto's er echt staan
df = df[df['abs_path'].apply(os.path.exists)].reset_index(drop=True)
print(f"✅ Geladen: {len(df)} foto's gevonden op de schijf.")

#%%
# --- STAP 3: TOP 40 SELECTIE ---
TOP_K = 40
counts = df['category'].value_counts()
valid_categories = counts.head(TOP_K).index.tolist()
df_clean = df[df['category'].isin(valid_categories)].copy()

print(f"✅ Top 40 geselecteerd. Totaal {len(df_clean)} afbeeldingen.")

#%%
# --- STAP 4: DATA SPLIT & GENERATORS ---
# De Split (70% train, 15% val, 15% test) met 'stratify' voor perfecte balans
train_df_clean, temp_df = train_test_split(
    df_clean, test_size=0.30, stratify=df_clean['category'], random_state=42
)
val_df_clean, test_df_clean = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df['category'], random_state=42
)

# Setup Augmentatie (Zonder rescale, EfficientNetV2 regelt dit intern)
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],  # FIX: Komma toegevoegd!
    horizontal_flip=True,
    zoom_range=0.2,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator()

IMG_SIZE = (384, 384)
BATCH_SIZE = 16 

print("Loading Generators...")
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df_clean, x_col="abs_path", y_col="category",
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)
val_generator = test_datagen.flow_from_dataframe(
    dataframe=val_df_clean, x_col="abs_path", y_col="category",
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df_clean, x_col="abs_path", y_col="category",
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical",
    shuffle=False
)

#%%
# --- STAP 5: MODEL CONFIGURATIE & PHASE 1 ---
class_indices = train_generator.classes
weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(class_indices),
    y=class_indices
)
class_weight_dict = dict(enumerate(weights))

base_model = tf.keras.applications.EfficientNetV2S(
    weights='imagenet', include_top=False, input_shape=(384, 384, 3)
)
base_model.trainable = False 

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.4),
    layers.BatchNormalization(),
    layers.Dense(40, activation='softmax')
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

checkpoint_path = os.path.join(BASE_DIR, 'efficientnet_lego_top40.keras')
callbacks = [
    ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss'),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)
]

print("Starting Phase 1: Training the head...")
history_phase1 = model.fit(
    train_generator, validation_data=val_generator,
    epochs=10, class_weight=class_weight_dict, callbacks=callbacks
)

#%%
# --- STAP 6: PHASE 2 - FINE-TUNING ---
model.save(os.path.join(BASE_DIR, 'model_phase1_complete.keras'))
base_model.trainable = True

# Bevries de eerste lagen, train alleen de bovenste ~100 lagen
fine_tune_at = len(base_model.layers) - 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

print("Starting Phase 2: Fine-tuning...")
history_phase2 = model.fit(
    train_generator, validation_data=val_generator,
    epochs=10, class_weight=class_weight_dict, callbacks=callbacks
)

#%%
# --- STAP 7: SMOOTHED WEIGHTS CORRECTIE ---
balanced_weights = compute_class_weight(
    class_weight='balanced', classes=np.unique(class_indices), y=class_indices
)
smoothed_weights = np.sqrt(balanced_weights)
smoothed_weight_dict = dict(enumerate(smoothed_weights))

# Herlaad beste model uit Phase 2 en compileer extreem laag
model = tf.keras.models.load_model(checkpoint_path)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

print("Starting Correction Phase with smoothed weights...")
model.fit(train_generator, validation_data=val_generator, epochs=3, class_weight=smoothed_weight_dict)
model.save(os.path.join(BASE_DIR, 'final_model_smoothed.keras'))

#%%
# --- STAP 8: EVALUATIE & PLOTS ---
y_pred_probs_smoothed = model.predict(test_generator)
y_pred_smoothed = np.argmax(y_pred_probs_smoothed, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

print("\n--- FINALE RESULTATEN (MET SMOOTHED WEIGHTS) ---")
print(classification_report(y_true, y_pred_smoothed, target_names=class_labels))

cm = confusion_matrix(y_true, y_pred_smoothed)
plt.figure(figsize=(20, 15))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Waar raakt het model in de war?')
plt.xlabel('Voorspeld')
plt.ylabel('Echt')
plt.show()

#%%
# --- STAP 9: GRAD-CAM INTERPRETABILITY ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    base_model = model.layers[0]
    target_layer = base_model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(base_model.inputs, target_layer.output)

    with tf.GradientTape() as tape:
        conv_outputs = grad_model(img_array)
        tape.watch(conv_outputs)
        x = conv_outputs
        for layer in model.layers[1:]:
            x = layer(x)
        preds = x
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def get_colab_style_img(img_path, heatmap):
    img_pil = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img_pil).astype(np.float32)

    heatmap_uint8 = np.uint8(255 * heatmap)
    thresh = np.percentile(heatmap_uint8, 60)
    heatmap_uint8[heatmap_uint8 < thresh] = 0
    
    colormap = cm.get_cmap("inferno")
    heatmap_colored = colormap(heatmap_uint8 / 255.0)[:, :, :3]
    heatmap_colored = np.uint8(255 * heatmap_colored)
    
    heatmap_img = Image.fromarray(heatmap_colored).resize(IMG_SIZE, resample=Image.BICUBIC)
    heatmap_res = np.array(heatmap_img).astype(np.float32)

    superimposed = (heatmap_res * 0.55) + (img_array * 0.7)
    return Image.fromarray(np.clip(superimposed, 0, 255).astype('uint8')), img_pil

def predict_and_visualize_final(test_df, model, n=10):
    _ = model(np.zeros((1, 384, 384, 3)))
    sample_indices = random.sample(range(len(test_df)), n)
    fig, axes = plt.subplots(n, 3, figsize=(14, n * 4.5))
    fig.suptitle("Grad-CAM: Origineel | Heatmap | Voorspelling vs. Werkelijkheid", fontsize=16, y=1.01)

    for row, idx in enumerate(sample_indices):
        img_path = test_df['abs_path'].iloc[idx]
        true_label = test_df['category'].iloc[idx]

        img_load = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
        img_array_exp = np.expand_dims(tf.keras.utils.img_to_array(img_load), axis=0)

        preds = model.predict(img_array_exp, verbose=0)
        pred_idx = np.argmax(preds[0])
        pred_label = CLASS_NAMES[pred_idx]
        confidence = preds[0][pred_idx] * 100
        correct = (pred_label == true_label)

        try:
            heatmap = make_gradcam_heatmap(img_array_exp, model, LAST_CONV_LAYER_NAME, pred_index=pred_idx)
            superimposed, original = get_colab_style_img(img_path, heatmap)
            axes[row, 0].imshow(original)
            axes[row, 1].imshow(superimposed)
        except Exception as e:
            print(f"Error bij {img_path}: {e}")

        axes[row, 0].set_title(f"Echt: {true_label}", fontsize=10)
        axes[row, 1].set_title("Grad-CAM (Inferno)", fontsize=10)
        axes[row, 2].axis('off')
        color = '#27ae60' if correct else '#c0392b'
        status_text = "CORRECT" if correct else "FOUT"
        
        axes[row, 2].text(
            0.5, 0.5, f"Voorspeld:\n{pred_label}\n\nZekerheid: {confidence:.1f}%\n\nResultaat: {status_text}",
            ha='center', va='center', fontsize=11, color=color, fontweight='bold', transform=axes[row, 2].transAxes,
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#ffffcc', edgecolor=color, linewidth=2)
        )
        for ax in axes[row, :2]: ax.axis('off')

    plt.tight_layout()
    plt.show()

predict_and_visualize_final(test_df_clean, model, n=10)