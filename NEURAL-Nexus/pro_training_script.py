import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. SATELLITE SPECIFIC CONFIGURATION ---
BASE_PATH = "/kaggle/input/datasets/purvanshj009/boom-baam-dataset/"
if not os.path.exists(BASE_PATH):
    BASE_PATH = "./"

TRAIN_CSV = os.path.join(BASE_PATH, "TRAIN.csv")
TEST_CSV = os.path.join(BASE_PATH, "TEST.csv")
IMG_SIZE = (160, 160) # Higher resolution for satellite detail
BATCH_SIZE = 16 # Reduced to prevent OOM
EPOCHS = 20

def load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    # Contrast sharpening for blurry aerial data
    img = tf.image.per_image_standardization(img)
    img = tf.image.adjust_contrast(img, 1.5)
    return img, label

def load_test(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.image.per_image_standardization(img)
    img = tf.image.adjust_contrast(img, 1.5)
    return img

def build_satellite_model():
    # Use ResNet101V2 as a deep backbone for blurred satellite features
    base = applications.ResNet101V2(input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')
    base.trainable = True

    # Aggressive Augmentation for satellite orientation independence
    model = models.Sequential([
        layers.RandomFlip("both"),
        layers.RandomRotation(factor=1.0), # Full 0-360 range
        layers.RandomContrast(0.2),
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dense(15, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    print("🚀 Starting Satellite-Optimized Professional Pipeline...")
    
    # 2. Data Loading
    df = pd.read_csv(TRAIN_CSV)
    df['ABS_PATH'] = df['IMAGE'].apply(lambda x: os.path.join(BASE_PATH, x))
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['LABEL'])
    
    train_ds = tf.data.Dataset.from_tensor_slices((train_df['ABS_PATH'], train_df['LABEL']))
    train_ds = train_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((val_df['ABS_PATH'], val_df['LABEL']))
    val_ds = val_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # 3. Training
    model = build_satellite_model()
    early_stop = callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True)
    lr_reduce = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[early_stop, lr_reduce])

    # 4. Evaluation
    print("\n📊 Final Satellite Performance Evaluation...")
    y_val_pred = np.argmax(model.predict(val_ds), axis=1)
    y_val_true = val_df['LABEL'].values
    print(classification_report(y_val_true, y_val_pred))
    
    # 5. Prediction on Test Set
    print("\n🔍 Generating FINAL.csv for Satellite Data...")
    test_df = pd.read_csv(TEST_CSV)
    test_df['ABS_PATH'] = test_df['IMAGE'].apply(lambda x: os.path.join(BASE_PATH, x))
    test_ds = tf.data.Dataset.from_tensor_slices(test_df['ABS_PATH']).map(load_test).batch(BATCH_SIZE)
    
    test_preds = np.argmax(model.predict(test_ds), axis=1)
    final_df = pd.DataFrame({'IMAGE': test_df['IMAGE'], 'LABEL': test_preds})
    final_df.to_csv("FINAL.csv", index=False)
    print("✅ FINAL.csv created successfully.")

    model.save("pro_satellite_model.h5")

if __name__ == "__main__":
    main()
