import sys, os
sys.path.append('/kaggle/input/efficientnet-keras-dataset/efficientnet_kaggle')
#!pip install -q /kaggle/input/tensorflow-extra-lib-ds/tensorflow_extra-1.0.2-py3-none-any.whl --no-deps

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)
import os
import pandas as pd
import numpy as np
import random
from glob import glob
from tqdm import tqdm
tqdm.pandas()
import gc
import librosa
import sklearn
import time
import glob
import shutil

import tensorflow as tf
tf.config.optimizer.set_jit(True)

import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

df = pd.read_csv('/kaggle/input/birdclef-2024/train_metadata.csv')

class CFG:
    debug = False
    verbose = 0
    device = 'CPU'
    seed = 42
    img_size = [128, 384]
    batch_size = 16
    infer_bs = 2
    tta = 1
    drop_remainder = True
    duration = 5 
    train_duration = 10
    sample_rate = 32000
    downsample = 1
    audio_len = duration*sample_rate
    nfft = 2028
    window = 2048
    hop_length = train_duration*32000 // (img_size[1] - 1)
    fmin = 20
    fmax = 16000
    normalize = True
    class_names = sorted(os.listdir('/kaggle/input/birdclef-2024/train_audio/'))
    num_classes = len(class_names)
    class_labels = list(range(num_classes))
    label2name = dict(zip(class_labels, class_names))
    name2label = {v:k for k,v in label2name.items()}
    target_col = ['target']
    tab_cols = ['filename','common_name','rate']

def load_audio(filepath, sr=32000, normalize=True):
    audio, orig_sr = librosa.load(filepath, sr=None)
    if sr != orig_sr:
        audio = librosa.resample(audio, orig_sr, sr)
    audio = audio.astype('float32').ravel()
    if normalize:
        mean = audio.mean()
        std = audio.std()
        audio = (audio - mean) / std if std > 0 else audio
    return audio, sr

@tf.function(jit_compile=True)
def MakeFrame(audio, duration=5, sr=32000):
    frame_length = int(duration * sr)
    frame_step = int(duration * sr)
    chunks = tf.signal.frame(audio, frame_length, frame_step, pad_end=True)
    return chunks

def extract_features(audio, sr):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)[0]
    harmonic, percussive = librosa.effects.hpss(audio)
    features = np.hstack([np.mean(mfccs, axis=1), 
                          np.mean(spectral_contrast, axis=1),
                          np.mean(chroma, axis=1),
                          np.mean(zero_crossing_rate),
                          np.mean(harmonic),
                          np.mean(percussive)])
    return features

def create_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(train_data, val_data, model, epochs=90):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        history = model.fit(train_data, validation_data=val_data, epochs=1, verbose=1)
        train_loss, train_acc = history.history['loss'][0], history.history['accuracy'][0]
        val_loss, val_acc = history.history['val_loss'][0], history.history['val_accuracy'][0]
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

def predict_and_ensemble(test_df, load_audio, MakeFrame, CKPTS, NUM_CKPTS):
    tick = time.time()
    ids = []
    preds1 = np.empty(shape=(0, CFG.num_classes), dtype=np.float32)

    for filepath in tqdm(test_df.filepath.tolist(), 'testing'):
        filename = filepath.split('/')[-1].replace('.ogg', '')
        audio, sr = load_audio(filepath)

        if audio.size == 0:
            continue

        features = extract_features(audio, sr)
        chunks = MakeFrame(audio)  # Ensure this is compatible with new feature dimensions

        chunk_preds = np.zeros(shape=(len(chunks), CFG.num_classes), dtype=np.float32)
        for model in CKPTS[:NUM_CKPTS]:
            rec_preds = model.predict(chunks)
            chunk_preds += rec_preds / NUM_CKPTS

        rec_ids = [f'{filename}_{(frame_id+1)*5}' for frame_id in range(len(chunks))]
        ids += rec_ids
        preds1 = np.concatenate([preds1, chunk_preds], axis=0)

    tock = time.time()
    execution_time = tock - tick
    print(f">> Time for submission: ~ {execution_time} seconds")

    return ids, preds1, execution_time

CKPT_DIR = '/kaggle/input/birdclef24-pretraining-train-model'
CKPT_PATHS = sorted([x for x in glob.glob(f'{CKPT_DIR}/fold-*keras')])
print("Checkpoints: ", CKPT_PATHS)

WRITABLE_DIR = '/kaggle/working/'
if not os.path.exists(WRITABLE_DIR):
    os.makedirs(WRITABLE_DIR)

for ckpt_path in CKPT_PATHS:
    shutil.copy(ckpt_path, WRITABLE_DIR)

CKPT_PATHS = sorted([f'{WRITABLE_DIR}/{os.path.basename(x)}' for x in glob.glob(f'{CKPT_DIR}/fold-*keras')])
CKPTS = [tf.keras.models.load_model(x, compile=False) for x in tqdm(CKPT_PATHS, desc="Loading ckpts")]
NUM_CKPTS = 1

# Placeholder for train and validation data
# Assuming X_train, y_train, X_val, y_val are already defined
# You need to replace the following lines with your actual data preparation
# Here we just create dummy data for illustration purposes
X_train = np.random.rand(100, 128, 384, 1)  # Replace with actual training data
y_train = np.random.randint(0, CFG.num_classes, 100)  # Replace with actual training labels
X_val = np.random.rand(20, 128, 384, 1)  # Replace with actual validation data
y_val = np.random.randint(0, CFG.num_classes, 20)  # Replace with actual validation labels

# Create TensorFlow datasets
train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(CFG.batch_size)
val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(CFG.batch_size)

# Create and train model
input_shape = (CFG.img_size[0], CFG.img_size[1], 1)  # Adjust based on your data
model = create_model(input_shape, CFG.num_classes)
train_model(train_data, val_data, model, epochs=90)

test_audio_dir = '/kaggle/input/birdclef-2024/test_soundscapes/'
test_paths = [test_audio_dir+f for f in sorted(os.listdir(test_audio_dir))]
if len(test_paths) == 1:
    test_audio_dir = '/kaggle/input/birdclef-2024/unlabeled_soundscapes/'
    test_paths = [test_audio_dir+f for f in sorted(os.listdir(test_audio_dir))][:2]

test_df = pd.DataFrame(test_paths, columns=['filepath'])
test_df['filename'] = test_df.filepath.map(lambda x: x.split('/')[-1].replace('.ogg',''))

ids, preds1, execution_time = predict_and_ensemble(test_df, load_audio, MakeFrame, CKPTS, NUM_CKPTS)

print("IDs:", ids)
print("Predictions:", preds1)
print("Execution Time:", execution_time)

pred_df = pd.DataFrame(ids, columns=['row_id'])
pred_df.loc[:, CFG.class_names] = preds1
pred_df.to_csv('submission.csv', index=False)
pred_df
