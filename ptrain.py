import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint

to_categorical = tf.keras.utils.to_categorical

# Define dataset path
DATA_PATH = os.path.join('P_Data')

# Define actions (ensure they match the ones used in keypoint collection)
actions = np.array(['Hello', 'Good', 'Morning','Help','House','Thankyou','Nice','Welcome','Yes','No'])

# Number of sequences and frames per sequence
no_sequences = 30  # Increased from 30 to 50
sequence_length = 30  # Increased from 30 to 40

# Create a label mapping dictionary
label_map = {label: num for num, label in enumerate(actions)}

# Initialize sequences and labels
sequences, labels = [], []

# Load collected keypoints
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            npy_path = os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num))
            res = np.load(npy_path)
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

# Convert to NumPy arrays
X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Use StratifiedShuffleSplit to ensure balanced train-test split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in sss.split(X, np.argmax(y, axis=1)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

# Define the improved LSTM model
model = Sequential([
    LSTM(128, return_sequences=True, activation='relu', input_shape=(sequence_length, X.shape[2])),
    BatchNormalization(),  # Helps stabilize training
    Dropout(0.3),

    LSTM(128, return_sequences=True, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    LSTM(64, return_sequences=False, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(len(actions), activation='softmax')
])

# Compile the model with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Define checkpoint to save the best model
checkpoint_path = "P2_model.h5"
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1)

# Train the model for more epochs
history = model.fit(X_train, y_train, epochs=290, batch_size=16, validation_data=(X_test, y_test), callbacks=[checkpoint])

# Load the best model and evaluate on the test set
model.load_weights(checkpoint_path)
loss, acc = model.evaluate(X_test, y_test, verbose=0)

# Calculate overall train accuracy (average across all epochs)
overall_train_acc = np.mean(history.history['accuracy'])  # Mean of training accuracy across all epochs
overall_val_acc = np.mean(history.history['val_accuracy'])  # Mean of validation accuracy across all epochs

# Print evaluation report
print("\n=== Model Evaluation Report ===")
print(f"Overall Train Accuracy : {overall_train_acc * 100:.2f}%")
print(f"Final Test Accuracy    : {acc * 100:.2f}%")
print(f"Final Test Loss        : {loss:.4f}")
print("===================================")
