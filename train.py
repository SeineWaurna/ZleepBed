import argparse
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

parser = argparse.ArgumentParser(description='Train a neural network with provided data.')
parser.add_argument('--model_name', type=str, required=True, help='Name of the saved model.')
parser.add_argument('--data_path', type=str, required=True, help='Path to the data zip file.')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training the model.')
args = parser.parse_args()

data_path = args.data_path
model_name = args.model_name
learning_rate = args.lr

train_data = pd.read_csv(f'{data_path}/train.csv')
val_data = pd.read_csv(f'{data_path}/val.csv')
test_data = pd.read_csv(f'{data_path}/test.csv')

x_train, y_train = train_data.iloc[:, 1:], train_data.iloc[:, 0]
x_val, y_val = val_data.iloc[:, 1:], val_data.iloc[:, 0]
x_test, y_test = test_data.iloc[:, 1:], test_data.iloc[:, 0]

y_train = to_categorical(y_train, num_classes=4)
y_val = to_categorical(y_val, num_classes=4)
y_test = to_categorical(y_test, num_classes=4)

model = Sequential([
    Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
    Dropout(0.8),
    Dense(4, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=500, verbose=1, mode='min')
mc = tf.keras.callbacks.ModelCheckpoint(model_name + '.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100000, batch_size=32, callbacks=[es, mc])

model = tf.keras.models.load_model(model_name + '.h5')

for (x_data, y_data), dataset_name in [((x_train, y_train), 'Train'), ((x_val, y_val), 'Validation'), ((x_test, y_test), 'Test')]:
    loss, acc = model.evaluate(x_data, y_data, verbose=0)
    print(f"{dataset_name} Accuracy: {acc}")

    y_pred = model.predict(x_data)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_data, axis=1)

    cm = confusion_matrix(y_true_classes, y_pred_classes)
    clr = classification_report(y_true_classes, y_pred_classes, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3'], zero_division=0)

    print(f"{dataset_name} Confusion Matrix:\n", cm)
    print(f"{dataset_name} Classification Report:\n", clr)
