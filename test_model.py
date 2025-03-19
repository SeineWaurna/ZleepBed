import serial
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('best_model.h5') 

arduino = serial.Serial('COM6', 9600)

while True:
    message = arduino.readline().decode().replace("\n","")
    values = [x.strip() for x in message.split(",")]
    if len(values) == 3:
        try:
            input_values = np.array([float(val) for val in values]).reshape(1, -1)
            predictions = model.predict(input_values, verbose=0)

            print(f"Probabilities: {predictions[0]}")
        except ValueError:
            print("Error in conversion. Please check the data format.")
