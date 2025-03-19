import argparse
import serial
import csv
import keyboard
import time
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.patches as patches

parser = argparse.ArgumentParser(description='Log sensor data and visualize in real-time.')
parser.add_argument('--save_path', type=str, required=True, help='Path to save the CSV file.')
args = parser.parse_args()

filename = args.save_path

arduino = serial.Serial('COM3', 9600)
arduino.flushInput()

with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["class", "value0", "value1", "value2"])

def log_to_csv(class_id, values):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([class_id, *values])
    print(f"Logged data: Class={class_id}, Values={values}")

def value_to_proportion(value):
    if value < 0: value = 0
    if value > 1023: value = 1023
    return (1023 - value) / 1023

fig, ax = plt.subplots()
squares = [
    patches.Rectangle((0.1, 0.6), 0.2, 0.2, color='grey'), 
    patches.Rectangle((0.4, 0.2), 0.2, 0.2, color='grey'), 
    patches.Rectangle((0.7, 0.6), 0.2, 0.2, color='grey')
]
for square in squares:
    ax.add_patch(square)
sensor_labels = ['sensor1', 'sensor2', 'sensor3']
texts = []
for i, square in enumerate(squares):
    ax.text(square.get_x() + square.get_width() / 2, square.get_y() + square.get_height() + 0.05, sensor_labels[i], ha='center', va='bottom')
    texts.append(ax.text(square.get_x() + square.get_width() / 2, square.get_y() + square.get_height() / 2, '', ha='center', va='center'))

def update(frame):
    if keyboard.is_pressed('q'):
        plt.close(fig)
        sys.exit(0)
    
    message = arduino.readline().decode().replace("\n","")
    values = message.split(",")
    if len(values) == 3:
        values = [int(x) for x in values]
        scaled_values = [value_to_proportion(val) for val in values]
        for i, (text, square) in enumerate(zip(texts, squares)):
            proportion = scaled_values[i]
            color = (proportion, 1-proportion, 0)
            square.set_color(color)
            text.set_text(f'{proportion:.2f}')
            
        for i in range(4):
            if keyboard.is_pressed(str(i)):
                log_to_csv(i, scaled_values)
                time.sleep(0.2)
                arduino.flushInput()
                break  

ani = FuncAnimation(fig, update, interval=70, cache_frame_data=False)
plt.show()
