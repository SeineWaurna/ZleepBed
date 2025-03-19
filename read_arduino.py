import serial

arduino = serial.Serial('COM6', 9600)

while True:
    message = arduino.readline().decode().replace("\n","")
    values = [x for x in message.split(",")]
    if len(values) == 3:
        print(f"value_0 = {values[0]}, value_1 = {values[1]}, value_2 = {values[2]}")



