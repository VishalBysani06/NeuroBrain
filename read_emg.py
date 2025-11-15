import serial
import csv
import time

port = 'COM6'   # change if needed
baud = 9600

arduino = serial.Serial(port, baud)
time.sleep(2)


filename = r"C:\Users\abhib\Desktop\Neuro Brain\emg_output.csv"

with open(filename, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["time", "emg_value"])

    print("Recording... Press CTRL + C to stop")

    start = time.time()

    try:
        while True:
            data = arduino.readline().decode().strip()
            if data.isdigit():
                t = time.time() - start
                writer.writerow([t, data])
                print(f"{t:.3f} sec  →  {data}")
    except KeyboardInterrupt:
        print("Stopped recording.")
