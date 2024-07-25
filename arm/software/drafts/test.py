import serial
import time

ser = serial.Serial("COM6", 9600)
ser.flush()

time.sleep(1)

# print(ser.write(b"\x00\x00\x7e\xab\x00\x7d\x00\x00"))
print(ser.write(b"\x00\x00~\xab\x00}\x00"))

time.sleep(1)

print(ser.read_all())
