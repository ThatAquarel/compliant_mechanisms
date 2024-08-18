import time

import serial
import struct


class commands:
    NUL = 0x00

    SYN = 0xAB
    ACK = 0x4B
    NAK = 0x5A
    ERR = 0x3C

    RESET = 0x01
    HOME_CARRIAGE = 0x02
    HOME_SERVO = 0x03
    MOVE_CARRIAGE = 0x04
    MOVE_SERVO = 0x05


class packet_t:
    cmd = 0
    buffer = 0

    def __init__(self, buffer=b"", cmd=commands.NUL):
        self.buffer = buffer
        self.cmd = cmd


def send_packet(ser_handle, packet: packet_t):
    packet_bytes = struct.pack(
        f"<xBBB{len(packet.buffer)}sBx",
        0x7E,  # U8 start of packet flag
        packet.cmd & 0xFF,  # U8 command
        len(packet.buffer) & 0xFF,  # U8 length of payload
        packet.buffer,  # U8 payload[len]
        0x7D,  # U8 end of packet flag
    )

    # print(packet_bytes)
    ser_handle.write(packet_bytes)


def recv_packet(ser_handle):
    ser_handle.read_until(b"\x7E")

    command, length = struct.unpack("<cB", ser_handle.read(2))
    payload = b""
    if length:
        payload = struct.unpack(f"<{length}s", ser_handle.read(length))

    if ser_handle.read(1) != b"\x7D":
        print("serial frame end error")

    return packet_t(payload, ord(command))


ser = serial.Serial(port="COM6", baudrate=115200, dsrdtr=True)

# SYN
# send_packet(ser, packet_t(cmd=commands.SYN))


def reset():
    send_packet(ser, packet_t(cmd=commands.RESET))
    print(recv_packet(ser).cmd)

    send_packet(ser, packet_t(cmd=commands.HOME_CARRIAGE))
    print(recv_packet(ser).cmd)

    send_packet(ser, packet_t(cmd=commands.HOME_SERVO))
    print(recv_packet(ser).cmd)


def wait_ready_move():
    while True:
        send_packet(
            ser,
            packet_t(buffer=struct.pack("<f", 0), cmd=commands.MOVE_CARRIAGE),
        )

        if recv_packet(ser).cmd == commands.ACK:
            break

        time.sleep(0.001)


def move(mm):
    steps_per_mm = 1000 / (60 * 2)
    send_packet(
        ser,
        packet_t(
            buffer=struct.pack("<f", mm * steps_per_mm), cmd=commands.MOVE_CARRIAGE
        ),
    )

    recv_packet(ser)


def s_move(deg):
    send_packet(
        ser,
        packet_t(
            buffer=struct.pack(
                "<9B",
                *np.array(deg, dtype=np.uint8),
            ),
            cmd=commands.MOVE_SERVO,
        ),
    )
    recv_packet(ser)


def s_move_interp(x1, x2, dt, steps=100):
    dx = x2 - x1
    for i in np.linspace(0, 1, steps):
        s_move(x1 + dx * i)
        time.sleep(dt / steps)


import numpy as np


reset()
wait_ready_move()
move(100)
input()

x_max = np.array([180, 0, 180, 84, 0, 180, 0, 180, 180], dtype=int)
x_min = np.array([0, 180, 132, 47, 180, 0, 180, 0, 36], dtype=int)
y_max = np.array([0, 137, 180, 0, 141, 180, 138, 168, 0], dtype=int)
y_min = np.array([180, 157, 0, 180, 107, 0, 28, 0, 180], dtype=int)
z_max = np.array([180, 180, 180, 180, 180, 180, 180, 180, 180], dtype=int)
z_min = np.zeros(9, dtype=int)

z_max_0 = np.array([180, 180, 180, 0, 0, 0, 0, 0, 0], dtype=int)
z_max_1 = np.array([180, 180, 180, 180, 180, 180, 0, 0, 0], dtype=int)
move(280)
s_move_interp(zeros, target, 2)

input()

move(0)
s_move_interp(target, zeros, 2)
reset()

ser.close()
