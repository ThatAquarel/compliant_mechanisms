#include <Arduino.h>
#include <Servo.h>

// carriage
#define CARRIAGE_INT_PIN 18
#define CARRIAGE_NEG_PIN 44
#define CARRIAGE_POS_PIN 46

#define ENC_A_PIN 19
#define ENC_B_PIN 20
#define ENC_B_PORTD (1 << 1)

#define KP 5.0
#define KI 0.00005
#define KD 0.01

volatile boolean carriage_homing = false;
volatile boolean carriage_ready = false;
volatile float enc_step = 0;
float setpoint = 0;
float prev_dx = 0, integral = 0;
unsigned long prev_t = 0;

// servo
#define N_LAYERS 3
#define N_SIDES 3

#define n_servos (N_LAYERS * N_SIDES)

const int pins[n_servos] = {
    36, 37, 38,
    39, 40, 41,
    42, 43, 45};
const int trim_min[n_servos] = {
    544, 544, 544,
    544, 544, 544,
    544, 544, 544};
const int trim_max[n_servos] = {
    2400, 2400, 2400,
    2400, 2400, 2400,
    2400, 2400, 2400};
const int home_angle[n_servos] = {
    90, 90, 90,
    90, 90, 90,
    90, 90, 90};

Servo servos[n_servos];

// serial communications
#define SERIAL_BUFFER_SIZE 64
#define START_FLAG 0x7E
#define END_FLAG 0x7D

#define SYN 0xAB
#define ACK 0x4B
#define NAK 0x5A
#define ERR 0x3C

#define RESET 0x01
#define HOME_CARRIAGE 0x02
#define HOME_SERVO 0x03
#define MOVE_CARRIAGE 0x04
#define MOVE_SERVO 0x05

uint8_t serial_buffer[SERIAL_BUFFER_SIZE];
byte n = 0;
byte packet_status = 0;
uint8_t packet_cmd, packet_len;
boolean new_packet = false;

// prototypes
void enc_isr();
void carriage_reset_isr();
void carriage_motor_set(float value);
void carriage_motor_stop();

void process_pid();
void process_cmd();

void recv_packet();
void send_packet(uint8_t cmd);

void setup()
{
  pinMode(LED_BUILTIN, OUTPUT);

  // carriage
  // optical homing sensor
  pinMode(CARRIAGE_INT_PIN, INPUT);
  attachInterrupt(digitalPinToInterrupt(CARRIAGE_INT_PIN), carriage_reset_isr, FALLING);

  // rotary encoder
  pinMode(ENC_A_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENC_A_PIN), enc_isr, RISING);
  pinMode(ENC_B_PIN, INPUT_PULLUP);

  // motor controller
  pinMode(CARRIAGE_NEG_PIN, OUTPUT);
  pinMode(CARRIAGE_POS_PIN, OUTPUT);

  // servos
  for (int i = 0; i < n_servos; i++)
  {
    servos[i].attach(pins[i], trim_min[i], trim_max[i]);
  }

  // serial
  Serial.begin(115200);
}

void enc_isr()
{
  if (PIND & ENC_B_PORTD)
  {
    enc_step++;
  }
  else
  {
    enc_step--;
  }
}

void carriage_reset_isr()
{
  if (!carriage_homing)
  {
    return;
  }

  cli();

  carriage_motor_stop();
  enc_step = 0;
  setpoint = 0;
  carriage_homing = false;
  carriage_ready = true;

  sei();
}

void loop()
{
  if (carriage_ready && !carriage_homing)
  {
    process_pid();
  }

  recv_packet();
  if (new_packet)
  {
    process_cmd();
    new_packet = false;
  }
}

void process_pid()
{
  unsigned long current_t = micros();
  float dt = (float)(current_t - prev_t) / 1000000.0;
  prev_t = current_t;

  float dx = setpoint - enc_step;
  integral += dx * dt;
  float derivative = (dx - prev_dx) / dt;
  float output = KP * dx + KI * integral + KD * derivative;
  prev_dx = dx;

  carriage_motor_set(output);
}

void carriage_motor_set(float value)
{
  float out = abs(value);
  out = min(out, 255);

  if (out < 15)
  {
    carriage_motor_stop();
    return;
  }

  if (value > 0)
  {
    analogWrite(CARRIAGE_NEG_PIN, 0);
    analogWrite(CARRIAGE_POS_PIN, out);
  }
  else
  {
    analogWrite(CARRIAGE_POS_PIN, 0);
    analogWrite(CARRIAGE_NEG_PIN, out);
  }
}

void carriage_motor_stop()
{
  analogWrite(CARRIAGE_POS_PIN, 0);
  analogWrite(CARRIAGE_NEG_PIN, 0);
}

void process_cmd()
{
  switch (packet_cmd)
  {
  case SYN:
    send_packet(ACK);
    break;
  case RESET:
    cli();
    carriage_motor_stop();
    carriage_ready = false;
    carriage_homing = false;
    sei();
    send_packet(ACK);
    break;
  case HOME_CARRIAGE:
    if (carriage_homing)
    {
      send_packet(ERR);
      break;
    }
    carriage_homing = true;
    if (digitalRead(CARRIAGE_INT_PIN))
    {
      analogWrite(CARRIAGE_NEG_PIN, 255);
    }
    else
    {
      carriage_reset_isr();
    }
    send_packet(ACK);
    break;
  case HOME_SERVO:
    for (int i = 0; i < n_servos; i++)
    {
      servos[i].write(home_angle[i]);
    }
    send_packet(ACK);
    break;
  case MOVE_CARRIAGE:
    if (!carriage_ready || carriage_homing)
    {
      send_packet(ERR);
      break;
    }
    if (n != sizeof(setpoint))
    {
      send_packet(NAK);
      break;
    }
    cli();
    memcpy(&setpoint, serial_buffer, sizeof(setpoint));
    sei();
    send_packet(ACK);
    break;
  case MOVE_SERVO:
    if (n != n_servos * sizeof(uint8_t))
    {
      send_packet(NAK);
      break;
    }
    for (int i = 0; i < n_servos; i++) {
      servos[i].write((int) serial_buffer[i]);
    }
    send_packet(ACK);
    break;
  default:
    send_packet(NAK);
    break;
  }
}

void recv_packet()
{
  // 0: await start flag  uint8
  // 1: command           uint8
  // 2: len               uint8
  // 3: buffer            uint8[len]
  // 4: end flag          uint8

  if (Serial.available() > 1)
  {
    uint8_t rc;
    rc = Serial.read();

    switch (packet_status)
    {
    case 0:
      if (rc == START_FLAG)
      {
        packet_status = 1;
        n = 0;
      }
      break;
    case 1:
      packet_cmd = rc;
      packet_status = 2;
      break;
    case 2:
      packet_len = rc;
      if (packet_len == 0)
      {
        packet_status = 4;
      }
      else
      {
        packet_status = 3;
      }
      break;
    case 3:
      serial_buffer[n] = rc;
      n++;
      if (n >= SERIAL_BUFFER_SIZE)
      {
        n = 0;
      }
      packet_len--;
      if (packet_len == 0)
      {
        packet_status = 4;
      }
      break;
    case 4:
      if (rc == END_FLAG)
      {
        new_packet = true;
      }
      packet_status = 0;
      break;
    }
  }
}

void send_packet(uint8_t cmd)
{
  Serial.write(0x00);
  Serial.write(START_FLAG);
  Serial.write(cmd);
  Serial.write((uint8_t)0x00);
  Serial.write(END_FLAG);
  Serial.write(0x00);
}
