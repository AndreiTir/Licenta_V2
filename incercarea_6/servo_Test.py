#!/usr/bin/env/

import time
import board
import pwmio
from adafruit_motor import servo

pwm_pan = pwmio.PWMOut(board.PWM1, duty_cycle=0, frequency=50)
pwm_tilt = pwmio.PWMOut(board.PWM2, duty_cycle=0, frequency=50)

# pwm_pan 0-180
# pwm_tilt = 30-135


def muta_servo(servo, delta, max, angles):
    eroare = abs(delta)
    sign = -1 * eroare/delta
    if eroare <= 20:
        pass
    elif (eroare > 20 and eroare <= (max+20)/3) and (servo.angle - 3 > angles[0] and servo.angle + 3 < angles[1]):
        servo.angle += 3 * sign
    elif (eroare > (max+20)/3 and eroare <= max) and (servo.angle - 7 > angles[0] and servo.angle + 7 < angles[1]):
        servo.angle += 7 * sign
    else:
        pass
