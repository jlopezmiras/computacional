from time import time
import numpy as np
import matplotlib.pyplot as plt
import math as m
from numba import njit, jit


G = 6.67e-11
MT = 5.9736e24
ML = 0.07349e24
DTL = 3.844e8
W = 2.6617e-6
RT = 6.37816e6
RL = 1.7374e6
K = G * MT / DTL**3
M_RED = ML/MT


def ecuaciones_movimiento(variables, t):

    r, phi, p_r, p_phi = variables

    r_L = m.sqrt(1 + r*r - 2*r*m.cos(phi-W*t))

    dr = p_r 
    dphi = p_phi / (r*r)
    dp_r = p_phi*p_phi / (r*r*r) - K*(1/(r*r) + M_RED/(r_L*r_L*r_L)*(r-m.cos(phi-W*t)))
    dp_phi = - K * M_RED * r / (r_L*r_L*r_L) * m.sin(phi-W*t)

    return dr, dphi, dp_r, dp_phi


@jit
def runge_kutta_4(variables, t, h):

    variables = np.array(variables)
    k1 = h * np.array(ecuaciones_movimiento(variables, t))
    k2 = h * np.array(ecuaciones_movimiento(variables+k1/2, t+h/2))
    k3 = h * np.array(ecuaciones_movimiento(variables+k2/2, t+h/2))
    k4 = h * np.array(ecuaciones_movimiento(variables+k3, t+h))

    variables += (k1 + 2*k2 + 2*k3 + k4)
    t += h

    return variables, t



# CONDICIONES INICIALES
t = 0.
r = RT/DTL
#phi = m.pi/6
#v = 11500/DTL
#theta = m.pi/14
phi = m.pi/3
v = 11100/DTL
theta = m.pi/4
p_r = v * m.cos(theta-phi)
p_phi = r * v * m.sin(theta-phi)

variables = [r, phi, p_r, p_phi]


# PARÁMETROS DE LA SIMULACIÓN
h = 1
iter = int(3e5)
contador = 0
fout = "cohete.dat"

f = open(fout, "w")

while (contador<iter):

    variables, t = runge_kutta_4(variables, t, h)

    if contador%500 == 0:

        r, phi = variables[0], variables[1]
        x_cohete, y_cohete = r*m.cos(phi), r*m.sin(phi)
        x_luna, y_luna = m.cos(W*t), m.sin(W*t)
        f.write("0.0, 0.0\n")
        f.write(f"{x_cohete}, {y_cohete}\n")
        f.write(f"{x_luna}, {y_luna}\n")
        f.write('\n')
    
    contador += 1


f.close()


