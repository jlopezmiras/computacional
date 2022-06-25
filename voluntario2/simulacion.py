from time import time
import numpy as np
import matplotlib.pyplot as plt
import math as m
from numba import njit, jit


G = 6.67e-11
MT = 5.9736e24
ML = 0.07349e24
MMET = 8.712684e15
RMET = 10000
DTL = 3.844e8
W = 2.6617e-6
RT = 6.37816e6
RL = 1.7374e6
K = G * MT / DTL**3
M_RED_L = ML/MT
M_RED_MET = MMET/MT


def cartesianas_to_polares(x, y):
    r = m.sqrt(x**2 + y**2)
    phi = m.atan(abs(y/x))
    if x < 0 and y > 0:
        phi = m.pi - phi
    elif x < 0 and y < 0:
        phi = phi - m.pi
    elif x > 0 and y < 0:
        phi = - phi
    return r, phi

def polares_to_cartesianas(r, phi):
    return r*m.cos(phi), r*m.sin(phi)

def vel_cartesianas_to_polares(x, y, vx, vy):
    vr = (x*vx + y*vy) / m.sqrt(x**2 + y**2)
    vphi = (x*vy - y*vx) / (x**2 + y**2)
    return vr, vphi

def vel_polares_to_cartesianas(r, phi, vr, vphi):
    vx = vr * m.cos(phi) - vphi / r * m.sin(phi)
    vy = vr * m.sin(phi) + vphi / r * m.cos(phi)
    return vx, vy



def movimiento_meteorito(meteorito, t):

    r, phi, p_r, p_phi = meteorito

    dr = p_r
    dphi = 0
    dp_r = - K * 1/(r*r)
    dp_phi = 0

    return dr, dphi, dp_r, dp_phi


def movimiento_trozo_meteorito(meteorito, phi_L, t):

    r, phi, p_r, p_phi = meteorito

    r_L = m.sqrt(1 + r*r - 2*r*m.cos(phi-phi_L-W*t))

    dr = p_r 
    dphi = p_phi / (r*r)
    dp_r = p_phi*p_phi / (r*r*r)
    dp_r -= K * (1/(r*r) + M_RED_L/(r_L*r_L*r_L)*(r-m.cos(phi-phi_L-W*t)))
    dp_phi = - K * M_RED_L * r / (r_L*r_L*r_L) * m.sin(phi-phi_L-W*t)

    return dr, dphi, dp_r, dp_phi


def movimiento_cohete(cohete, d_M, phi_L, t):

    r, phi, p_r, p_phi = cohete

    r_L = m.sqrt(1 + r*r - 2*r*m.cos(phi-phi_L-W*t))
    r_M = m.sqrt(d_M*d_M + r*r - 2*r*d_M*m.sin(phi))

    dr = p_r 
    dphi = p_phi / (r*r)
    dp_r = p_phi*p_phi / (r*r*r)
    dp_r -= K * (1/(r*r) + M_RED_L/(r_L*r_L*r_L)*(r-m.cos(phi-phi_L-W*t)) + M_RED_MET/(r_M*r_M*r_M)*(r-d_M*m.sin(phi)))
    dp_phi = - K * M_RED_L * r / (r_L*r_L*r_L) * m.sin(phi-phi_L-W*t) + K * M_RED_MET * r / (r_M*r_M*r_M) * m.cos(phi)

    return dr, dphi, dp_r, dp_phi


def runge_kutta_4_trozos(trozo_meteorito, luna, t, h):

    meteorito = np.array(trozo_meteorito)
    
    k1 = h * np.array(movimiento_trozo_meteorito(meteorito, luna, t))
    k2 = h * np.array(movimiento_trozo_meteorito(meteorito+k1/2, luna, t))
    k3 = h * np.array(movimiento_trozo_meteorito(meteorito+k2/2, luna, t))
    k4 = h * np.array(movimiento_trozo_meteorito(meteorito+k3, luna, t))

    meteorito += (k1 + 2*k2 + 2*k3 + k4)
    t += h

    return meteorito, t


def runge_kutta_4(cohete, meteorito, luna, t, h):

    cohete, meteorito = np.array(cohete), np.array(meteorito)
    
    l1 = h * np.array(movimiento_meteorito(meteorito, t))
    k1 = h * np.array(movimiento_cohete(cohete, meteorito[0], luna, t))

    l2 = h * np.array(movimiento_meteorito(meteorito+l1/2, t))
    k2 = h * np.array(movimiento_cohete(cohete+k1/2, meteorito[0]+l1[0]/2, luna, t+h/2))

    l3 = h * np.array(movimiento_meteorito(meteorito+l2/2, t))
    k3 = h * np.array(movimiento_cohete(cohete+k2/2, meteorito[0]+l2[0]/2, luna, t+h/2))

    l4 = h * np.array(movimiento_meteorito(meteorito+l3, t))
    k4 = h * np.array(movimiento_cohete(cohete+k3, meteorito[0]+l3[0], luna, t+h))

    cohete += (k1 + 2*k2 + 2*k3 + k4)
    meteorito += (l1 + 2*l2 + 2*l3 + l4)
    t += h

    return cohete, meteorito, t



def dar_impulso(cohete, impulso, theta=None):

    r, phi, p_r, p_phi = cohete

    # Hallo las componentes cartesianas de la posición
    x, y = polares_to_cartesianas(r, phi)

    # Hallo las componentes cartesianas del momento
    p_x, p_y = vel_polares_to_cartesianas(r, phi, p_r, p_phi)

    # Hallo el ángulo que forma el momento
    if not theta:
        theta = cartesianas_to_polares(p_x, p_y)[1]

    # Sumo la velocidad a las componentes cartesianas del momento
    p_x += impulso * m.cos(theta)
    p_y += impulso * m.sin(theta)

    # Devuelvo el momento en polares
    return vel_cartesianas_to_polares(x, y, p_x, p_y)



    

# CONDICIONES INICIALES
t = 0.

# Ángulo inicial de la Luna
phi_L = m.pi/4

# Cohete
r = RT/DTL
phi = phi_L - 0.0099983
v = 11400/DTL
theta = phi_L + 0.025

p_r = v * m.cos(theta-phi)
p_phi = r * v * m.sin(theta-phi)

cohete = [r, phi, p_r, p_phi]

v_detonacion = 200/DTL


# Meteorito
r = 10
phi = m.pi/2
p_r = -5000/DTL
p_phi = 0

meteorito = [r, phi, p_r, p_phi]



# PARÁMETROS DE LA SIMULACIÓN
h = 1
iter = int(5e5)
contador = 0
fout = "simulacion.dat"

f = open(fout, "w")

data_vr, data_vtheta, data_v = [], [], []
dist_meteorito = []

aterrizaje = False

# Gasto energetico a lo largo de la misión
gasto_energetico = [0, cohete[2]**2 + cohete[3]**2/cohete[0]**2]


while contador < iter and not aterrizaje:

    cohete, meteorito, t = runge_kutta_4(cohete, meteorito, phi_L, t, h)

    r_L = m.sqrt(1 + cohete[0]*cohete[0] - 2*cohete[0]*m.cos(cohete[1]-phi_L-W*t))
    r_M = m.sqrt(meteorito[0]*meteorito[0] + cohete[0]*cohete[0] - 2*cohete[0]*meteorito[0]*m.sin(cohete[1]))


    gasto_energetico.append(gasto_energetico[-1])

    if t==19637:
        gasto_energetico_impulso = -cohete[2]**2 - cohete[3]**2/cohete[0]**2
        cohete[2], cohete[3] = dar_impulso(cohete, -2.28e-6)
        gasto_energetico_impulso += cohete[2]**2 + cohete[3]**2/cohete[0]**2 
        gasto_energetico[-1] += abs(gasto_energetico_impulso)
    
    elif t==101850:
        gasto_energetico_impulso = -cohete[2]**2 - cohete[3]**2/cohete[0]**2
        cohete[2], cohete[3] = dar_impulso(cohete, 2.28e-6, -m.pi/2)
        cohete[2], cohete[3] = dar_impulso(cohete, 2.28e-6, -m.pi/2)
        cohete[2], cohete[3] = dar_impulso(cohete, 2.28e-6, -m.pi/2)
        cohete[2], cohete[3] = dar_impulso(cohete, 2.28e-6, -m.pi/2)
        cohete[2], cohete[3] = dar_impulso(cohete, 2.28e-6, -m.pi/2)
        cohete[2], cohete[3] = dar_impulso(cohete, 2.28e-6, -m.pi/2)
        gasto_energetico_impulso += cohete[2]**2 + cohete[3]**2/cohete[0]**2 
        gasto_energetico[-1] += abs(gasto_energetico_impulso)

    
     
    
    # if t==102000:
    #     gasto_energetico -= cohete[2]**2 + cohete[3]**2/cohete[0]**2
    #     cohete[2], cohete[3] = dar_impulso(cohete, 1.35e-6, -m.pi/2)
    #     gasto_energetico += cohete[2]**2 + cohete[3]**2/cohete[0]**2

    if r_M < RMET/DTL:
        aterrizaje = True
        print("ATERRIZAJE")
        print(f"Distancia: {r_M}")
        print(f"Tiempo: {t}")
        t_mision = t
        v_meteorito = np.array(vel_polares_to_cartesianas(*meteorito))
        v_cohete = np.array(vel_polares_to_cartesianas(*cohete))
        vel_relativa = v_cohete - v_meteorito
        print(f"Velocidad relativa: {vel_relativa}")
        print(f"Velocidad del meteorito: {np.sqrt(v_meteorito[0]**2+v_meteorito[1]**2)}")
        print(f"Gasto energético total: {gasto_energetico[-1]}")
        


    if contador%150 == 0:

        r_cohete, phi_cohete = cohete[0], cohete[1]
        x_cohete, y_cohete = polares_to_cartesianas(r_cohete, phi_cohete)
        r_meteorito, phi_meteorito = meteorito[0], meteorito[1]
        x_meteorito, y_meteorito = polares_to_cartesianas(r_meteorito, phi_meteorito)
        x_luna, y_luna = m.cos(phi_L+W*t), m.sin(phi_L+W*t)
        f.write("0.0, 0.0\n")
        f.write(f"{x_cohete}, {y_cohete}\n")
        f.write(f"{x_meteorito}, {y_meteorito}\n")
        f.write(f"{x_luna}, {y_luna}\n")
        f.write('\n')

        data_vr.append(cohete[2])
        data_vtheta.append(cohete[3])
        data_v.append(np.sqrt(cohete[2]**2 + cohete[3]**2/cohete[0]**2))

        r_M = m.sqrt(r_meteorito*r_meteorito + r_cohete*r_cohete - 2*r_cohete*r_meteorito*m.sin(phi_cohete))
        dist_meteorito.append(r_M)
    
    contador += 1


if aterrizaje:

    # Se divide en dos meteoritos
    # Condiciones iniciales
    r = meteorito[0]
    phi = meteorito[1]
    p_r = 0
    p_phi_1 = v_detonacion*r*r
    p_phi_2 = -v_detonacion*r*r

    meteorito1 = [r, phi, p_r, p_phi_1]
    meteorito2 = [r, phi, p_r, p_phi_2]

    while contador < iter:

        meteorito1, _ = runge_kutta_4_trozos(meteorito1, phi_L, t, h)
        meteorito2, t = runge_kutta_4_trozos(meteorito2, phi_L, t, h)

        if contador%600 == 0:

            r_meteorito1, phi_meteorito1 = meteorito1[0], meteorito1[1]
            x_meteorito1, y_meteorito1 = polares_to_cartesianas(r_meteorito1, phi_meteorito1)
            r_meteorito2, phi_meteorito2 = meteorito2[0], meteorito2[1]
            x_meteorito2, y_meteorito2 = polares_to_cartesianas(r_meteorito2, phi_meteorito2)
            x_luna, y_luna = m.cos(phi_L+W*t), m.sin(phi_L+W*t)
            f.write("0.0, 0.0\n")
            f.write(f"{x_meteorito1}, {y_meteorito1}\n")
            f.write(f"{x_meteorito2}, {y_meteorito2}\n")
            f.write(f"{x_luna}, {y_luna}\n")
            f.write('\n')
        
        contador += 1

f.close()


data_vr, data_vtheta, data_v = np.array(data_vr), np.array(data_vtheta), np.array(data_v)
dist_meteorito = np.array(dist_meteorito)
t = np.array([i for i in range(len(data_vr))])*150

fig, ax = plt.subplots()
ax.plot(t, data_vr, label="vr")
ax.plot(t, data_vtheta, label="vtheta")
ax.plot(t, data_v, label="v")

plt.legend()

plt.savefig("grafica_velocidad")
# plt.show()
plt.close(fig)

fig, ax = plt.subplots()
plt.xlim(0, 125000)
plt.ylim(0, 10)
ax.plot(t, dist_meteorito)
plt.savefig("grafica_distancia_meteorito")
# plt.show()
plt.close(fig)


fig, ax = plt.subplots()
t = np.arange(-1, t_mision+1)
ax.plot(t, gasto_energetico)
plt.savefig("grafica_gasto_energetico")
plt.show()
# plt.close(fig)

# print(f"Minima distancia : t={t[np.argmin(dist_meteorito)]}, \t d={min(dist_meteorito)}")
# print(f"Velocidad: {data_v[np.argmin(dist_meteorito)]}")


