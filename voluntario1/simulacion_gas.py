import numpy as np
import math as m
from numba import njit

from matplotlib import pyplot as plt



@njit
def fuerza(L, r):

    acel = np.zeros((len(r), 2))

    for i in range(len(r)):
        for j in range(len(r)):

            if j != i:
                r_ij_x = np.array([r[j,0]-r[i,0], r[j,0]-r[i,0]+L, r[j,0]-r[i,0]-L])
                r_ij_y = np.array([r[j,1]-r[i,1], r[j,1]-r[i,1]+L, r[j,1]-r[i,1]-L])

                r_ij = np.array([r_ij_x[np.argmin(np.abs(r_ij_x))], r_ij_y[np.argmin(np.abs(r_ij_y))]])

                dist = m.sqrt(r_ij[0]**2 + r_ij[1]**2)

                r_ij = r_ij / dist

                acel[i] -= 24 * (2/dist**13 - 1/dist**7) * r_ij

    return acel




# ALGORITMO DE VERLET (RESOLUCIÓN DE LA ECUACIÓN DIFERENCIAL)
# --------------------------------------------------------------------------------------
# Resuelve las ecuaciones del movimiento mediante el algoritmo de Verlet

# Recibe la masa, la posición, la velocidad, el paso de tiempo y la aceleración:
#   m (vector 1D: nplanets)   --> vector de la masa (reescalada) de cada planeta 
#   r (vector 2D: nplanets,2) --> vector de vectores posicion (reescalados) de cada planeta 
#   v (vector 2D: nplanets,2) --> vector de vectores velocidad (reescalados) de cada planeta 
#   h (escalar)               --> paso de la simulación 
#   a (vector 2D: nplanets,2) --> vector de vectores aceleración de cada planeta 
#
# Lleva a cabo el algoritmo de Verlet a partir de las posiciones, velocidades y aceleraciones calculadas
# en el paso inmediatamente anterior y devolviendo las posiciones, velocidades y aceleraciones del 
# paso siguiente
# --------------------------------------------------------------------------------------
# Utiliza el decorador @njit del módulo numba para ser compilado en tiempo real y 
# mejorar el coste de tiempo
@njit
def Verlet(L, r, v, h, a):
    
    w = v + 0.5*h*a   
    r += h * w    # posiciones actualizadas de los planetas con paso h
    r = r % L
    a = fuerza(L, r)   # aceleración actualizada a partir de las nuevas posiciones
    v = w + 0.5*h*a   # velocidades actualizadas con las nuevas aceleraciones

    return r,v,a



def posiciones_iniciales(N, L):

    n = int(m.sqrt(N))

    if n*n == N:
       pos = [(np.array([i,j])*L/(n+1) + L/(n+1)) for i in range(n) for j in range(n)]

    else:
        pos = [np.array([i*L/(n+2) + L/(n+2), j*L/(n+1) + L/(n+1)]) for i in range(n) for j in range(n)]

        for j in range(N-n*n):
            pos.append(np.array([n, j])*L/(n+2) + L/(n+2))

    pos += np.random.uniform(-0.5, 0.5, (N,2))

    return np.array(pos)



@njit
def calculo_energia_pot(L, r_data):

    energia = np.zeros(len(r_data))

    for time in range(len(r_data)):
        r = r_data[time]
        for i in range(len(r)):
            for j in range(len(r)):

                if j != i:
                    r_ij_x = np.array([r[j,0]-r[i,0], r[j,0]-r[i,0]+L, r[j,0]-r[i,0]-L])
                    r_ij_y = np.array([r[j,1]-r[i,1], r[j,1]-r[i,1]+L, r[j,1]-r[i,1]-L])

                    r_ij = np.array([r_ij_x[np.argmin(np.abs(r_ij_x))], r_ij_y[np.argmin(np.abs(r_ij_y))]])

                    dist = m.sqrt(r_ij[0]**2 + r_ij[1]**2)

                    energia[time] += 4 * (dist**(-12) - dist**(-6))

    return energia









# PROGRAMA PRINCIPAL
#---------------------------------------------------------------------------------------
if __name__=='__main__':

    L = 10.  # longitud de la caja
    N = 20   # número de partículas
    dt = 0.002    # paso temporal
    tmax = 60   # tiempo total de simulación

    fout = "data.dat"

    # Posiciones iniciales aleatorias
    pos = posiciones_iniciales(N, L)

    # Ángulo inicial de velocidad aleatorio
    ang = np.random.uniform(0, 2*m.pi, N)
    vel = np.array(list(zip(np.cos(ang), np.sin(ang))))

    # Cálculo de la aceleración inicial
    acel = fuerza(L, pos)

    print("Posiciones")
    print(pos)
    print("Velocidades")
    print(vel)
    print("Aceleraciones")
    print(acel)

    r_data = np.empty((round(tmax/dt)+1, N, 2))
    v_data = np.empty((round(tmax/dt)+1, N, 2))

    r_data[0] = pos
    v_data[0] = vel
    

    
    t = 0
    contador = 0


    
    f = open(fout, "w")
    np.savetxt(f, pos, delimiter=", ")
    f.write("\n")


    while t < tmax:

        pos, vel, acel = Verlet(L, pos, vel, dt, acel)

        r_data[contador+1] = pos
        v_data[contador+1] = vel

        if contador % 20 == 0:

            np.savetxt(f, pos, delimiter=", ")
            f.write("\n")

        t += dt
        contador += 1

    f.close()


    energia_cin = 0.5 * np.sum(v_data[:,:,0]**2 + v_data[:,:,1]**2, axis=1)
    energia_pot = calculo_energia_pot(L, r_data)
    energia_tot = energia_cin + energia_pot


    t = np.arange(0,tmax+dt,dt)

    plt.plot(t, energia_cin, color="blue")
    plt.plot(t, energia_pot, color="orange")

    plt.show()

    plt.plot(t, energia_tot)

    plt.show()

    temp = np.average(energia_cin[round(20/dt):round(50/dt)])/N

    print(temp)

    
    





    

    