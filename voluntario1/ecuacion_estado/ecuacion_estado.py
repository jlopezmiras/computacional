import os
import numpy as np
import math as m
from numba import njit
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

# Función que evalúa el potencial de lennard Jones dado el vector
# r que almacena todas las posiciones de todas las partículas
# Tiene en cuenta las condiciones de contorno periódicas
@njit
def lennard_jones(L, r):

    acel = np.zeros((len(r), 2))

    for i in range(len(r)):
        for j in range(len(r)):

            if j != i:
                r_ij_x = np.array([r[j,0]-r[i,0], r[j,0]-r[i,0]+L, r[j,0]-r[i,0]-L])
                r_ij_y = np.array([r[j,1]-r[i,1], r[j,1]-r[i,1]+L, r[j,1]-r[i,1]-L])

                r_ij = np.array([r_ij_x[np.argmin(np.abs(r_ij_x))], r_ij_y[np.argmin(np.abs(r_ij_y))]])

                dist = m.sqrt(r_ij[0]**2 + r_ij[1]**2)

                if dist < 3:

                    r_ij = r_ij / dist

                    acel[i] -= 24 * (2/dist**13 - 1/dist**7) * r_ij

    return acel




# ALGORITMO DE VERLET (RESOLUCIÓN DE LA ECUACIÓN DIFERENCIAL)
@njit
def Verlet(L, r, v, h, a):
    
    w = v + 0.5*h*a   
    r += h * w    # posiciones actualizadas de las partículas con paso h
    presion = calcula_presion(L, r, v, h)
    r = r % L
    a = lennard_jones(L, r)   # aceleración actualizada a partir de las nuevas posiciones
    v = w + 0.5*h*a   # velocidades actualizadas con las nuevas aceleraciones

    return r, v, a, presion


# Función para calcular la presión cuando las partículas atraviesan las paredes
@njit
def calcula_presion(L, r, v, dt):

    fuerza = 0.
    for i in range(len(r)):
        if r[i,0] > L or r[i,0] < 0:
            fuerza += abs(2*v[i,0])
        if r[i,1] > L or r[i,1] < 0:
            fuerza += abs(2*v[i,1])

    return fuerza/dt/(4*L)


# Función para determinar las posiciones iniciales en función de si son aleatorias o 
# se quieren en red cuadrada o hexagonal
def posiciones_iniciales(N, L):

    n = int(m.sqrt(N))

    if n*n == N:
       pos = [(np.array([i,j])*L/n + L/(2*n)) for i in range(n) for j in range(n)]

    else:
        pos = [np.array([i*L/(n+1) + L/(n+1)/2, j*L/n + L/(2*n)]) for i in range(n) for j in range(n)]

        for j in range(N-n*n):
            pos.append(np.array([n, j])*L/(n+1) + L/(n+1)/2)

    pos += np.random.uniform(-0.5, 0.5, (N,2))

    return np.array(pos)


# Función que calcula la energía potencial dado el vector r que almacena todas las 
# posiciones de todas las partículas
# Tiene en cuenta las condiciones de contorno periódicas
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


@njit
def main(L, N, dt, tmax, pos0, vel0):

    # Variables para almacenar los resultados
    r_data = np.zeros((round(tmax/dt)+1, N, 2))
    v_data = np.zeros((round(tmax/dt)+1, N, 2))
    presion_data = np.zeros(round(tmax/dt)+1)

    r_data[0] = pos0
    v_data[0] = vel0

    # Cálculo de la aceleración inicial
    acel = lennard_jones(L, pos0)

    # Inicio del bucle de la simulación
    t = 0
    contador = 0
    pos = pos0
    vel = vel0

    while t < tmax:

        # Algoritmo de Verlet
        pos, vel, acel, presion = Verlet(L, pos, vel, dt, acel)

        # Guardar los datos
        r_data[contador+1] = pos
        v_data[contador+1] = vel
        presion_data[contador+1] = presion

        t += dt
        contador += 1


    # CÁLCULO DE LAS ENERGÍAS
    energia_cin = 0.5 * np.sum(v_data[:,:,0]**2 + v_data[:,:,1]**2, axis=1)

    # CÁLCULO DE LA TEMPERATURA POR TEOREMA DE EQUIPARTICIÓN Y DE LA PRESIÓN
    T_equiparticion = np.average(energia_cin[round(20/dt):round(50/dt)])/N
    presion = np.average(presion_data[round(20/dt):round(50/dt)])

    return T_equiparticion, presion

# Función lineal para hacer el ajuste
def linear(x, m, n):
    return m*x + n




# PROGRAMA PRINCIPAL
#---------------------------------------------------------------------------------------
if __name__=='__main__':

    path = os.path.dirname(os.path.abspath(__file__))
    
    # PARÁMETROS INICIALES 
    L = 10.  # longitud de la caja
    N = 16   # número de partículas

    # AJUSTES DE LA SIMULACIÓN
    dt = 0.002    # paso temporal
    tmax = 60     # tiempo total de simulación

    fout = "ecuacion_estado.txt"
    name_graph = "ecuacion_estado"

    # Cálculo de posiciones iniciales aleatorias
    pos0 = posiciones_iniciales(N, L)
    
    # Variables de Temperatura y presión
    temperatura = []
    presion = []

    # Simulaciones con módulos de velocidad iniciales variables
    modulo = 0.5
    while modulo < 5:
        ang = np.random.uniform(0, 2*m.pi, N)
        vel0 = modulo * np.array(list(zip(np.cos(ang), np.sin(ang))))

        t, p = main(L, N, dt, tmax, pos0, vel0)
        temperatura.append(t)
        presion.append(p)

        modulo += 0.3

    # Ajuste y gráfica de la presión frente a la temperatura

    temperatura = np.array(temperatura)
    presion = np.array(presion)

    fout = path + "/" + fout
    np.savetxt(fout, list(zip(temperatura, presion)), fmt='%f')

    params, cov = curve_fit(linear, temperatura, presion)
    errores = np.sqrt(np.diag(cov))
    print(f"m = {params[0]} +/- {errores[0]}")
    print(f"n = {params[1]} +/- {errores[1]}")

    x = np.linspace(0, max(temperatura)*1.2, 300)
    y = linear(x, *params)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, y, alpha=0.7)
    ax.scatter(temperatura, presion, s=7)

    plt.xlabel("Temperatura")
    plt.ylabel("Presión")

    ax.grid("--", alpha=0.5)

    plt.title("Ecuación de estado", fontweight="bold")

    name_graph = path + "/" + name_graph
    fig.savefig(name_graph)
    # plt.show()

    