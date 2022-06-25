import os, os.path
import numpy as np
import math as m
from numba import njit, jit
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm, maxwell


def safe_open_w(path):
    ''' Open "path" for writing, creating any parent directories as needed.
    ''' 
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w')


@njit
def lennard_jones(L, r):

    acel = np.zeros((len(r), 2))

    for i in range(len(r)):
        for j in range(len(r)):

            if j != i:
                # r_ij_x = np.array([r[j,0]-r[i,0], r[j,0]-r[i,0]+L, r[j,0]-r[i,0]-L])
                # r_ij_y = np.array([r[j,1]-r[i,1], r[j,1]-r[i,1]+L, r[j,1]-r[i,1]-L])

                # r_ij = np.array([r_ij_x[np.argmin(np.abs(r_ij_x))], r_ij_y[np.argmin(np.abs(r_ij_y))]])

                # dist = m.sqrt(r_ij[0]**2 + r_ij[1]**2)
                dist, r_ij = calcula_distancia(L, r[j], r[i])

                if dist < 3:

                    r_ij = r_ij / dist

                    acel[i] -= 24 * (2/dist**13 - 1/dist**7) * r_ij

    return acel


@njit
def calcula_distancia(L, r1, r2):

    r_ij_x = np.array([r1[0]-r2[0], r1[0]-r2[0]+L, r1[0]-r2[0]-L])
    r_ij_y = np.array([r1[1]-r2[1], r1[1]-r2[1]+L, r1[1]-r2[1]-L])

    r_ij = np.array([r_ij_x[np.argmin(np.abs(r_ij_x))], r_ij_y[np.argmin(np.abs(r_ij_y))]])

    distancia = m.sqrt(r_ij[0]**2 + r_ij[1]**2)

    return distancia, r_ij


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
    a = lennard_jones(L, r)   # aceleración actualizada a partir de las nuevas posiciones
    v = w + 0.5*h*a   # velocidades actualizadas con las nuevas aceleraciones

    return r, v, a


def posiciones_iniciales(N, L, shape="aleatorio", minimum_distance=0.85, iter_max = 1e6):

    n = int(m.sqrt(N))

    # Posiciones iniciales aleatorias. Hay que tener en cuenta que las partículas
    # no pueden empezar muy juntas porque el potencial es altamente repulsivo para 
    # distancias muy cortas
    # Se añade un punto aleatorio y empieza un bucle en el que se genera un nuevo 
    # punto aleatorio y si cumple la restricción de distancia mínima con todos
    # los demás se añade. 
    # Se realiza el bucle hasta que se añadan N puntos o, en caso de que no se pueda
    # llevar a cabo debido a que la densidad de partículas sea muy grande, se llegue
    # a un número máximo de iteraciones
    if shape == "aleatorio":
        
        pos = [np.random.uniform(0, L, 2)]
        iter = 0
        while len(pos) < N and iter < iter_max:
            pos_nueva = np.random.uniform(0, L, 2)
            aceptado = True
            for element in pos:
                if calcula_distancia(L, pos_nueva, element)[0] < minimum_distance:
                    aceptado = False

            if aceptado:
                pos.append(pos_nueva)

            iter += 1


    elif shape == "cuadrado":
        pos = [(np.array([i,j])*L/n + L/(2*n)) for i in range(n) for j in range(n)]

    elif shape == "hexagonal":
        space_x = L / (n * 3)
        space_y = L / n
        pos = [(np.array([6*i, 2*j]) + 0.5) for i in range(n//2) for j in range(n//2)]
        pos += [(np.array([6*i+2, 2*j]) + 0.5) for i in range(n//2) for j in range(n//2)]
        pos += [(np.array([6*i+3, 2*j+1]) + 0.5) for i in range(n//2) for j in range(n//2)]
        pos += [(np.array([6*i+5, 2*j+1]) + 0.5) for i in range(n//2) for j in range(n//2)]

        pos = np.array(pos) * np.array([space_x, space_y])

    return np.array(pos)


@njit
def calculo_energia_pot(L, r_data):

    energia = np.zeros(len(r_data))

    for time in range(len(r_data)):
        r = r_data[time]
        for i in range(len(r)):
            for j in range(len(r)):

                if j != i:

                    dist, _ = calcula_distancia(L, r[i], r[j])
                    # r_ij_x = np.array([r[j,0]-r[i,0], r[j,0]-r[i,0]+L, r[j,0]-r[i,0]-L])
                    # r_ij_y = np.array([r[j,1]-r[i,1], r[j,1]-r[i,1]+L, r[j,1]-r[i,1]-L])

                    # r_ij = np.array([r_ij_x[np.argmin(np.abs(r_ij_x))], r_ij_y[np.argmin(np.abs(r_ij_y))]])

                    # dist = m.sqrt(r_ij[0]**2 + r_ij[1]**2)

                    energia[time] += 4 * (dist**(-12) - dist**(-6))

    return energia


def promedios_temporales(x, n_puntos):

    extra = len(x) % n_puntos
    y_mod = x[0:-extra].reshape(-1, n_puntos).mean(axis=1)

    return y_mod


def calculo_temperatura(N, v, t1, t2, dt):

     # CÁLCULO DE LAS ENERGÍAS
    energia_cin = 0.5 * np.sum(v[:,:,0]**2 + v[:,:,1]**2, axis=1)

    T_equiparticion = np.average(energia_cin[round(t1/dt):round(t2/dt)])/N

    return T_equiparticion



def main(L, N, dt, tmax, pos0, vel0, cambio_velocidad=(1.5, 6), particula=0, tiempo_medida=40, 
    archivos=None, dir=None, freq=20):

    if dir:
        for key in archivos.keys():     
            archivos[key] = dir + archivos[key]


    # Variables para almacenar los resultados
    v_data = np.empty((round(tmax/dt)+2, N, 2))
    correlacion = np.empty((round(tiempo_medida/dt/10), N-1))

    v_data[0] = vel0

    # Cálculo de la aceleración inicial
    acel = lennard_jones(L, pos0)    

    reescalamiento, n_veces = cambio_velocidad

    pos = pos0
    vel = vel0

    center_bins_tot = []
    cuentas_tot = []
    temp_tot = []

    for ii in range(n_veces):
        t = 0
        contador = 0
        subcontador = 0
        while t < tmax:

            # Algoritmo de Verlet
            pos, vel, acel = Verlet(L, pos, vel, dt, acel)

            # Guardar los datos
            # r_data[contador+1] = pos
            v_data[contador+1] = vel
            if contador > (tmax-tiempo_medida)/dt and contador % 10 == 0:
                for j in range(N-1):
                    jj = (particula + j + 1) % N
                    correlacion[subcontador, j] = calcula_distancia(L, pos[particula], pos[jj])[0]
                subcontador += 1
    
            t += dt
            contador += 1     
        
        temp = calculo_temperatura(N, v_data, tmax-tiempo_medida, tmax, dt)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        n, bins, _ = ax.hist(correlacion.flatten(), bins=40, 
                    color='#0504aa', alpha=0.7, rwidth=0.85, density=True)

        
        center_bins = np.empty_like(n)

        for k in range(len(n)):
            center_bins[k] = (bins[k]+bins[k+1])/2
        
        
        center_bins_tot.append(center_bins)
        cuentas_tot.append(n)
        temp_tot.append(temp)

        plt.xlabel("r")
        plt.ylabel("g(r)")

        plt.title(f"Función de correlación de pares (T={temp:.2f})", fontweight="bold")
        name_graph = archivos["graph_funcion_correlacion"]
        plt.savefig(f"{name_graph}_{ii+1}")

        vel *= reescalamiento


    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(center_bins_tot)):
        ax.plot(center_bins_tot[i], cuentas_tot[i], label=f"T = {temp_tot[i]:.2f}")

    plt.xlabel("r")
    plt.ylabel("g(r)")

    plt.title(f"Función de correlación de pares vs temperatura \n (L={L:.2f}, N={N})", fontweight="bold")

    plt.legend(loc="best")
    plt.savefig(archivos["graph_funcion_correlacion_temperatura"])




# PROGRAMA PRINCIPAL
#---------------------------------------------------------------------------------------

path = os.path.dirname(os.path.abspath(__file__)) + "/"

# PARÁMETROS INICIALES 
L = 4.  # longitud de la caja
N = 16   # número de partículas

# AJUSTES DE LA SIMULACIÓN
dt = 0.001    # paso temporal
tmax = 60     # tiempo total de simulación

# NOMBRES DE TODOS LOS ARCHIVOS A GUARADAR  
archivos = {
    "graph_temperatura" : "temperatura",
    "graph_funcion_correlacion" : "funcion_correlacion",
    "graph_funcion_correlacion_temperatura" : "funcion_correlacion_temperatura"
    }

# Cálculo de posiciones iniciales en red cuadrada y en reposo
pos0 = posiciones_iniciales(N, L, shape="cuadrado")
vel0 = np.zeros_like(pos0)

particula = 1
#cambio_velocidad = (1.5, [20, 30, 35, 45])

main(L, N, dt, tmax, pos0, vel0, archivos=archivos, dir=path)



# Sistema gaseoso
L = 10.
N = 20

pos0 = posiciones_iniciales(N, L, shape="aleatorio", minimum_distance=1.0)
vel0 = np.random.uniform(-1, 1, pos0.shape)

archivos = {
    "graph_temperatura" : "temperatura",
    "graph_funcion_correlacion" : "funcion_correlacion_gas",
    "graph_funcion_correlacion_temperatura" : "funcion_correlacion_temperatura_gas"
    }

dir = path


pos0 = posiciones_iniciales(N, L, shape="aleatorio", minimum_distance=1.0)
vel0 = np.random.uniform(-1, 1, pos0.shape)

main(L, N, dt, tmax, pos0, vel0, archivos=archivos, dir=dir)
