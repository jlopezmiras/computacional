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
    y_extra = x[-extra:].mean()

    return np.append(y_mod, y_extra)


def grafica_energia(L, r, v, dt, tmax, name_graph):

     # CÁLCULO DE LAS ENERGÍAS
    energia_cin = 0.5 * np.sum(v[:,:,0]**2 + v[:,:,1]**2, axis=1)
    energia_pot = calculo_energia_pot(L, r)
    energia_tot = energia_cin + energia_pot

    # GRÁFICA DE LAS ENERGÍAS
    t = np.arange(0,tmax+dt,dt)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    n_puntos = int(tmax / dt / 300)
    t = promedios_temporales(t, n_puntos)
    energia_cin = promedios_temporales(energia_cin, n_puntos)
    energia_pot = promedios_temporales(energia_pot, n_puntos)
    energia_tot = promedios_temporales(energia_tot, n_puntos)

    ax.plot(t, energia_cin, color="blue", label="Energía cinética")
    ax.plot(t, energia_pot, color="orange", label="Energía potencial")
    ax.plot(t, energia_tot, color="green", label="Energía total")

    ax.grid("--", alpha=0.5)

    ax.legend(loc="best")
    fig.savefig(name_graph)
    # fig.show()

    # CÁLCULO DE LA TEMPERATURA POR TEOREMA DE EQUIPARTICIÓN
    T_equiparticion = np.average(energia_cin[round(20/dt):round(50/dt)])/N

    return T_equiparticion


def grafica_desplazamiento_cuadrado(r_data, particulas, dt, tmax, name_graph):

    t = np.arange(0, tmax+dt, dt)
    

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if isinstance(particulas, list):
        desplazamiento = r_data[:,particulas[0]] - r_data[:,particulas[1]]

        plt.ylabel("$<(r_i-r_j)^2>$")
        plt.title("Seaparación media cuadrática entre dos partículas")

    else:
        desplazamiento = r_data[:,particulas] - r_data[0,particulas,:]
        
        plt.ylabel("$<(r-r_0)^2>$")
        plt.title("Desplazamiento medio cuadrado de una partícula")

    desplazamiento_cuadrado = desplazamiento[:,0]**2 + desplazamiento[:,1]**2

    # Hacemos medias para que los datos se visualicen mejor, de forma que solo
    # se tengan 300 puntos
    n_puntos = int(tmax / dt / 300)
    t = promedios_temporales(t, n_puntos)
    desplazamiento_cuadrado = promedios_temporales(desplazamiento_cuadrado, n_puntos)

    ax.plot(t, desplazamiento_cuadrado)

    plt.xlabel("Tiempo")
    
    plt.savefig(name_graph)

    return


def main(L, N, dt, tmax, pos0, vel0, cambio_velocidad=(1.2, 6), particula=0, tiempo_medida=40, 
    archivos=None, dir="", freq=20):

    if dir!="":
        for key in archivos.keys():     
            archivos[key] = dir + archivos[key]


    # Variables para almacenar los resultados
    r_data = np.empty((round(tmax/dt)+1, N, 2))
    v_data = np.empty((round(tmax/dt)+1, N, 2))
    correlacion = np.empty((round(tiempo_medida/dt/10), N-1))

    r_data[0] = pos0
    v_data[0] = vel0

    # Cálculo de la aceleración inicial
    acel = lennard_jones(L, pos0)    

    reescalamiento, n_veces = cambio_velocidad

    pos = pos0
    vel = vel0

    for ii in range(n_veces):
        t = 0
        contador = 0
        subcontador = 0
        while t < tmax:

            # Algoritmo de Verlet
            pos, vel, acel = Verlet(L, pos, vel, dt, acel)

            # Guardar los datos
            # r_data[contador+1] = pos
            # v_data[contador+1] = vel
            if contador > (tmax-tiempo_medida)/dt and contador % 10 == 0:
                for j in range(N-1):
                    jj = (j - particula - 1) % N
                    correlacion[subcontador, jj] = calcula_distancia(L, pos[particula], pos[jj])[0]
                subcontador += 1
    
            t += dt
            contador += 1     
        
        print(correlacion[0])

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.hist(correlacion.flatten(), bins=40, color='#0504aa', alpha=0.7, rwidth=0.85, density=True)

        plt.show()

        vel *= reescalamiento
   




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
    "fout" : "posiciones.dat",
    "graph_energias" : "energias",
    "graph_desplazamiento_cuadrado" : "desplazamiento_medio_cuadrado"
    }

# Cálculo de posiciones iniciales en red cuadrada y en reposo
pos0 = posiciones_iniciales(N, L, shape="cuadrado")
vel0 = np.zeros_like(pos0)

dir = path + "transicion_rapida/"

particula = 1
#cambio_velocidad = (1.5, [20, 30, 35, 45])

main(L, N, dt, tmax, pos0, vel0, archivos=archivos, dir=dir)

