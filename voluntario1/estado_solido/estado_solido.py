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

    return r,v,a


def posiciones_iniciales(N, L, shape="aleatorio", minimum_distance=0.8, iter_max = 2e6):

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

            # if iter % 100 == 0:
            #     print(len(pos))
        print("hecho")


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

        # fig, ax = plt.subplots()
        # for element in pos:
        #     ax.add_patch(plt.Circle(list(element), 0.1))

        # plt.xlim(0, 4)
        # plt.ylim(0, 4)
        
        # plt.show()


        

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


def grafica_temperatura(v, dt, tmax, name_graph):

    energia_cin = 0.5 * np.sum(v[:,:,0]**2 + v[:,:,1]**2, axis=1)
    t = np.arange(0,tmax+dt,dt)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    t = promedios_temporales(t, 300)
    temperatura = promedios_temporales(energia_cin, 300)/N
    ax.plot(t, temperatura, color="blue", label="Energía cinética")

    ax.grid("--", alpha=0.5)

    ax.legend(loc="best")
    fig.savefig(name_graph)
    # fig.show()

    # CÁLCULO DE LA TEMPERATURA POR TEOREMA DE EQUIPARTICIÓN
    T_equiparticion = np.average(energia_cin[round(20/dt):round(50/dt)])/N

    return T_equiparticion


def promedios_temporales(x, n_puntos):

    extra = len(x) % n_puntos
    y_mod = x[0:-extra].reshape(-1, n_puntos).mean(axis=1)
    y_extra = x[-extra:].mean()

    return np.append(y_mod, y_extra)



def histograma(v, T_equiparticion, distribution, name_graph, bins=50):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Histograma de velocidades 
    n, bins, _ = ax.hist(v, bins=40, color='#0504aa', alpha=0.7, rwidth=0.85, density=True)

    # Elegir distribución entre maxwell y normal
    if distribution == "maxwell":
        stats = maxwell
        x = np.linspace(0, bins.max(), 1000)
        plt.xlim(0, bins.max())
    elif distribution == "norm":
        stats = norm
        x = np.linspace(-bins.max(), bins.max(), 1000)
        plt.xlim(-bins.max(), bins.max())

    params = stats.fit(v, floc=0)

    # Gráficas de las distribuciones
    ax.plot(x, stats.pdf(x, *params), color='orange', label=f"Distribucion {distribution} ajustada")
    ax.plot(x, stats.pdf(x, 0, m.sqrt(T_equiparticion)), color='black', label=f"Distribucion {distribution} T. equipartición")

    ax.legend(loc="best")
    fig.savefig(name_graph)
    # plt.show()

    return params[1]**2


def main(L, N, dt, tmax, pos0, vel0, reduccion_velocidad=None, dir=None, freq=20):

    # Nombres de todos los archivos a guardar
    archivos = {
    "fout" : "posiciones.dat",
    "graph_energias" : "energias",
    "graph_vel_modulo" : "velocidad_modulo",
    "graph_vel_x" : "velocidad_x",
    "graph_vel_y" : "velocidad_y",
    "temperaturas" : "temperaturas.txt"
    }
    if dir:
        for key in archivos.keys():     
            archivos[key] = dir + archivos[key]


    # Variables para almacenar los resultados
    r_data = np.empty((round(tmax/dt)+2, N, 2))
    v_data = np.empty((round(tmax/dt)+2, N, 2))

    r_data[0] = pos0
    v_data[0] = vel0

    # Cálculo de la aceleración inicial
    acel = lennard_jones(L, pos0)

    # Inicio del bucle de la simulación
    f = safe_open_w(archivos["fout"])
    np.savetxt(f, pos0, delimiter=", ")
    f.write("\n")

    t = 0
    contador = 0
    pos = pos0
    vel = vel0

    while t < tmax:

        # Algoritmo de Verlet
        pos, vel, acel = Verlet(L, pos, vel, dt, acel)

        # Guardar los datos
        r_data[contador+1] = pos
        v_data[contador+1] = vel

        if reduccion_velocidad:
            reescalamiento, tiempos = reduccion_velocidad
            for tiempo in tiempos:
                if contador == int(tiempo/dt):
                    vel /= reescalamiento
                    print(f"reescalado en {tiempo}")


        # Escribir los datos en fichero (cada 10 pasos)
        if contador % freq == 0:
            np.savetxt(f, pos, delimiter=", ")
            f.write("\n")

        t += dt
        contador += 1

    f.close()

    T_equiparticion = grafica_temperatura(v_data, dt, tmax, name_graph=archivos["graph_energias"])


    # ----------- HISTOGRAMAS Y DISTRIBUCIONES DE VELOCIDADES ----------------
    
    # t=0


    # t=20 - t=50
    t1 = round(20/dt)
    t2 = round(50/dt)

    # Módulo de la velocidad

    modulo_v = np.sqrt(v_data[t1:t2,:,0]**2 + v_data[t1:t2,:,1]**2).flatten()
    T_ajustada1 = histograma(modulo_v, T_equiparticion, "maxwell", archivos["graph_vel_modulo"])
    #print(f"Temperatura ajustada a la distribución maxwell: {T_ajustada1}")


    # Componentes de la velocidad

    v_x = v_data[t1:t2,:,0].flatten()
    T_ajustada2 = histograma(v_x, T_equiparticion, "norm", archivos["graph_vel_x"])
    #print(f"Temperatura ajustada a la distribución normal (x): {T_ajustada2}")
    
    v_y = v_data[t1:t2,:,1].flatten()
    T_ajustada3 = histograma(v_y, T_equiparticion, "norm", archivos["graph_vel_y"])
    #print(f"Temperatura ajustada a la distribución normal (y): {T_ajustada3}\n")


    with open(archivos["temperaturas"], "w") as f:
        f.write(f"Temperatura de equipartición: {T_equiparticion}\n")
        f.write(f"Temperatura ajustada a la distribución maxwell: {T_ajustada1}\n")
        f.write(f"Temperatura ajustada a la distribución normal (x): {T_ajustada2}\n")
        f.write(f"Temperatura ajustada a la distribución normal (y): {T_ajustada3}")

    return






# PROGRAMA PRINCIPAL
#---------------------------------------------------------------------------------------

path = os.path.dirname(os.path.abspath(__file__)) + "/"

# PARÁMETROS INICIALES 
L = 4.  # longitud de la caja
N = 16   # número de partículas

# AJUSTES DE LA SIMULACIÓN
dt = 0.002    # paso temporal
tmax = 60    # tiempo total de simulación

# Cálculo de posiciones iniciales en red cuadrada y en reposo
pos0 = posiciones_iniciales(N, L, shape="cuadrado")
vel0 = np.zeros_like(pos0)

dir = path + "red_cuadrada/"

#main(L, N, dt, tmax, pos0, vel0, dir=dir)


# Cálculo de posiciones iniciales aleatorias y en reposo
tmax = 200
dt = 0.002
pos0 = posiciones_iniciales(N, L, shape="aleatorio")

dir = path + "posiciones_aleatorias/"
reduccion_vel = (1.5, np.arange(20, 180, 20))

main(L, N, dt, tmax, pos0, vel0, reduccion_velocidad=reduccion_vel, dir=dir, freq=50)


# Cálculo de posiciones iniciales en red hexagonal y en reposo
tmax = 200
dt = 0.0002
pos0 = posiciones_iniciales(N, L, shape="hexagonal")

dir = path + "red_hexagonal/"
reduccion_vel = (1.5, np.arange(20, 180, 10))

main(L, N, dt, tmax, pos0, vel0, reduccion_velocidad=reduccion_vel, dir=dir, freq=80)
