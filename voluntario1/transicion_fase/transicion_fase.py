import os, os.path
import numpy as np
import math as m
from numba import njit
from matplotlib import pyplot as plt

# Función para abrir archivos y crear carpetas si no existen
def safe_open_w(path):
    ''' Open "path" for writing, creating any parent directories as needed.
    ''' 
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w')

# Función que evalúa el potencial de lennard Jones dado el vector
# r que almacena todas las posiciones de todas las partículas
# Tiene en cuenta las condiciones de contorno periódicas
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

# Función que calcula la distancia entre dos partículas en la caja con las condiciones de 
# contorno periódicas
@njit
def calcula_distancia(L, r1, r2):

    r_ij_x = np.array([r1[0]-r2[0], r1[0]-r2[0]+L, r1[0]-r2[0]-L])
    r_ij_y = np.array([r1[1]-r2[1], r1[1]-r2[1]+L, r1[1]-r2[1]-L])

    r_ij = np.array([r_ij_x[np.argmin(np.abs(r_ij_x))], r_ij_y[np.argmin(np.abs(r_ij_y))]])

    distancia = m.sqrt(r_ij[0]**2 + r_ij[1]**2)

    return distancia, r_ij


# ALGORITMO DE VERLET (RESOLUCIÓN DE LA ECUACIÓN DIFERENCIAL)
def Verlet(L, r, v, h, a, real_pos):
    
    w = v + 0.5*h*a   
    r += h * w    # posiciones actualizadas de las partículas con paso h
    r = r % L
    a = lennard_jones(L, r)   # aceleración actualizada a partir de las nuevas posiciones
    v = w + 0.5*h*a   # velocidades actualizadas con las nuevas aceleraciones
    real_pos += h * w # la posición sin tener en cuenta condiciones de contorno

    return r, v, a, real_pos


# Función para determinar las posiciones iniciales en función de si son aleatorias o 
# se quieren en red cuadrada o hexagonal
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


# Función para calcular promedios de un vector x tomando intervalos con
# un número determinado de puntos (n_puntos)
def promedios_temporales(x, n_puntos):

    extra = len(x) % n_puntos
    y_mod = x[0:-extra].reshape(-1, n_puntos).mean(axis=1)

    return y_mod

# Función que realiza gráfica de la temperatura a lo largo del tiempo
def grafica_temperatura(N, v, dt, tmax, name_graph):

     # CÁLCULO DE LAS ENERGÍAS
    energia_cin = 0.5 * np.sum(v[:,:,0]**2 + v[:,:,1]**2, axis=1)

    # GRÁFICA DE LAS ENERGÍAS
    t = np.arange(0,tmax+dt,dt)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    t = promedios_temporales(t, n_puntos=500)
    temperatura = promedios_temporales(energia_cin, n_puntos=500) / N

    ax.plot(t, temperatura, color="blue")

    ax.grid("--", alpha=0.5)

    plt.xlabel("Tiempo")
    plt.ylabel("Temperatura")

    plt.title("Evolución temporal de la temperatura", fontweight="bold")


    fig.savefig(name_graph)

    # fig.show()

    # CÁLCULO DE LA TEMPERATURA POR TEOREMA DE EQUIPARTICIÓN
    T_equiparticion = np.average(energia_cin[round(20/dt):round(50/dt)])/N

    return T_equiparticion


# Gráficas de los desplazamientos cuadrados. Si solo se da un número de partícula,
# se realiza la gráfica de desplazamiento cuadrado. Si se dan dos, se realiza
# la gráfica de separación cuadrática entre ambas
def grafica_desplazamiento_cuadrado(r_data, particulas, dt, tmax, name_graph):

    t = np.arange(0, tmax+dt, dt)
    

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Separación cuadrática si hay dos partículas (la variable partículas es una lista)
    if isinstance(particulas, list):
        desplazamiento = r_data[:,particulas[0]] - r_data[:,particulas[1]]

        plt.ylabel("$<(r_i-r_j)^2>$")
        plt.title("Separación media cuadrática entre dos átomos", fontweight="bold")

    # Desplazamiento cuadrático si hay solo una partícula (la variable partículas es un número)
    else:
        desplazamiento = r_data[:,particulas] - r_data[0,particulas,:]
        
        plt.ylabel("$<(r-r_0)^2>$")
        plt.title("Desplazamiento medio cuadrático de un átomo", fontweight="bold")

    desplazamiento_cuadrado = desplazamiento[:,0]**2 + desplazamiento[:,1]**2

    # Hacemos medias para que los datos se visualicen mejor, de forma que solo
    # se tengan 300 puntos
    t = promedios_temporales(t, n_puntos=500)
    desplazamiento_cuadrado = promedios_temporales(desplazamiento_cuadrado, n_puntos=500)

    ax.plot(t, desplazamiento_cuadrado, color="blue")

    ax.grid("--", alpha=0.5)

    plt.xlabel("Tiempo")
    
    plt.savefig(name_graph)

    return



def main(L, N, dt, tmax, pos0, vel0, cambio_velocidad=None, particula=0, archivos=None, dir=None, freq=20):

    if dir:
        for key in archivos.keys():     
            archivos[key] = dir + archivos[key]


    # Variables para almacenar los resultados
    r_data = np.empty((round(tmax/dt)+1, N, 2))
    v_data = np.empty((round(tmax/dt)+1, N, 2))
    real_r_data = np.empty((round(tmax/dt)+1, N, 2))

    r_data[0] = pos0
    v_data[0] = vel0
    real_r_data[0] = pos0

    # Cálculo de la aceleración inicial
    acel = lennard_jones(L, pos0)

    # Inicio del bucle de la simulación
    f = safe_open_w(archivos["fout"])
    f.write(f"{tmax}\n\n")
    np.savetxt(f, pos0, delimiter=", ")
    f.write("\n")

    t = 0
    contador = 0
    pos = pos0
    real_pos = pos0
    vel = vel0

    while t < tmax:

        # Se puede dar un cambio de velocidad en ciertos tiempos para cambiar la temperatura manualmente
        # cambio_velocidad es un alista
        # El primer elemento es el factor de aumento
        # El segundo es una lista de los tiempos en los que ha de producirse
        if cambio_velocidad:
            reescalamiento, tiempos = cambio_velocidad
            for tiempo in tiempos:
                if contador == int(tiempo/dt):
                    vel *= reescalamiento

        # Algoritmo de Verlet
        pos, vel, acel, real_pos = Verlet(L, pos, vel, dt, acel, real_pos)

        # Guardar los datos
        r_data[contador+1] = pos
        v_data[contador+1] = vel
        real_r_data[contador+1] = real_pos

        # Escribir los datos en fichero (cada 10 pasos)
        if contador % freq == 0:
            np.savetxt(f, pos, delimiter=", ")
            f.write("\n")

        t += dt
        contador += 1

    f.close()

    T_equiparticion = grafica_temperatura(N, v_data, dt, tmax, name_graph=archivos["graph_temperatura"])

    grafica_desplazamiento_cuadrado(real_r_data, particula, dt, tmax, name_graph=archivos["graph_desplazamiento_cuadrado"])
   




# PROGRAMA PRINCIPAL
#---------------------------------------------------------------------------------------

path = os.path.dirname(os.path.abspath(__file__)) + "/"

# PARÁMETROS INICIALES 
L = 4.  # longitud de la caja
N = 16   # número de partículas

# AJUSTES DE LA SIMULACIÓN
dt = 0.002    # paso temporal
tmax = 60     # tiempo total de simulación

# NOMBRES DE TODOS LOS ARCHIVOS A GUARADAR
archivos = {
    "fout" : "posiciones.dat",
    "graph_temperatura" : "temperatura",
    "graph_desplazamiento_cuadrado" : "desplazamiento_medio_cuadrado"
    }

# Cálculo de posiciones iniciales en red cuadrada y en reposo
pos0 = posiciones_iniciales(N, L, shape="cuadrado")
vel0 = np.zeros_like(pos0)

dir = path + "transicion_rapida/"

particula = 4
cambio_velocidad = (1.5, [20, 30, 35, 45])


main(L, N, dt, tmax, pos0, vel0, cambio_velocidad, particula, archivos=archivos, dir=dir)


# ---------------------- TRANSICIÓN LENTA -----------------------------

archivos = {
    "fout" : "posiciones.dat",
    "graph_temperatura" : "temperatura",
    "graph_desplazamiento_cuadrado" : "separacion_media_cuadrada"
    }

# AJUSTES DE LA SIMULACIÓN
dt = 0.002    # paso temporal
tmax = 520     # tiempo total de simulación

# Cálculo de posiciones iniciales en red cuadrada y en reposo
pos0 = posiciones_iniciales(N, L, shape="cuadrado")
vel0 = np.zeros_like(pos0)

dir = path + "transicion_lenta/"

particula = [1,2]
cambio_velocidad = (1.1, [60, 120, 180, 240, 300, 360, 420, 480])

main(L, N, dt, tmax, pos0, vel0, cambio_velocidad, particula, archivos=archivos, dir=dir, freq=60)
