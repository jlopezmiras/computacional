import os, os.path
import numpy as np
import math as m
from numba import njit
from matplotlib import pyplot as plt
from scipy.stats import norm, maxwell


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
                
                dist, r_ij = calcula_distancia(L, r[j], r[i])

                if dist < 3:

                    r_ij = r_ij / dist

                    acel[i] -= 24 * (2/dist**13 - 1/dist**7) * r_ij

    return acel



# ALGORITMO DE VERLET (RESOLUCIÓN DE LA ECUACIÓN DIFERENCIAL)
@njit
def Verlet(L, r, v, h, a):
    
    w = v + 0.5*h*a   
    r += h * w    # posiciones actualizadas de las partículas con paso h
    r = r % L
    a = lennard_jones(L, r)   # aceleración actualizada a partir de las nuevas posiciones
    v = w + 0.5*h*a   # velocidades actualizadas con las nuevas aceleraciones

    return r,v,a


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


# Función que calcula la distancia entre dos partículas en la caja con las condiciones de 
# contorno periódicas
@njit
def calcula_distancia(L, r1, r2):

    r_ij_x = np.array([r1[0]-r2[0], r1[0]-r2[0]+L, r1[0]-r2[0]-L])
    r_ij_y = np.array([r1[1]-r2[1], r1[1]-r2[1]+L, r1[1]-r2[1]-L])

    r_ij = np.array([r_ij_x[np.argmin(np.abs(r_ij_x))], r_ij_y[np.argmin(np.abs(r_ij_y))]])

    distancia = m.sqrt(r_ij[0]**2 + r_ij[1]**2)

    return distancia, r_ij


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
                    
                    dist, _ = calcula_distancia(L, r[j], r[i])

                    energia[time] += 4 * (dist**(-12) - dist**(-6))

    return energia



# Función para calcular promedios de un vector x tomando intervalos con
# un número determinado de puntos (n_puntos)
def promedios_temporales(x, n_puntos=300):

    extra = len(x) % n_puntos
    x_mod = x[0:-extra].reshape(-1, n_puntos).mean(axis=1)
    x_mod = np.append(x[0], x_mod)
    x_extra = x[-extra:].mean()

    return np.append(x_mod, x_extra)


# Función para realizar la gráfica de las energías
def grafica_energia(L, r, v, dt, tmax, name_graph, v0):

     # CÁLCULO DE LAS ENERGÍAS
    energia_cin = 0.5 * np.sum(v[:,:,0]**2 + v[:,:,1]**2, axis=1)
    energia_pot = calculo_energia_pot(L, r)
    energia_tot = energia_cin + energia_pot

    # GRÁFICA DE LAS ENERGÍAS
    t = np.arange(0,tmax+dt,dt)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(t, energia_cin, color="orange", lw=0.7, alpha=0.75)
    ax.plot(t, energia_pot, color="blue", lw=0.7, alpha=0.75)
    ax.plot(t, energia_tot, color="#545454", lw=0.6, alpha=0.75)

    n_puntos = int(4./dt)
    t_m = promedios_temporales(t, n_puntos)
    energia_cin_m = promedios_temporales(energia_cin, n_puntos)
    energia_pot_m = promedios_temporales(energia_pot, n_puntos)
    energia_tot_m = promedios_temporales(energia_tot, n_puntos)

    ax.plot(t_m, energia_cin_m, color="orange", label="Energía cinética")
    ax.plot(t_m, energia_pot_m, color="blue", label="Energía potencial")
    ax.plot(t_m, energia_tot_m, color="#525558", label="Energía total")

    ax.grid("--", alpha=0.5)

    plt.legend(loc="lower right")

    plt.xlabel("Tiempo")
    plt.ylabel("Energía")

    plt.title(fr"Conservación de la energía $(v_0 = {v0:.0f})$", fontweight="bold")

    fig.savefig(name_graph)
    # fig.show()

    # CÁLCULO DE LA TEMPERATURA POR TEOREMA DE EQUIPARTICIÓN
    T_equiparticion = np.average(energia_cin[round(20/dt):round(50/dt)])/N

    return T_equiparticion



# Función para realizar los histogramas de las velocidades
def histograma(vel0, v, T_equiparticion, distribution, name_graph, bins=30):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()

    # Histograma de velocidades 
    n, bins, _ = ax.hist(v, bins=bins, color='#707070', alpha=0.7, rwidth=0.85, density=True, label="t=20-50")
    ax2.hist(vel0, bins=7, color='#444444', alpha=0.75, density=False, rwidth=0.3, label="t=0")

    # Elegir distribución entre maxwell y normal
    if distribution == "v":
        stats = maxwell
        x = np.linspace(0, bins.max(), 500)
        ax.set_xlim(0, bins.max())
        ax2.set_xlim(0, bins.max())
        plt.title("Distribución del módulo de la velocidad", fontweight="bold")
        loc="upper right"
        ax.set_xlabel(r"$v$")
        ax.set_ylabel(r"$P(v)$")
        ax2.set_ylabel(r"$P(v_0)$")
        ax2.bar(vel0[0], 20, width=0.018*bins.max(), color='#444444', alpha=0.75)

    else:
        stats = norm
        xmax = max(abs(bins.max()), abs(bins.min()))
        x = np.linspace(-xmax, xmax, 500)
        ax.set_xlim(-xmax, xmax)
        ax2.set_xlim(-xmax, xmax)
        ax2.set_ylim(0,20)
        loc="lower center"

        if distribution == "vx":
            plt.title("Distribución de la componente x de la velocidad", fontweight="bold")
            ax.set_xlabel(r"$v_x$")
            ax.set_ylabel(r"$P(v_x)$")
            ax2.set_ylabel(r"$P(v_{0x})$")
        elif distribution == "vy":
            plt.title("Distribución de la componente y de la velocidad", fontweight="bold")
            ax.set_xlabel(r"$v_y$")
            ax.set_ylabel(r"$P(v_y)$")
            ax2.set_ylabel(r"$P(v_{0y})$")

    params = stats.fit(v, floc=0)

    params2 = stats.fit(v)

    # Gráficas de las distribuciones
    ax.plot(x, stats.pdf(x, 0, m.sqrt(T_equiparticion)), color='#A42684', alpha=0.9,
            label=f"Equip.")
    ax.plot(x, stats.pdf(x, *params), color='orange', alpha=0.9, label=f"Ajuste A")
    ax.plot(x, stats.pdf(x, *params2), color='green', alpha=0.9, label=f"Ajuste B")

    fig.legend(loc='center left', bbox_to_anchor=(0.70, 0.73))

    fig.savefig(name_graph)
    # plt.show()

    return params[1]**2, params2[1]**2


# Función principal
def main(L, N, dt, tmax, pos0, vel0, dir=None):

    # Nombres de todos los archivos a guardar
    archivos = {
    "fout" : "posiciones.dat",
    "graph_energias" : "energias",
    "graph_vel_inicial" : "velocidad_inicial",
    "graph_vel_modulo" : "velocidad_modulo",
    "graph_vel_x" : "velocidad_x",
    "graph_vel_y" : "velocidad_y",
    "temperaturas" : "temperaturas.txt"
    }
    if dir:
        for key in archivos.keys():
            archivos[key] = dir + archivos[key]


    # Variables para almacenar los resultados
    r_data = np.empty((round(tmax/dt)+1, N, 2))
    v_data = np.empty((round(tmax/dt)+1, N, 2))

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

    # Comienzo de la simulación
    while t < tmax:

        # Algoritmo de Verlet
        pos, vel, acel = Verlet(L, pos, vel, dt, acel)

        # Guardar los datos
        r_data[contador+1] = pos
        v_data[contador+1] = vel

        # Escribir los datos en fichero (cada 20 pasos)
        if contador % 20 == 0:
            np.savetxt(f, pos, delimiter=", ")
            f.write("\n")

        t += dt
        contador += 1

    f.close()

    T_equiparticion = grafica_energia(L, r_data, v_data, dt, tmax, 
            name_graph=archivos["graph_energias"], v0=m.sqrt(vel0[0,0]**2+vel0[0,1]**2))



    # ----------- HISTOGRAMAS Y DISTRIBUCIONES DE VELOCIDADES ----------------
    
    # t=0

    modulo_v_ini = np.sqrt(vel0[:,0]**2 + vel0[:,1]**2).flatten()
    v_x_ini = vel0[:,0].flatten()
    v_y_ini = vel0[:,1].flatten()


    # t=20 - t=50
    t1 = round(20/dt)
    t2 = round(50/dt)

    # Módulo de la velocidad

    modulo_v = np.sqrt(v_data[t1:t2,:,0]**2 + v_data[t1:t2,:,1]**2).flatten()
    T_maxwell_1, T_maxwell_2 = histograma(modulo_v_ini, modulo_v, T_equiparticion, "v", archivos["graph_vel_modulo"])


    # Componentes de la velocidad

    v_x = v_data[t1:t2,:,0].flatten()
    T_norm_x_1, T_norm_x_2 = histograma(v_x_ini, v_x, T_equiparticion, "vx", archivos["graph_vel_x"])
    
    v_y = v_data[t1:t2,:,1].flatten()
    T_norm_y_1, T_norm_y_2 = histograma(v_y_ini, v_y, T_equiparticion, "vy", archivos["graph_vel_y"])

    # Se guradan las distintas temperaturas en un archivo
    with open(archivos["temperaturas"], "w") as f:
        f.write(f"Temperatura de equipartición: {T_equiparticion}\n\n")

        f.write(f"Temperatura ajustada a la distribución maxwell 1: {T_maxwell_1}\n")
        f.write(f"Temperatura ajustada a la distribución maxwell 2: {T_maxwell_2}\n\n")


        f.write(f"Temperatura ajustada a la distribución normal 1 (x): {T_norm_x_1}\n")
        f.write(f"Temperatura ajustada a la distribución normal 2 (x): {T_norm_x_2}\n\n")

        f.write(f"Temperatura ajustada a la distribución normal 1 (y): {T_norm_y_1}\n")
        f.write(f"Temperatura ajustada a la distribución normal 2 (y): {T_norm_y_2}\n\n")

    return [T_equiparticion, T_maxwell_1, T_maxwell_2, (T_norm_x_1+T_norm_y_1)/2, (T_norm_x_2+T_norm_y_2)/2]






# PROGRAMA PRINCIPAL
#---------------------------------------------------------------------------------------
if __name__=='__main__':

    # PARÁMETROS INICIALES 
    L = 10.  # longitud de la caja
    N = 20   # número de partículas

    path = os.path.dirname(os.path.abspath(__file__))

    # AJUSTES DE LA SIMULACIÓN
    dt = 0.002    # paso temporal
    tmax = 60     # tiempo total de simulación

    # Cálculo de posiciones iniciales aleatorias
    pos0 = posiciones_iniciales(N, L, minimum_distance=1.0)
    
    # Ángulo inicial de velocidad aleatorio módulo 1
    ang = np.random.uniform(0, 2*m.pi, N)
    vel0 = np.array(list(zip(np.cos(ang), np.sin(ang))))

    dir = path + "/velocidad_1/"
    temp1 = main(L, N, dt, tmax, pos0, vel0, dir=dir)


    # Ángulo inicial de velocidad aleatorio módulo 2
    pos0 = posiciones_iniciales(N, L, minimum_distance=1.0)
    ang = np.random.uniform(0, 2*m.pi, N)
    vel0 = 2*np.array(list(zip(np.cos(ang), np.sin(ang))))

    dir = path + "/velocidad_2/"
    temp2 = main(L, N, dt, tmax, pos0, vel0, dir=dir)


    # Ángulo inicial de velocidad aleatorio módulo 3
    pos0 = posiciones_iniciales(N, L, minimum_distance=1.0)
    ang = np.random.uniform(0, 2*m.pi, N)
    vel0 = 3*np.array(list(zip(np.cos(ang), np.sin(ang))))

    dir = path + "/velocidad_3/"
    temp3 = main(L, N, dt, tmax, pos0, vel0, dir=dir)


    # Ángulo inicial de velocidad aleatorio módulo 4
    pos0 = posiciones_iniciales(N, L, minimum_distance=1.0)
    ang = np.random.uniform(0, 2*m.pi, N)
    vel0 = 4*np.array(list(zip(np.cos(ang), np.sin(ang))))

    dir = path + "/velocidad_4/"
    temp4 = main(L, N, dt, tmax, pos0, vel0, dir=dir)


    # Comparación de temperaturas

    temperaturas = list(zip(temp1, temp2, temp3, temp4))

    labels = ["T. equipartición",
                "Distr. v (A)",
                "Distr. v (B)",
                "Distr. componente v (A)",
                "Distr. componente v (B)"]

    colors = ["#525558", "#ffa43a", "#ffbf75", "#759eff", "#75c7ff"]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    barWidth = 0.12

    x = np.arange(len(temperaturas[0]))
    plt.bar(x, temperaturas[0], color=colors[0], width=barWidth, edgecolor='grey', label=labels[0])

    for i in range(len(temperaturas)-1):
        x = [j + barWidth for j in x]
        plt.bar(x, temperaturas[i+1], color=colors[i+1], width=barWidth, edgecolor ='grey', label=labels[i+1])

    plt.xticks([r + 2*barWidth for r in range(len(temperaturas[0]))],
        [1, 2, 3, 4])

    plt.legend(loc="upper left")

    plt.xlabel("Velocidad inicial")
    plt.ylabel("Temperatura")

    plt.title("Comparación de métodos de cálculo de temperatura", fontweight="bold")

    plt.savefig(path + "/comparacion_temperaturas")

