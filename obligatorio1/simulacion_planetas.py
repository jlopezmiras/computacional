from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import math
import timeit
from numba import njit


# CONSTANTES FÍSICAS
#---------------------------------------------------------------------------------------
c = 1.496e11 # distancia Tierra-Sol (m)
G = 6.67e-11 # constante de gravedad (Nm²/kg²)
Ms = 1.99e30  # masa del Sol (kg)


# LECTURA DE DATOS
# --------------------------------------------------------------------------------------
# Lee los datos del archivo nfile y devuelve tres vectores:
#   planetas (vector 1D: nplanetas)  --> nombre de cada planeta
#   m (vector 1D: nplanetas) --> la masa de cada planeta (en kg)
#   r0 (vector 2D: nplanetas,2) --> la posición inicial de cada planeta asuminedo que se encuentra en el eje X (y=0) (en m)
#   v0 (vector 2D: nplanetas,2) --> la velocidad inicial de cada planeta asuminedo que se mueve inicialmente en vertical (v_x=0) (en m/s)
#   periodo (vector 1D: nplanetas) --> periodo orbital de cada planeta en dias
# Los datos tienen que estar formateados por columnas de la forma nombre, masa, distancia al sol, velocidad, periodo
# y cada fila es un planeta distinto
#----------------------------------------------------------------------------------------
def leerDatos(nfile):
    
    with open(nfile, "r") as f:
        data = [line.split() for line in f.read().splitlines()]

    # Elimina l aprimera línea (encabezado)
    data.pop(0)

    # Crea los vectores masa, posición inicial y velocidad inicial 
    planetas,m0,r0,v0,periodos = [],[],[],[],[]
    for linea in data:
        planetas.append(linea[0])
        m0.append(float(linea[1]))
        r0.append([float(linea[2]),0])
        v0.append([0,float(linea[3])])
        periodos.append(float(linea[4]))

    return planetas,m0,r0,v0,periodos



# REESCALAMIENTO
# --------------------------------------------------------------------------------------
# Función que reescala los valores a unidades de distancia tierra-sol
# Devuelve los valores rescalados de la masa, la posición y la velocidad
# --------------------------------------------------------------------------------------
def reescalamiento(m,r,v):

    m = m/Ms
    r = r/c
    v = math.sqrt(c/(G*Ms))*v

    return m,r,v



# ESTABILIDAD
# --------------------------------------------------------------------------------------
# Función útil para comprobar la estabilidad del sistema solar
# Recibe como argumentos la posición, la velocidad y los factores de reescalamiento, de forma 
# que se multiplican las posiciones (x_i, y_i) de todos los planetas (i) por el vector reescalamiento 
# (rx_i, ry_i). El vector scale_r tiene que ser un vector 2D de dimensiones (nplanets,2). 
# Ídem para las velocidades.
# En caso de no recibir parámetros en scale_r o scale_v, se crea un reescalamiento de valores aleatorios
#   scale_r --> factores multiplicativos comprendidos entre 0.75,1.25 o -0.75,-1.25
#   scale_v --> factores multiplicativos comprendidos entre 0.95,1.05 o -0.95,-1.05
# --------------------------------------------------------------------------------------
def comprobarEstabilidad(r, v, scale_r=None, scale_v=None):
    
    if not scale_r:
        scale_r = (np.random.rand(len(r),2)/2 + 0.75)*np.random.choice((-1,1),size=(len(r),2))
    if not scale_v:
        scale_v = (np.random.rand(len(v),2)/10 + 0.95)*np.random.choice((-1,1),size=(len(r),2))

    r = r*scale_r
    v = v*scale_v

    return r,v



# CÁLCULO DE ACELERACIÓN
# --------------------------------------------------------------------------------------
# Calcula la aceleración utilizando la ley de la gravitacion de Newton.
#
# Recibe los parámetros:
#   r (nplanets,2) --> vector de vectores posicion (reescalados) de cada planeta 
#   m (nplanets)   --> vector de la masa (reescalada) de cada planeta 
#
# Devuelve el vector de todos los vectores aceleración de los planetas, teniendo en cuenta
# que hay un sol de masa reescalada 1 en el centro
# --------------------------------------------------------------------------------------
# Utiliza el decorador @njit del módulo numba para ser compilado en tiempo real y 
# mejorar el coste de tiempo
@njit
def calculaAceleracion(m,r):
    
    n = len(r) # número de planetas

    a = np.empty((n,2)) # declaro el vector de aceleraciones a devolver de dimensiones (nplanets,2)
    # Un bucle for para hallar la aceleración de cada planeta i
    for i in range(n):
        dist = math.sqrt(r[i,0]*r[i,0] + r[i,1]*r[i,1]) # distancia al sol (0,0)
        a[i,0] = -r[i,0]/(dist*dist*dist)   # componente x de la aceleración
        a[i,1] = -r[i,1]/(dist*dist*dist)   # componente y de la aceleración
        # Suma de las aceleraciones producidas por todos los demás planetas (distintos de i)
        for j in range(n):
            if j!=i:
                r_ij_x = r[i,0]-r[j,0]  # componente x de la posición relativa ij
                r_ij_y = r[i,1]-r[j,1]  # componente y de la posición relativa ij
                dist = math.sqrt( r_ij_x*r_ij_x + r_ij_y*r_ij_y ) # distancia entre planetas ij
                a[i,0] -= m[j]*r_ij_x/(dist*dist*dist)  # componente x de la aceleración
                a[i,1] -= m[j]*r_ij_y/(dist*dist*dist)  # componente y de la aceleración
    return a




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
def Verlet(m,r,v,h,a):
    
    w = v + 0.5*h*a   # parámetro necesario para los siguientes cálculos
    r += h*w    # posiciones actualizadas de los planetas con paso h
    a = calculaAceleracion(m,r)   # aceleración actualizada a partir de las nuevas posiciones
    v = w + 0.5*h*a   # velocidades actualizadas con las nuevas aceleraciones

    return r,v,a




# CÁLCULO DE LOS PERIODOS
# --------------------------------------------------------------------------------------
# Función que calcula los periodos de los planetas en tiempo real, por lo que hay que llamarla en cada paso
#
# Recibe los siguientes parámetros:
#   xpos_i (vector 1D: nplanets) --> vector booleano que indica si el planeta está inicialmente en y>0 (True) o y<0 (False)
#   r (vector 2D: nplanets,2) --> vector de vectores posicion (reescalados) de cada planeta 
#   step_count (vector 1D: nplanets) --> vector de enteros que cuenta los pasos antes de que el planeta cruce y=0
#   period_count (vector 2D: nplanets,2) --> vector que almacena un par (numero de pasos totales, numero de vueltas) para cada planeta
#
# A partir de la situación inicial del planeta (y>0 o y<0) la función cuenta los pasos totales que se van dando hasta que se
# invierte dicha situación, esto es, cruza por y=0. En ese momento, invierte la situación inicial del planeta (vector
# xpos_i) y añade a period_count el número de pasos que ha tardado en cruzar y=0. Así, period_count lleva una cuenta
# de cuantas veces el planeta ha cruzado y=0 y cuantos pasos ha tardado. El periodo se calculará finalmente como 
# el cociente entre estos dos números dividido entre 2, ya que en un periodo el planeta cruza dos veces y=0.
# De esta forma se tiene en cuenta el recorrido del planeta a lo largo de toda la simulación y no solo la primera
# vuelta.
# --------------------------------------------------------------------------------------
# Utiliza el decorador @njit del módulo numba para ser compilado en tiempo real y 
# mejorar el coste de tiempo
@njit
def calculaPeriodos(xpos_i, r, step_count, period_count):

    ypositiva = r[:,1]>0.   # vector booleano que muestra el planeta está en y>0 (True) o y<0 (False)
    # Bucle para cada planeta
    for i in range(len(ypositiva)):

        # Se suma un paso
        step_count[i]+=1
        # Si se ha invertido la situación del planeta (ha pasado de y>0 a y<0 o al revés)
        if ypositiva[i] is not xpos_i[i]:

            xpos_i[i] = not xpos_i[i]   # cambio la situación inicial del planeta para los siguientes pasos
            period_count[i] += np.array([step_count[i], 1])  # añado 1 a la cuenta de periodos y los pasos que ha tardado el planeta
            step_count[i] = 0   # inicializo el contador de pasos del planeta a 0

    return
            



# CÁLCULO DE LAS ENERGÍAS CINÉTICAS
# --------------------------------------------------------------------------------------
# Función que calcula las energías cinéticas de los planetas en tiempo real, por lo que hay que llamarla en cada paso
#
# Recibe los siguientes parámetros:
#   m (vector 1D: nplanets)   --> vector de la masa (reescalada) de cada planeta 
#   r (vector 2D: nplanets,2) --> vector de vectores posicion (reescalados) de cada planeta 
#
# Calcula una cantidad proporcional a la energía cinética, es decir, unidades arbitrarias de energía, teniendo
# en cuenta la velocidad del planeta. Devuelve el vector energia que almacena la energía de cada planeta en
# este instante
# --------------------------------------------------------------------------------------
# Utiliza el decorador @njit del módulo numba para ser compilado en tiempo real y 
# mejorar el coste de tiempo
@njit
def calculaEnergia(m, v):

    energia = np.zeros_like(m) 
    for i in range(len(m)):
        energia[i] = m[i]*(v[i,0]*v[i,0] + v[i,1]*v[i,1])

    return energia





# PROGRAMA PRINCIPAL
#---------------------------------------------------------------------------------------
if __name__=='__main__':

    # Establezco los nombres de los archivos del que leer los datos iniciales y donde guardar los resultados
    filein = "datos_iniciales.txt"  
    fileout = "planets_data.dat"   
    file_periodos = "periodos.txt"
    file_energia = "grafica_energias"

    # Establezco los parámetros iniciales de la simulación
    h = 1e-2  # paso
    tmax = 4e3  # tiempo que dura la simulacion
    iterations = int(tmax/h)  # numero de iteraciones totales

    planetas, m0, r0, v0, periodos_reales = leerDatos(filein) 
    m,r,v = reescalamiento(np.array(m0),np.array(r0),np.array(v0))

    nplanets = len(m)

    # Se pueden modificar los datos iniciales para comprobar la estabilidad
    # scale_r = np.array([[1,0],[1.1,0],[1.2,0],[1.3,0],[1.4,0],[1.5,0],[1.6,0],[1.7,0]])
    #r,v = comprobarEstabilidad(r,v) 
    

    f = open(fileout, "w")

    # Calculo la primera aceleracion para pasársela al algoritmo Verlet
    a = calculaAceleracion(m,r) 

    # Establezco el vector de situación inicial de los planetas  según si la velocidad en y es
    # positiva o negativa para utilizarlo en la función de periodos
    ypositiva_i = v[:,1]>0.

    # Inicializo los vectores necesarios para las funciones periodo y energia
    step_count = np.zeros(nplanets, dtype=int)
    periodo_count = np.zeros((nplanets,2), dtype=int)
    energia = np.empty((iterations, nplanets))  # almacenará el vector de energías durante toda la simulación para graficarlo


    # Empiezo el bucle de la simulación
    t=0
    contador = 0
    while t<tmax:
    
        r,v,a = Verlet(m,r,v,h,a)

        calculaPeriodos(ypositiva_i,r,step_count,periodo_count)
        energia[contador] = calculaEnergia(m,v)

        # Guardo 1 de cada 100 pasos en el archivo para la animación posterior
        if contador%100==0:
            
            np.savetxt(f, r, delimiter=", ")
            f.write("\n")

        t+=h
        contador+=1

    f.close()


    # ESCRIBIR EN UN FICHERO EL PERIODO DE LOS PLANETAS
    # ----------------------------------------------------------------

    periodos = periodo_count[:,0]//periodo_count[:,1]*math.sqrt(c**3/(G*Ms))/3600/24*h*2

    periodos_reales = np.array(periodos_reales)

    # Formateo los resultados para escribirlos
    planetas = [name.ljust(8," ") for name in planetas]
    periodos = np.round(periodos, 4)

    # Calculo diferencias y diferencias relativas
    dif_periodos = np.round(periodos_reales-periodos, 4)
    dif_rel_periodos = np.round(np.abs(dif_periodos)/periodos_reales*100, 4)


    # Escribo los resultados en el fichero file_periodos
    with open(file_periodos, "w") as f:
        f.write("Planeta \tPeriodo calculado\tPeriodo real\tDiferencia\tDiferencia relativa (%)\n\n")
        for i in range(len(r)):
            f.write(planetas[i] + "\t")
            f.write(str(periodos[i]).ljust(17," ") +"\t")
            f.write(str(periodos_reales[i]).ljust(12," ") + "\t")
            f.write(str(dif_periodos[i]).ljust(10," ") + "\t")
            f.write(str(dif_rel_periodos[i])+"\n")



    # GRAFICAR LA ENERGÍA CINÉTICA DE CADA PLANETA
    # ----------------------------------------------------------------

    # Hacer reescalamiento del tiempo
    t = np.linspace(0,tmax,iterations)*math.sqrt(c**3/(G*Ms))/3600/24/365

    fig = plt.figure(num=1, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)

    # Hacer la gráfica de cada planeta en unidades de energía arbitrarias
    for i in range(len(r)):
        ax.plot(t, energia[:,i]*1e4, label=planetas[i])

    # Ajustes de la gráfica
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax.legend(loc='upper left', bbox_to_anchor=(1, 1),fancybox=True, shadow=True)
    plt.xlabel("Tiempo (años)")
    plt.ylabel("Energía cinética")
    plt.yscale("log")
    plt.title("Energía cinética del sistema solar",fontsize=16,fontweight='bold')

    fig.savefig(file_energia)
