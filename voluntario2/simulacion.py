import numpy as np
import matplotlib.pyplot as plt
import math as m
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


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

# Cambio de coordenadas de posición cartesianas a polares
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

# Cambio de coordenadas de posición polares a cartesianas
def polares_to_cartesianas(r, phi):
    return r*m.cos(phi), r*m.sin(phi)

# Cambio de coordenadas de velocidad cartesianas a polares
def vel_cartesianas_to_polares(x, y, vx, vy):
    vr = (x*vx + y*vy) / m.sqrt(x**2 + y**2)
    vphi = (x*vy - y*vx) / (x**2 + y**2)
    return vr, vphi

# Cambio de coordenadas de velocidad polares a cartesianas
def vel_polares_to_cartesianas(r, phi, vr, vphi):
    vx = vr * m.cos(phi) - vphi / r * m.sin(phi)
    vy = vr * m.sin(phi) + vphi / r * m.cos(phi)
    return vx, vy


# Ecuaciones del movimiento para el meteorito (solo gravedad terrestre)
def movimiento_meteorito(meteorito, t):

    r, phi, p_r, p_phi = meteorito

    dr = p_r
    dphi = 0
    dp_r = - K * 1/(r*r)
    dp_phi = 0

    return dr, dphi, dp_r, dp_phi


# Ecuaciones del movimiento para los trozos de meteorito (gravedad terrestre y lunar)
def movimiento_trozo_meteorito(meteorito, phi_L, t):

    r, phi, p_r, p_phi = meteorito

    r_L = m.sqrt(1 + r*r - 2*r*m.cos(phi-phi_L-W*t))

    dr = p_r 
    dphi = p_phi / (r*r)
    dp_r = p_phi*p_phi / (r*r*r)
    dp_r -= K * (1/(r*r) + M_RED_L/(r_L*r_L*r_L)*(r-m.cos(phi-phi_L-W*t)))
    dp_phi = - K * M_RED_L * r / (r_L*r_L*r_L) * m.sin(phi-phi_L-W*t)

    return dr, dphi, dp_r, dp_phi


# Ecuaciones del movimiento para la nave (atracción por Tierra, Luna y meteorito)
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


# Metodo de runge kutta para los trozos de meteorito
# Devuelve las coordenadas (r, phi, p_r, p_phi) en el tiempo t+h
def runge_kutta_4_trozos(trozo_meteorito, luna, t, h):

    meteorito = np.array(trozo_meteorito)
    
    k1 = h * np.array(movimiento_trozo_meteorito(meteorito, luna, t))
    k2 = h * np.array(movimiento_trozo_meteorito(meteorito+k1/2, luna, t))
    k3 = h * np.array(movimiento_trozo_meteorito(meteorito+k2/2, luna, t))
    k4 = h * np.array(movimiento_trozo_meteorito(meteorito+k3, luna, t))

    meteorito += (k1 + 2*k2 + 2*k3 + k4)
    t += h

    return meteorito, t


# Método de runge kutta para el meteorito y la nave
# Devuelve las coordenadas (r, phi, p_r, p_phi) en el tiempo t+h
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


# Función que da impulso al cohete un ángulo theta
def dar_impulso(cohete, impulso, theta=None):

    r, phi, p_r, p_phi = cohete

    # Hallo las componentes cartesianas de la posición
    x, y = polares_to_cartesianas(r, phi)

    # Hallo las componentes cartesianas del momento
    p_x, p_y = vel_polares_to_cartesianas(r, phi, p_r, p_phi)

    # Si no se da el ángulo, asumimos que el impulso se
    # da en la dirección del movimiento,
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

v_detonacion = 368.792/DTL  #velocidad perpendicular que se da a los trozos


# Meteorito
r = 10
phi = m.pi/2
p_r = -5000/DTL
p_phi = 0

meteorito = [r, phi, p_r, p_phi]



# PARÁMETROS DE LA SIMULACIÓN
h = 1
iter = int(1e6)
contador = 0
fout = "simulacion.dat" # archivo donde guardar los datos

f = open(fout, "w")

# Vectores de almacenamiento para las graficas
trayectorias = [[], [], []]
dist_meteorito = []
gasto_energetico = [0, cohete[2]**2 + cohete[3]**2/cohete[0]**2]

# Variable que determina si se ha aterrizado
aterrizaje = False

# Se empieza lasimulación. Si no se aterriza, la simulación terminará
# tras un número de iteraciones

while contador < iter and not aterrizaje:

    cohete, meteorito, t = runge_kutta_4(cohete, meteorito, phi_L, t, h)

    # Distancias a la Luna y al meteorito

    r_L = m.sqrt(1 + cohete[0]*cohete[0] - 2*cohete[0]*m.cos(cohete[1]-phi_L-W*t))
    r_M = m.sqrt(meteorito[0]*meteorito[0] + cohete[0]*cohete[0] - 2*cohete[0]*meteorito[0]*m.sin(cohete[1]))


    # Añadimos el gasto energético en este instante y determinamos si se da un impulso o no
    gasto_energetico.append(gasto_energetico[-1])

    gasto_energetico_impulso = -cohete[2]**2 - cohete[3]**2/cohete[0]**2
    if t==19637:
        print(f"Primer impulso \t Tiempo: 19637 \t Distancia a la Luna: {r_L}")
        cohete[2], cohete[3] = dar_impulso(cohete, -2.28e-6, 1.055)
    elif t==101900:
        print(f"Segundo impulso\t Tiempo: 101900 \t Distancia al meteorito: {r_M}")
        cohete[2], cohete[3] = dar_impulso(cohete, 2.28e-6, -m.pi/2)
    elif t==101901:
        cohete[2], cohete[3] = dar_impulso(cohete, 2.28e-6, -m.pi/2)
    elif t==101902:
        cohete[2], cohete[3] = dar_impulso(cohete, 2.28e-6, -m.pi/2)
    elif t==101903:
        cohete[2], cohete[3] = dar_impulso(cohete, 2.28e-6, -m.pi/2)
    elif t==101904:
        cohete[2], cohete[3] = dar_impulso(cohete, 2.28e-6, -m.pi/2)
    elif t==101905:
        cohete[2], cohete[3] = dar_impulso(cohete, 2.28e-6, -m.pi/2)
    elif t==101906:
        cohete[2], cohete[3] = dar_impulso(cohete, 2.28e-6, -m.pi/2+0.03)


    gasto_energetico_impulso += cohete[2]**2 + cohete[3]**2/cohete[0]**2 
    gasto_energetico[-1] += abs(gasto_energetico_impulso)


    # Se comprueba si se llega a aterrizar (encontrarse a menos de 10 km del meteorito)

    if r_M < RMET/DTL:
        aterrizaje = True
        print("\nATERRIZAJE")
        print(f"Distancia: {r_M}")
        print(f"Tiempo: {t}")
        print(f"Distancia Tierra-meteorito: {meteorito[0]}")
        t_mision = t
        v_meteorito = np.array(vel_polares_to_cartesianas(*meteorito))
        v_cohete = np.array(vel_polares_to_cartesianas(*cohete))
        vel_relativa = v_cohete - v_meteorito
        pos_relativa = np.array(polares_to_cartesianas(cohete[0], cohete[1]))
        pos_relativa -= np.array(polares_to_cartesianas(meteorito[0], meteorito[1]))
        print(f"Posición relativa: {pos_relativa}")
        print(f"Velocidad relativa: {vel_relativa}")
        print(f"Velocidad del meteorito: {v_meteorito}")
        print(f"Gasto energético total: {gasto_energetico[-1]}")
        

    # Cada 150 pasos se almacenan los datos en el archivo .dat para la animación
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

        dist_meteorito.append(r_M)

        trayectorias[0].append(np.array([x_cohete, y_cohete]))
        trayectorias[1].append(np.array([x_meteorito, y_meteorito]))
        trayectorias[2].append(np.array([x_luna, y_luna]))
    
    contador += 1

# Gráfica de trayectorias
trayectorias = np.array(trayectorias)
fig, ax = plt.subplots()
ax.plot(trayectorias[0,:,0], trayectorias[0,:,1])
plt.xlabel(r"$x/d_{TL}$")
plt.ylabel(r"$y/d_{TL}$")
plt.title("Trayectoria de la nave", fontweight="bold")

plt.savefig("grafica_trayectorias")
# plt.show()
plt.close(fig)


# Gráfica de distancia al meteorito

t_vector = np.array([i for i in range(len(dist_meteorito))])*150

fig, ax = plt.subplots()
ax.plot(t_vector / 3600, dist_meteorito)

plt.xlabel("Tiempo (h)")
plt.ylabel("Distancia $(d_{TM})$")

plt.title("Distancia entre la nave y el meteorito", fontweight="bold")

plt.savefig("grafica_distancia_meteorito")
# plt.show()
plt.close(fig)



# Gráfica de gasto energético

fig, ax = plt.subplots()
t_vector = np.arange(-1, t_mision+1) / 3600
gasto_energetico = np.array(gasto_energetico) * DTL * DTL / 1e6 / 2
ax.plot(t_vector, gasto_energetico)

plt.xlabel("Tiempo (h)")
plt.ylabel("Energía (MJ/kg)")
plt.title("Gasto energético acumulado", fontweight="bold")

plt.savefig("grafica_gasto_energetico")
# plt.show()
plt.close(fig)


# Si se ha aterrizado, se produce la explosión y los trozos siguen moviéndose

if aterrizaje:


    h = 1
    iter = int(1e5)

    contador = 0

    # Se divide en dos meteoritos
    # Condiciones iniciales
    r = meteorito[0]
    phi = meteorito[1]
    p_r = meteorito[2]
    p_phi_1 = v_detonacion*r
    p_phi_2 = -v_detonacion*r

    v_detonacion *= 1.2

    meteorito1 = [r, phi, p_r, p_phi_1]
    meteorito2 = [r, phi, p_r, p_phi_2]

    distancia_Tierra_1 = [r]
    distancia_Tierra_2 = [r]


    while contador < iter:

        meteorito1, _ = runge_kutta_4_trozos(meteorito1, phi_L, t, h)
        meteorito2, t = runge_kutta_4_trozos(meteorito2, phi_L, t, h)

        if contador%20 == 0:

            distancia_Tierra_1.append(meteorito1[0])
            distancia_Tierra_2.append(meteorito2[0])

        if contador%150 == 0:

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


    # Gráfica de distancia a la Tierra

    distancia_Tierra_1 = np.array(distancia_Tierra_1)
    distancia_Tierra_2 = np.array(distancia_Tierra_2)
    t_vector = np.array([i for i in range(len(distancia_Tierra_1))]) * 20 / 3600

    radio_T = np.zeros_like(t_vector) + RT/DTL
    atmosfera = np.zeros_like(t_vector) + (10e6 + RT)/DTL
    satelite = np.zeros_like(t_vector) + (36e6 + RT)/DTL
    luna = np.zeros_like(t_vector) + 1

    fig, ax = plt.subplots()
    ax.plot(t_vector, distancia_Tierra_1, alpha=0.8, label="Trozo 1")
    ax.plot(t_vector, distancia_Tierra_2, alpha=0.8, label="Trozo 2")
    ax.plot(t_vector, radio_T, alpha=0.8, linestyle="--", label="Radio terrestre")
    ax.plot(t_vector, atmosfera, alpha=0.8, linestyle="--", label="Fin de la atmósfera")
    ax.plot(t_vector, satelite, alpha=0.8, linestyle="--", label="Órbita geocéntrica")
    ax.plot(t_vector, luna, alpha=0.8, linestyle="--", label="Distancia Tierra-Luna")

    axins = zoomed_inset_axes(ax, 10, loc=2)
    axins.plot(t_vector, distancia_Tierra_1, alpha=0.8)
    axins.plot(t_vector, distancia_Tierra_2, alpha=0.8)
    axins.plot(t_vector, radio_T, alpha=0.8, linestyle="--")
    axins.plot(t_vector, atmosfera, alpha=0.8, linestyle="--")
    axins.plot(t_vector, satelite, alpha=0.8, linestyle="--")
    axins.plot(t_vector, luna, alpha=0.8, linestyle="--")

    x1, x2, y1, y2 = 22000/3600, 26500/3600, 0, 0.3
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    axins.set_xticks([])
    axins.set_yticks([])

    mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5")

    ax.set_ylabel("Distancia a la Tierra ($d_{TL}$)")
    ax.set_xlabel("Tiempo (h)")

    ax.set_ylim(-0.1, 6.5)

    ax.set_title("Trayectorias de los trozos de meteorito (E=141,61 Gt)", fontweight="bold")

    ax.legend(loc="lower right")

    plt.draw()


    plt.savefig(f"grafica_trayectoria_trozos")
    # plt.show()
    plt.close(fig)

    print(f"Distancia minima: {min([min(distancia_Tierra_1), min(distancia_Tierra_2)])}")

f.close()







