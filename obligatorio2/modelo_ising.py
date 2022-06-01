import numpy as np
import math
from numba import njit
import timeit



@njit
def ising(temp,espines,steps,fout=None):
    
    # Dimensiones de la red NxN
    N = np.shape(espines)[0]

    # Vector que almacena la configuración de espines en todos los pasos montecarlo
    Stot = np.empty((steps+1,N,N))
    Stot[0] = espines

    for pasoMc in range(steps):
        # Un paso Monte Carlo son N² iteraciones
        for i in range(N*N):

            n,m = np.random.randint(0,N,size=2)

            if n!=0 and m!=0 and n!=N-1 and m!=N-1:
                deltaE = 2*espines[n,m]*(espines[n+1,m] + espines[n-1,m] + espines[n,m+1] + espines[n,m-1])
            else: 
                deltaE = 2*espines[n,m]*(espines[(n+1)%N,m] + espines[(n-1)%N,m] + espines[n,(m+1)%N] + espines[n,(m-1)%N])

            p = min(1,math.exp(-deltaE/temp))
            if np.random.random()<p:
                espines[n,m] = -espines[n,m]

        Stot[pasoMc+1] = espines
        
    return Stot


# Parámetros iniciales

N = 128
steps = 300


espines_rd = np.random.choice([-1,1],size=(N,N))
espines_up = np.full((N,N),1)
espines_down = np.full((N,N),-1)


fout = "ising_data_T_baja_desordenado.dat"
temp = 0.01
Stot = ising(temp, espines_rd, steps)

f = open(fout, "w")
for i in range(len(Stot)):
    np.savetxt(f,Stot[i].astype(int),fmt='%s',delimiter=', ')
    f.write('\n')
f.close()



fout = "ising_data_T_baja_ordenado.dat"
temp = 0.01
Stot = ising(temp, espines_up, steps)

f = open(fout, "w")
for i in range(len(Stot)):
    np.savetxt(f,Stot[i].astype(int),fmt='%s',delimiter=', ')
    f.write('\n')
f.close()



fout = "ising_data_T_media_desordenado.dat"
temp = 2.5
Stot = ising(temp, espines_rd, steps)

f = open(fout, "w")
for i in range(len(Stot)):
    np.savetxt(f,Stot[i].astype(int),fmt='%s',delimiter=', ')
    f.write('\n')
f.close()



fout = "ising_data_T_media_ordenado.dat"
temp = 2.5
Stot = ising(temp, espines_up, steps)

f = open(fout, "w")
for i in range(len(Stot)):
    np.savetxt(f,Stot[i].astype(int),fmt='%s',delimiter=', ')
    f.write('\n')
f.close()



fout = "ising_data_T_alta_desordenado.dat"
temp = 4.8
Stot = ising(temp, espines_rd, steps)

f = open(fout, "w")
for i in range(len(Stot)):
    np.savetxt(f,Stot[i].astype(int),fmt='%s',delimiter=', ')
    f.write('\n')
f.close()



fout = "ising_data_T_alta_ordenado.dat"
temp = 4.8
Stot = ising(temp, espines_up, steps)

f = open(fout, "w")
for i in range(len(Stot)):
    np.savetxt(f,Stot[i].astype(int),fmt='%s',delimiter=', ')
    f.write('\n')
f.close()

