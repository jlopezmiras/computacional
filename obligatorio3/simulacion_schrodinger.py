import numpy as np
import math as m
import matplotlib.pyplot as plt
from numba import njit

@njit
def schrodinger(N, nciclos, altura_potencial, steps):

    k0 = 2*m.pi*nciclos/N
    s = 1/(4*k0**2)

    phi = np.zeros((steps+1, N+1), dtype=np.complex_) 
    b = np.zeros(N+1, dtype=np.complex_)
    alpha = np.zeros(N, dtype=np.complex_)
    beta = np.zeros(N, dtype=np.complex_)
    chi = np.zeros(N+1, dtype=np.complex_)

    potencial = np.array([altura_potencial*k0**2 if (i>2*N/5 and i<3*N/5) else 0 for i in range(N+1)])

    phi[0] = np.exp(1j*k0*np.arange(N+1)) * np.exp(-8 * (4*np.arange(N+1)-N)**2 / N**2)
    phi[0,0] = 0.0
    phi[0,N] = 0.0

    for i in range(N-1,0,-1):
        alpha[i] = -1/(-2+2j/s - potencial[i+1] + alpha[i+1])

    
    for n in range(steps):
        
        # Calculo b_j = 4i phi_j,n / s
        b = 4j * phi[n] / s

        for i in range(N-1,0,-1):
            beta[i] = (b[i+1] - beta[i+1]) / (-2+2j/s - potencial[i+1] + alpha[i+1])

        for i in range(N):
            chi[i+1] = alpha[i]*chi[i] + beta[i]

        phi[n+1] = chi - phi[n]

    return phi



fout = "schrodinger_data.dat"
graph_out = "norma_altura_potencial_0-3"

N = 200
nciclos = N/16
altura_potencial = 0.3
NSTEPS = 1000


k0 = 2*m.pi*nciclos/N
POTENCIAL = np.array([altura_potencial if (i>2*N/5 and i<3*N/5) else 0 for i in range(N+1)])


phi = np.abs(schrodinger(N, nciclos, altura_potencial, NSTEPS))**2
#phi2 = np.abs(schrodinger(N, nciclos*2, altura_potencial, NSTEPS))**2 
#phi3 = np.abs(schrodinger(N, nciclos*4, altura_potencial, NSTEPS))**2 
x = np.arange(0,N+1)/N

norma_phi = np.meshgrid(x,np.sum(phi, axis=1))[1]
norma_phi /= norma_phi[0,0]



f = open(fout, "w")
for i in range(NSTEPS+1):
    vector_to_save = np.array(list(zip(x, POTENCIAL, phi[i], norma_phi[i])))
    np.savetxt(f,vector_to_save,delimiter=', ')
    f.write('\n')
f.close()


t = np.arange(0, NSTEPS+1)





