from re import X
import numpy as np
import math as m
from numba import njit


def schrodinger(N, nciclos, altura, steps):

    k0 = 2*m.pi*nciclos/N
    s = 1/(4*k0**2)

    phi = np.zeros((N+1,steps+1), dtype=complex) 
    b = np.zeros(N+1, dtype=complex)
    alpha = np.zeros(N, dtype=complex)
    beta = np.zeros(N, dtype=complex)
    chi = np.zeros(N+1, dtype=complex)

    potencial = np.array([altura*k0**2 if (i>2*N/5 and i<3*N/5) else 0 for i in range(N+1)])

    phi[:,0] = np.exp(1j*k0*np.arange(N+1)) * np.exp(-8 * (4*np.arange(N+1)-N)**2 / N**2)
    phi[0,0] = 0.
    phi[N,0] = 0.

    for i in reversed(range(N-1)):
        alpha[i] = -1/(-2+2j/s + alpha[i+1])

    
    for n in range(steps):
        
        # Calculo b_j = 4i phi_j,n / s
        b = 4j * phi[:,n] / s

        for i in reversed(range(N-1)):
            beta[i] = (b[i+1] - beta[i+1]) / (-2+2j/s + alpha[i+1])

        for i in range(N):
            chi[i+1] = alpha[i]*chi[i] + beta[i]

        phi[:,n+1] = chi - phi[:,n]

    return phi


fout = "schrodinger_data.dat"

N = 100
nciclos = N/4
altura = 3
NSTEPS = 100


phi = np.real(schrodinger(N, nciclos, altura, NSTEPS))
x = np.arange(0,N+1)/N


f = open(fout, "w")
for i in range(len(phi)):
    vector_to_save = np.column_stack((x, phi[i]))
    np.savetxt(f,vector_to_save,delimiter=', ')
    f.write('\n')
f.close()

