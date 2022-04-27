import numpy as np
import math as m
from numba import njit


def schrodinger(N, nciclos, long_onda, steps):

    k0 = 2*m.pi*nciclos/N
    s = 1/(4*k0**2)

    phi = np.zeros((N+1,steps), dtype=complex) 
    b = np.zeros(N+1, dtype=complex)
    alpha = np.zeros(N, dtype=complex)
    beta = np.zeros(N, dtype=complex)

    potencial = np.array([long_onda*k0**2 if (j>2*N/5 and j<3*N/5) else 0 for j in range(N+1)])

    phi[:,0] = np.exp(1j*k0*np.arange(N+1)) * np.exp(-8 * (4*np.arange(N+1)-N)**2 / N**2)
    phi[0,0] = 0.
    phi[N,0] = 0.

    for j in reversed(range(N-1)):
        alpha[j] = -1/(-2+2j/s + alpha[j+1])

    
    for n in range(steps):
        pass


    

schrodinger(10, 10, 10, 100)

