import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sci
import scipy.linalg as sci

def cheb(N):
    Ny = N
    num = Ny-1

    vec = np.arange(0,Ny)

    yF = np.cos(np.pi*vec/(Ny-1))
    D0 = np.array(np.cos(0*np.pi*vec/num))

    for j in np.arange(1,num+1):
        D0 = np.column_stack((D0,np.cos(j*np.pi*vec/num)))

    # Create the higher order derivatives
    D1 = np.column_stack((np.zeros(Ny), D0[:,0], 4*D0[:,1]))
    D2 = np.column_stack((np.zeros(Ny), np.zeros(Ny), 4*D0[:,0]))

    for j in np.arange(3,num+1):
        D1 = np.column_stack((D1, 2*j*D0[:,j-1]+j*D1[:,j-2]/(j-2)))
        D2 = np.column_stack((D2, 2*j*D1[:,j-1]+j*D2[:,j-2]/(j-2)))

    D0_inv = np.linalg.inv(D0)
    
    D1 = np.dot(D1,D0_inv)
    D2 = np.dot(D2,D0_inv)

    return yF,D1,D2

def compute_energy_gain( tau, Re = 2000,alpha = 1.0,beta = 0.25):

    # Discretising using Chebyshev collocation points
    N = 100

    y,D1,D2 = cheb(N)


    U = 1 - y **2

    dU = np.dot(D1,U)

    # In matrix form
    dU = np.diag(dU)
    U = np.diag(U)

    I = np.eye(N)
    zero = np.zeros((N,N))



    M = -1j * alpha * U  + 1/Re * ( - alpha**2 * I  - beta**2 * I + D2)

    # The Linearised Navier Stokes Operator matrix
    L = np.block([
                [M, -dU,zero, -1j * alpha * I],
                [zero, M , zero, -D1],
                [zero, zero, M, -1j * beta * I],
                [1j * alpha * I, D1, 1j * beta * I,zero]
                ]
                )

    # Zero out rows of L
    # u
    L[0,:] = 0
    L[N-1,:] = 0

    # v
    L[N,:] = 0
    L[2 * N - 1, :] = 0

    # w
    L[2 * N, :] = 0
    L[3 * N -1, :] = 0

    # Diagonal entries at walls are 1 
    #  u
    L[0 , 0] = 1
    L[N-1, N-1] = 1 

    #   v
    L[N,N] = 1 
    L[2 * N -1, 2 * N -1] = 1 

    #   w
    L[2 * N, 2 * N] = 1 
    L[3 * N - 1, 3 * N -1] = 1 

    L = tau * L
    e = sci.expm(L)
    
    '''
    W = e

    F = np.linalg.cholesky(W)

    E = F @ e @ np.linalg.i(F)
    '''

    U,sigma,V = np.linalg.svd(e)

    G = (sigma[0])**2

 
    return G

N = 100
G = np.zeros(N)

t = np.linspace(0,50,N)

for i in range(N):
    G[i] = compute_energy_gain(t[i])

plt.plot(t,G)
plt.show()



