import numpy as np

from scipy.linalg import eig

import matplotlib.pyplot as plt

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

def run_stability_analysis(Re, alpha,beta = 0):

    N = 200

    # Discretising using Chebyshev collocation points
    y,D1,D2 = cheb(N)


    # Mean-Flow Profile
    U = 1 - y**2

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

    # lhs Boundary conditions

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
    L[0 , 0] = 1 - 200j
    L[N-1, N-1] =  - 200j

    #   v
    L[N,N] = 1 - 200j
    L[2 * N -1, 2 * N -1] = 1 - 200j

    #   w
    L[2 * N, 2 * N] = 1 - 200j
    L[3 * N - 1, 3 * N -1] = 1 - 200j



    Ib = I

    Ib[0,0] = 0
    Ib[N-1,N-1] = 0

    # rhs of the eigenproblem

    B = np.block([
                [Ib, zero, zero, zero],
                [zero, Ib, zero, zero],
                [zero, zero, Ib, zero],
                [zero, zero, zero, zero],


            ])
    
    B = -1j * B


    # rhs boundary conditions
    # u
    B[0,:] = 0
    B[N-1,:] = 0

    # v
    B[N,:] = 0
    B[2 * N - 1, :] = 0

    # w
    B[2 * N, :] = 0
    B[3 * N -1, :] = 0


    eigvals, eigvecs = eig(L,B)

 

    return eigvals




eigvals = run_stability_analysis(5772.22, 1.02)


# Plotting
plt.figure(figsize=(6,6))
plt.axhline(0, color='black', linewidth=0.8)  
plt.axvline(0, color='black', linewidth=0.8)  

plt.scatter(eigvals.real, eigvals.imag, color='red', s=4)  
plt.xlabel('Real part')
plt.ylabel('Imaginary part')
plt.grid(True)
plt.axis([-0.1, 1.5, -2.0, 0.1]) 
plt.show()