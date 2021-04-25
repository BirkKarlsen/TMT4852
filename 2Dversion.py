import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import scipy.integrate as spi
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit


# Now for the 2D case. We will now add convection into the mix.
# We assume a square lattice of points in the x and z directions and get.

def initialize_L_and_R2D(Nx, Nz, vx, vz, dt, dx, D, K):
    N = Nx * Nz
    L = sps.lil_matrix((N,N))
    R = sps.lil_matrix((N,N))

    for i in range(Nx):
        for j in range(Nz):
            # The gamma-coefficients for the matrices
            Km = K[j] * dt / (2 * dx**2)
            dK = D[j] * dt / (8 * dx**2)
            vxm = vx[i,j] * dt / (4 * dx)
            vzm = vz[i,j] * dt / (4 * dx)
            dvx = (vx[(i+1)%Nx, j] - vx[(i-1)%Nx, j]) * dt / (4 * dx)
            dvz = (vz[i, (j+1)%Nz] - vz[i, (j-1)%Nx]) * dt / (4 * dx)

            n = j + Nz * i

            # Main diagonal
            L[n,n] = 1 + 4*Km + dvx + dvz
            R[n,n] = 1 - 4*Km - dvx - dvz

            # The superdiagonal
            L[n,(n+1)%Nz + Nz * i] = - dK - Km + vzm
            R[n,(n+1)%Nz + Nz * i] = dK + Km - vzm

            # The subdiagonal
            L[n,(n-1)%Nz + Nz * i] = - dK - Km - vzm
            R[n,(n-1)%Nz + Nz * i] = dK + Km + vzm

            # The supersuperdiagonal
            L[n,(n+Nz)%N] = - Km + vxm
            R[n,(n+Nz)%N] = Km - vxm

            # The subsubdiagonal
            L[n,(n-Nz)%N] = - Km - vxm
            R[n,(n-Nz)%N] = Km + vxm

    # Last part here is for boundary conditions, which I will add in later.

    return L.todia(), R.todia()


def initialize_L_and_R2D_V2(Nx, Nz, vx, vz, dt, dx, D, K, Ga):
    N = Nx * Nz
    L = sps.lil_matrix((N,N))
    R = sps.lil_matrix((N,N))

    for i in range(Nx):
        for j in range(Nz):
            # The gamma-coefficients for the matrices
            Km = K[j] * dt / (2 * dx**2)
            dK = D[j] * dt / (8 * dx**2)
            vxm = vx[i,j] * dt / (4 * dx)
            vzm = vz[i,j] * dt / (4 * dx)
            dvx = (vx[(i+1)%Nx, j] - vx[(i-1)%Nx, j]) * dt / (4 * dx)
            dvz = (vz[i, (j+1)%Nz] - vz[i, (j-1)%Nx]) * dt / (4 * dx)

            n = j + Nz * i

            # Main diagonal
            L[n,n] = 1 + 4*Km + dvx + dvz
            R[n,n] = 1 - 4*Km - dvx - dvz
            if j == 0:
                L[n,n] += Ga
                R[n,n] -= Ga

            # The superdiagonal
            L[n,(n+1)%Nz + Nz * i] = - dK - Km + vzm
            R[n,(n+1)%Nz + Nz * i] = dK + Km - vzm

            # The subdiagonal
            L[n,(n-1)%Nz + Nz * i] = - dK - Km - vzm
            R[n,(n-1)%Nz + Nz * i] = dK + Km + vzm

            # The supersuperdiagonal
            L[n,(n+Nz)%N] = - Km + vxm
            R[n,(n+Nz)%N] = Km - vxm

            # The subsubdiagonal
            L[n,(n-Nz)%N] = - Km - vxm
            R[n,(n-Nz)%N] = Km + vxm

    # Last part here is for boundary conditions, which I will add in later.

    return L.todia(), R.todia()

def initialize_St(S_func, t, N, Nt, Nz, Nx):
    S = sps.lil_matrix((N, Nt))
    for i in range(Nt):
        S[0::Nz,i] = S_func(t[i]) * np.ones(Nx)
    return S


def time_iteration2D(L, R, Ci, S, i):
    V = np.matmul(R.toarray(), Ci) + (1/2) * (S.toarray()[:,i] + S.toarray()[:,i+1])
    return spsl.bicgstab(L.tocsr(), V, x0=Ci, tol=1e-9 )


def solver2D(kw, K_func, C0_func, S_func, dx, dt, depth, width, totalTime, vx, vz):
    Nz = int(depth // dx)
    Nx = int(width // dx)
    Nt = int(totalTime // dt)
    N = Nx * Nz

    x = np.linspace(0, width, Nx)
    z = np.linspace(0, depth, Nz)
    t = np.linspace(0, totalTime, Nt)

    K = K_func(z)
    S = initialize_St(S_func, t, N, Nt, Nz, Nx)
    C = np.zeros((N, Nt))
    C[:,0] = C0_func(x,z)
    D = np.roll(K, -1) - np.roll(K, 1)

    alpha = dt / (2 * dx**2)
    Gamma = 2 * alpha * kw * dx * (1 - (-(3/2) * K[0] + 2*K[1] - (1/2)*K[2])/(2*K[0]))

    print("Initializing L and R...")
    L, R = initialize_L_and_R2D(Nx, Nz, vx, vz, dt, dx, D, K)
    S = 2 * Gamma * S

    print("Simulating...")
    for i in tqdm(range(Nt-1)):
        C[:,i+1], status = time_iteration2D(L, R, C[:,i], S, i)

        if status != 0:
            print("Iteration failed with ", status)

    return C, z, x, t, K

def convert_1D_to_2D(C, Nx, Nz):
    newC = np.zeros((Nx, Nz))
    for i in range(Nx):
        newC[i,:] = C[(i)*Nz:(i+1)*Nz]
    return np.transpose(newC)

# This is a test run of the 2D code to see if it is working as intended.

kw = 0
depth = 1
width = 1
dx = 0.01
dt = 0.01
totalTime = 1
Nx = int(width//dx)
Nz = int(depth//dx)
vx = 3 * np.ones((Nx, Nz))
vz = np.zeros((Nx, Nz))

def C02D(x,z):
    C0 = np.zeros(np.shape(x)[0] * np.shape(z)[0])
    Nz = np.shape(z)[0]
    for i in range(np.shape(x)[0]):
        C0[i * Nz:(i+1)*Nz] = np.exp(-(1/2) * ((x[i]-width/4)**2 + (z-depth/4)**2)/0.000001)
    return C0

def doublegyre(x, y, t, A=0.1, e=0.01, w=0.01):
    a = e * np.sin(w*t)
    b = 1 - 2*e*np.sin(w*t)
    f = a*x**2 + b*x
    return np.array([
            -np.pi*A*np.sin(np.pi*f) * np.cos(np.pi*y),              # x component of velocity
             np.pi*A*np.cos(np.pi*f) * np.sin(np.pi*y) * (2*a*x + b) # y component of velocity
        ])



# Plot for illustration
# x = np.linspace(0, width, Nx)
# z = np.linspace(0, depth, Nz)
#
# X, Z = np.meshgrid(x, z)
#
# vx, vz = doublegyre(X, Z, 0, 1, 0.01, 0.01)
#
# fig = plt.figure(figsize = (9,5))
# plt.quiver(X, Z, vx, vz)
# plt.show()
#
# C, z, x, t, K = solver2D(kw, K1, C02D, S1, dx, dt, depth, width, totalTime, vx, vz)
#
# plt.imshow(convert_1D_to_2D(C[:,0], Nx, Nz))
# plt.colorbar()
# plt.show()
#
# plt.imshow(convert_1D_to_2D(C[:,50], Nx, Nz))
# plt.colorbar()
# plt.show()
