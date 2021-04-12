import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import matplotlib.pyplot as plt
from tqdm import tqdm


# Physical constants for the project

D_0 = 1         # Diffusion Constant
V_a = 1         # ---
R = 1           # Ideal gas constant
v_0 = 1         # Convection parameter
ro = 1          # Density of water
g = 9.81        # gravitational acceleration
p_ref = 1       # Reference pressure at ocean surface
h = 100         # Ocean depth

# Computational constants for the project

dt = 0.1        # Time step
dx = 0.1        # Spatial step


# Functions

# Spatially varying functions for the simulation

# The gradient of the local diffusion coefficient
def grad_D(p, T, dTz, V_a, D_0, R):
    return D_0 * (V_a / (2.303 * R * T)) * ((p / T) * dTz + ro * g) * np.exp(- (V_a * p)/(2.303 * R * T))

# The local diffusion coefficient as a function of pressure and temperature
def D(p, T, V_a, D_0, R):
    return D_0 * np.exp(-(V_a * p)/(2.303 * R * T))

# Pressure as a function of depth
def p_afo_z(z, p_ref):
    return p_ref + ro * g * (h - z)

# Temperature as a function of depth. Here assumed linear dependence with z
def T_afo_z(z, T_ref, A):
    return T_ref + A * (h - z)


# Functions related to the numerical computations

# These functions are for the 1D case. Will generalize to 2D and 3D later.
def initialize_L1D(alpha, Gamma, K, Km):
    du = - alpha / 4 * Km[:-1] - alpha * K[:-1]
    dl = alpha / 4 * Km[1:] - alpha * K[1:]
    d = 1 + 2 * alpha * K

    du[0] = -2 * alpha * K[0]
    dl[-1] = -2 * alpha * K[-1]
    d[0] += Gamma

    return sps.diags([du, d, dl], [1, 0, -1])

def initialize_R1D(alpha, Gamma, K, Km):
    du = alpha / 4 * Km[:-1] + alpha * K[:-1]
    dl = - alpha / 4 * Km[1:] + alpha * K[1:]
    d = 1 - 2 * alpha * K

    du[0] = 2 * alpha * K[0]
    dl[-1] = 2 * alpha * K[-1]
    d[0] -= Gamma

    return sps.diags([du, d, dl], [1, 0, -1])


def time_iteration1D(L, R, S, Ci, i):
    V = np.matmul(R.toarray(), Ci) + (1/2) * (S.toarray()[:,i] + S.toarray()[:,i+1])
    return spsl.spsolve(L.tocsr(), V)


def solver1D(kw, K_func, C0_func, S_func, dz, dt, depth, totalTime):
    # Make the environment to simulate
    Nz = int(depth//dz)
    Nt = int(totalTime//dt)
    z = np.linspace(0, depth, Nz)
    t = np.linspace(0, totalTime, Nt)

    # Turn the initial distribution and K into arrays
    K = K_func(z)
    S = sps.lil_matrix((Nz, Nt))
    S[0,:] = S_func(t)
    C = np.zeros((Nz, Nt))
    C[:,0] = C0_func(z)
    Km = np.roll(K, -1) - np.roll(K, 1)

    # Compute some important coefficients for the simulation
    alpha = dt / (2 * dz**2)
    Gamma = 2 * alpha * kw * dz * (1 - (-(3/2) * K[0] + 2*K[1] - (1/2)*K[2])/(2*K[0]))

    # The matrices for the time-iterations
    L = initialize_L1D(alpha, Gamma, K, Km)
    R = initialize_R1D(alpha, Gamma, K, Km)
    S = 2 * Gamma * S

    # The actual time-iteration
    for i in range(Nt-1):
        C[:,i+1] = time_iteration1D(L, R, S, C[:,i], i)

    return C, z, t, K

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
            dK = D[j] * dt / (8 * dz**2)
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


def initialize_St(S_func, t, N, Nt, Nz, Nx):
    S = sps.lil_matrix((N, Nt))
    for i in range(Nt):
        S[0::Nz,i] = S_func(t[i]) * np.ones(Nx)
    return S


def time_iteration2D(L, R, Ci, S, i):
    V = np.matmul(R.toarray(), Ci) + (1/2) * (S.toarray()[:,i] + S.toarray()[:,i+1])
    return spsl.spsolve(L.tocsr(), V)


def solver2D(kw, K_func, C0_func, S_func, dx, dt, depth, width, totalTime, vx, vz):
    Nz = int(depth // dx)
    Nx = int(width // dx)
    Nt = int(totalTime // dt)
    N = Nx * Nz

    x = np.linspace(0,width, Nx)
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
        C[:,i+1] = time_iteration2D(L, R, C[:,i], S, i)

    return C, z, x, t, K

def convert_1D_to_2D(C, Nx, Nz):
    newC = np.zeros((Nx, Nz))
    for i in range(Nx):
        newC[i,:] = C[(i)*Nz:(i+1)*Nz]
    return np.transpose(newC)

# This is just a test run to check that the solver for the 1D case is working as intended.
kw = 0
depth = 100
dz = 0.1
dt = 0.1
totalTime = 100

def K1(z):
    return np.ones(np.shape(z))

def C01(z):
    return np.exp(-(z-depth/2)**2 / 2)

def S1(t):
    return np.ones(np.shape(t))

# This is a test run of the 2D code to see if it is working as intended.

kw = 0
depth = 10
width = 10
dx = 0.1
dt = 0.1
totalTime = 10
Nx = int(width//dx)
Nz = int(depth//dx)
vx = 5 * np.ones((Nx, Nz))
vz = np.zeros((Nx, Nz))

def C02D(x,z):
    C0 = np.zeros(np.shape(x)[0] * np.shape(z)[0])
    Nz = np.shape(z)[0]
    for i in range(np.shape(x)[0]):
        C0[i * Nz:(i+1)*Nz] = np.exp(-(1/2) * ((x[i]-width/2)**2 + (z-depth/2)**2))
    return C0


C, z, x, t, K = solver2D(kw, K1, C02D, S1, dx, dt, depth, width, totalTime, vx, vz)
