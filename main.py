import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import scipy.linalg as spl
import matplotlib.pyplot as plt
import time

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

    # The actual time-iteration
    for i in range(Nt-1):
        C[:,i+1] = time_iteration1D(L, R, S, C[:,i], i)

    return C, z, t, K

kw = 0
depth = 100
dz = 0.01
dt = 0.1
totalTime = 100

def K1(z):
    return np.ones(np.shape(z))

def C01(z):
    return np.exp(-(z-depth/2)**2 / 2)

def S1(t):
    return np.ones(np.shape(t))

C, z, t, K = solver1D(kw, K1, K1, S1, dz, dt, depth, totalTime)

plt.plot(z, C[:,0])
plt.plot(z,C[:,1])
plt.show()