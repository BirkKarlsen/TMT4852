import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import scipy.integrate as spi
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit



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

# Some constants
p02 = 20.95     # [atm] https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
C0O2 = 0.325    # [mol/m^3]
D0 = 2.32e-9    # [m^2/s]

# Functions

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

def initialize_RL1D_with_BC(alpha, Gammat, Gammab, K, Km):
    duL = - alpha / 4 * Km[:-1] - alpha * K[:-1]
    dlL = alpha / 4 * Km[1:] - alpha * K[1:]
    dL = 1 + 2 * alpha * K

    duL[0] = -2 * alpha * K[0]
    dlL[-1] = -2 * alpha * K[-1]
    dL[0] += Gammat
    dL[-1] += Gammab

    duR = alpha / 4 * Km[:-1] + alpha * K[:-1]
    dlR = - alpha / 4 * Km[1:] + alpha * K[1:]
    dR = 1 - 2 * alpha * K

    duR[0] = 2 * alpha * K[0]
    dlR[-1] = 2 * alpha * K[-1]
    dR[0] -= Gammat
    dR[-1] -= Gammab

    return sps.diags([duL, dL, dlL], [1, 0, -1]), sps.diags([duR, dR, dlR], [1, 0, -1])


@jit(nopython=True)
def TDMA(a, b, c, d):
    N = np.shape(d)[0]

    cm = np.zeros(N - 1)
    dm = np.zeros(N)
    x = np.zeros(N)

    cm[0] = c[0] / b[0]
    dm[0] = d[0] / b[0]
    for i in range(1, N):
        if i != N - 1:
            cm[i] = c[i] / (b[i] - a[i - 1] * cm[i - 1])

        dm[i] = (d[i] - a[i - 1] * dm[i - 1]) / (b[i] - a[i - 1] * cm[i - 1])

    x[-1] = dm[-1]
    for i in range(N - 2, -1, -1):
        x[i] = dm[i] - cm[i] * x[i + 1]

    return x

def solver1D_TDMA(kw, K_func, C0_func, S_func, dz, dt, depth, totalTime):
    # Make the environment to simulate
    Nz = int(depth//dz) + 2
    Nt = int(totalTime//dt) + 2
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
    for i in tqdm(range(1, Nt)):
        V = np.matmul(R.toarray(), C[:,i-1]) + (1 / 2) * (S.toarray()[:, i-1] + S.toarray()[:, i])
        C[:,i] = TDMA(L.diagonal(-1), L.diagonal(0), L.diagonal(1), V)

    return C, z, t, K, S[0,:]

def solver1D_TDMA_w_BC(kwt, kwb, K_func, C0_func, S_func, dz, dt, depth, totalTime):
    # Make the environment to simulate
    Nz = int(depth // dz) + 2
    Nt = int(totalTime // dt) + 2
    z = np.linspace(0, depth, Nz)
    t = np.linspace(0, totalTime, Nt)

    # Turn the initial distribution and K into arrays
    K = K_func(z)
    S = sps.lil_matrix((Nz, Nt))
    S[0, :] = S_func(t)
    C = np.zeros((Nz, Nt))
    C[:, 0] = C0_func(z)
    Km = np.roll(K, -1) - np.roll(K, 1)

    # Compute some important coefficients for the simulation
    alpha = dt / (2 * dz ** 2)
    Gammat = 2 * alpha * kwt * dz * (1 - (-(3 / 2) * K[0] + 2 * K[1] - (1 / 2) * K[2]) / (2 * K[0]))
    Gammab = 2 * alpha * kwb * dz * (1 - (-(3/2) * K[-1] + 2 * K[-2] - (1/2) * K[-3]) / (2 * K[-1]))

    # The matrices for the time-iterations
    L, R = initialize_RL1D_with_BC(alpha, Gammat, Gammab, K, Km)
    S = 2 * Gammat * S

    # The actual time-iteration
    for i in tqdm(range(1, Nt)):
        V = np.matmul(R.toarray(), C[:, i - 1]) + (1 / 2) * (S.toarray()[:, i - 1] + S.toarray()[:, i])
        C[:, i] = TDMA(L.diagonal(-1), L.diagonal(0), L.diagonal(1), V)

    return C, z, t, K, S[0, :]


# These are functions to plot various things

def plot_situation(C, z, t, K):
    fig, axs = plt.subplots(2, 2, figsize=(8,8))

    axs[0,0].plot(C[:,0], z)
    axs[0,0].set(xlim=(0,1))
    axs[0,0].set_title("First time step")
    axs[0,0].invert_yaxis()

    axs[0,1].plot(C[:,-1], z)
    axs[0,1].set(xlim=(0,1))
    axs[0,1].set_title("Last time step")
    axs[0,1].invert_yaxis()

    axs[1,0].plot(K, z)
    axs[1,0].set_title("Diffusion coefficient")
    axs[1,0].invert_yaxis()

    M = spi.simps(C, z, axis=0)
    axs[1,1].plot(t, 100 * (M[0] - M)/M[0])
    axs[1,1].set_title("Masses")

    plt.show()

def plot_variance_and_expval(C, z, t):
    s0z = 1 ** 2
    mu0 = depth / 2 * np.ones(np.shape(t))
    Kk = 1

    Cz = np.zeros(np.shape(C))
    for i in range(np.shape(C)[1]):
        Cz[:, i] = C[:, i] * z

    mu = spi.simps(Cz, z, axis=0) / spi.simps(C, z, axis=0)

    Czmu = np.zeros(np.shape(C))
    for i in range(np.shape(Czmu)[1]):
        Czmu[:, i] = C[:, i] * (z - mu[i]) ** 2

    sigma = spi.simps(Czmu, z, axis=0) / spi.simps(C, z, axis=0)

    plt.plot(t, mu)
    plt.plot(t, mu0)

    plt.show()

    plt.plot(t, sigma)
    plt.plot(t, s0z + 2 * Kk * t)

    plt.show()


# Some temporary test variables
kw = 10
depth = 100
dz = 0.01
dt = 0.1
totalTime = 100

# functions for K(z), C_0(z) and S(t)

def K1(z):
    return 50 * np.ones(np.shape(z))

def K2(z):
    return 2 * z + 1

def K3(z):
    return 60 * np.exp(- z / depth)

def K4(z):
    K0 = 1e-3
    Ka = 2e-2
    za = 7
    Kb = 5e-2
    zb = 10
    return K0 + Ka * (z/za) * np.exp(-z/za) + Kb * ((depth - z)/zb) * np.exp(-(depth - z)/zb)

def K5(z):
    K0 = 1e-4
    K1 = 1e-2
    a = 0.5
    z0 = 100
    return K1 + (K0 - K1)/(1 + np.exp(-a*(z - z0)))

def C01(z):
    return np.exp(-(z-depth/2)**2 / (2 * (1)**2))

def C02(z):
    return 0 * np.ones(np.shape(z))

def S1(t):
    return 0.1 * np.ones(np.shape(t))

def S2(t):
    return np.zeros(np.shape(t))

# This is just a test run to check that the solver for the 1D case is working as intended.

C, z, t, K, St = solver1D_TDMA_w_BC(kw, kw, K3, C02, S1, dz, dt, depth, totalTime)

plt.plot(z, C[:,-1])
plt.plot(z, 0.1*np.ones(np.shape(z)))
plt.ylim(0,0.2)
plt.show()

