import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# Physical constants for the project

D_0 = 1         # Diffusion Constant
V_a = 1         # ---
R = 1           # Ideal gas constant
v_0 = 1         # Convection parameter
ro = 1          # Density of water
g = 9.81        # gravitational akseleration
p_ref = 1       # Refrence pressure at ocean surface
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
def initialize_L(alpha, Gamma, K, Km):
    du = - alpha / 4 * Km[:-1] - alpha * K[:-1]
    dl = alpha / 4 * Km[1:] - alpha * K[1:]
    d = 1 + 2 * alpha * K

    du[0] = -2 * alpha * K[0]
    dl[-1] = -2 * alpha * K[-1]
    d[0] += Gamma

    return sp.sparse.diags([du, d, dl], [1, 0, -1])

def initialize_R(alpha, Gamma, K, Km):
    du = alpha / 4 * Km[:-1] + alpha * K[:-1]
    dl = - alpha / 4 * Km[1:] + alpha * K[1:]
    d = 1 - 2 * alpha * K

    du[0] = 2 * alpha * K[0]
    dl[-1] = 2 * alpha * K[-1]
    d[0] -= Gamma

    return sp.sparse.diags([du, d, dl], [1, 0, -1])

def time_iteration(L, R, S, Ci):
    V = np.matmul(R, Ci) + (1/2) * (S[:,i] + S[:,i+1])

