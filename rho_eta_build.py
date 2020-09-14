# Building rho(eta)

import numpy as np
import math
import sympy as sym

def make_fock_basis_vec(n, rho_size):
    basis_vec = np.zeros(rho_size, dtype = float)
    basis_vec[n] = 1
    
    return basis_vec

def make_fock_basis_array(rho_size):
    basis_array = []
    
    for i in range(rho_size):
        basis_array.append(make_fock_basis_vec(i, rho_size))
    
    return basis_array

def make_rho_coh(rho_size, n_bar):
    rho_coh = np.zeros([rho_size, rho_size], dtype = float)
    fock_basis_array = make_fock_basis_array(rho_size)
    
    for i in range(rho_size):
        for j in range(rho_size):
            rho_coh += (np.e**(-n_bar) * np.sqrt(n_bar**i / math.factorial(i)) * np.sqrt(
                n_bar**j / math.factorial(j))) * np.outer(fock_basis_array[i], fock_basis_array[j])

    return rho_coh

def make_rho_th(rho_size, n_bar):
    rho_th = np.zeros([rho_size, rho_size], dtype = float)
    fock_basis_array = make_fock_basis_array(rho_size)
    
    for i in range(rho_size):
        for j in range(rho_size):
            if i == j:
                rho_th += (n_bar**i / (n_bar + 1)**(i + 1)) * np.outer(fock_basis_array[i], fock_basis_array[i])
    
    return rho_th

def make_rho_eta(eta, rho_size, n_bar):
    rho_coh = make_rho_coh(rho_size, n_bar)
    rho_th = make_rho_th(rho_size, n_bar)
    rho_eta = (eta * rho_coh) + ((1 - eta) * rho_th)

    return rho_eta

def make_rho_eta_diff(eta, rho_size, n_bar): # It's a constant matrix, so eta doesn't matter
    rho_eta = make_rho_eta(eta, rho_size, n_bar)

    rho_eta_diff = np.zeros((rho_size, rho_size), dtype = float)
    eta_var = sym.symbols('x')
    
    for i in range(rho_size):
        for j in range(rho_size):
            rho_eta_diff[i, j] = sym.diff(make_rho_eta(eta_var, rho_size, n_bar)[i, j], eta_var)

    return rho_eta_diff