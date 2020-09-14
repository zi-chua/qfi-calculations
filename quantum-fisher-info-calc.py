# Calculating the quantum fisher info for my POVMs

from fisher_info import * # this method of importing doesn't work anymore after I updated VS Code

import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
import math


# QFI

def compute_qfi(eta, rho_size, n_bar):
    
    rho_eta = make_rho_eta(eta, rho_size, n_bar)
    rho_eigenvals, rho_eigenvecs = np.linalg.eigh(rho_eta)
    rho_eigenvecs = np.transpose(rho_eigenvecs)
    
    rho_eta_diff = np.zeros((rho_size, rho_size), dtype = float)
    eta_var = sym.symbols('x')
    
    for i in range(rho_size):
        for j in range(rho_size):
            rho_eta_diff[i, j] = sym.diff(make_rho_eta(eta_var, rho_size, n_bar)[i, j], eta_var)
    
    qfi = 0
    for i in range(rho_size):
        for j in range(rho_size):
            qfi += 2 * abs(np.conj(rho_eigenvecs[i]) @ rho_eta_diff @ rho_eigenvecs[j])**2 / (rho_eigenvals[i] + rho_eigenvals[j])

    return qfi

compute_qfi(0.5, 3, 0.02)

qfi_range_eta_array = []
eta_vals = np.arange(0, 1, 0.01)
for eta in eta_vals:
    np.append(qfi_range_eta_array, compute_qfi(eta, 3, 0.02))
qfi_range_eta_array = np.asarray(qfi_range_eta_array)
plt.plot(eta_vals, qfi_range_eta_array)
len(ratio_vals)
ratio_vals = eta_vals / (1 - eta_vals)
plt.plot(ratio_vals, qfi_range_eta_array)

get_fisher_info_eta(0.5, make_sld_povm_array(0.5, 3, 0.02), 3, 0.02)


def make_qfi_range_eta_array(eta_start, eta_stop, rho_size, n_bar):
    eta_vals = np.arange(eta_start, eta_stop+1/100, (eta_stop - eta_start)/100)
    qfi_range_eta_array = []
    
    for eta in eta_vals:
        qfi_range_eta_array.append(compute_qfi(eta, rho_size, n_bar))

    return eta_vals, qfi_range_eta_array    

eta_start, eta_stop, rho_size, n_bar = 0, 1, 3, 0.02
eta_vals, qfi_range_eta_array = make_qfi_range_eta_array(eta_start, eta_stop, rho_size, n_bar)

fig, ax = plt.subplots(1, 2, figsize = (12, 5))

ax[0].plot(eta_vals[:-1], qfi_range_eta_array[:-1])
ax[0].set_title('QFI vs Eta')
ax[0].set_xlabel('Eta')
ax[0].set_ylabel('QFI')
ax[0].text(min(eta_vals), 0.75*max(qfi_range_eta_array[:-1]), f"rho_size = {rho_size} \nn_bar = {n_bar} \nI'm omitting eta = 1 since \nfor some reason the QFI \nblows up at eta = 1.")

log_qfi_range_eta_array = [math.log10(qfi) for qfi in qfi_range_eta_array[:-1]]
ax[1].plot(eta_vals[:-1], log_qfi_range_eta_array)
ax[1].set_title('log(QFI) vs Eta')
ax[1].set_label('Eta')
ax[1].set_ylabel('log(QFI)')
ax[1].text(min(eta_vals), 0.75*max(log_qfi_range_eta_array), f"rho_size = {rho_size} \nn_bar = {n_bar} \nI'm omitting eta = 1 again.")

plt.savefig('qfi-vs-eta.jpg')

plt.plot(ratio_array[:-1], qfi_range_eta_array[:-1])
plt.title('QFI vs Coh/Therm ratio')
plt.xlabel('Coh/Therm ratio')
plt.ylabel('QFI')
plt.text(min(ratio_array), 0.9*max(qfi_range_eta_array[:-1]), f'rho_size = {rho_size} \nn_bar = {n_bar}')