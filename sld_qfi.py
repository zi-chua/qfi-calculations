# SLD --> QFI

import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import sympy as sym
import pandas as pd
sns.set()
from rho_eta_build import *
from hadamard_fi import *
from homodyne_fi import *


def compute_sld_matrix(eta, rho_size, n_bar):
    
    rho_eigenvals, rho_eigenvecs = np.linalg.eigh(make_rho_eta(eta, rho_size, n_bar))
    rho_eigenvecs = np.transpose(rho_eigenvecs)
    rho_eta_diff = make_rho_eta_diff(eta, rho_size, n_bar)

    sld_matrix = np.zeros((rho_size, rho_size), dtype = float)
    
    for i in range(rho_size):
        for j in range(rho_size):
            if (rho_eigenvals[i] + rho_eigenvals[j]) != 0:
                sld_matrix += ((2 * np.conj(rho_eigenvecs[i]) @ rho_eta_diff @ rho_eigenvecs[j]) / (rho_eigenvals[i] + rho_eigenvals[j])) * np.outer(rho_eigenvecs[i], rho_eigenvecs[j])

    return sld_matrix

def make_sld_povm_array(eta, rho_size, n_bar):
    """The povms are projectors of the SLD eigenvectors"""
    
    sld_eigvecs = np.transpose(np.linalg.eigh(compute_sld_matrix(eta, rho_size, n_bar))[1])

    sld_povm_array = []
    for eigvec in sld_eigvecs:
        sld_povm_array.append(np.outer(eigvec, eigvec))

    return sld_povm_array

def compute_sld_fisher_info(eta, rho_size, n_bar):
    rho_eta = make_rho_eta(eta, rho_size, n_bar)
    rho_eta_diff = make_rho_eta_diff(0, rho_size, n_bar)
    sld_povm_array = make_sld_povm_array(eta, rho_size, n_bar)
    
    fisher_info = 0

    for povm in sld_povm_array:
        fisher_info += (np.trace(povm @ rho_eta_diff))**2 / np.trace(povm @ rho_eta)

    return fisher_info

def compute_qfi(eta, rho_size, n_bar):
    
    rho_eta = make_rho_eta(eta, rho_size, n_bar)
    rho_eigenvals, rho_eigenvecs = np.linalg.eigh(rho_eta)
    rho_eigenvecs = np.transpose(rho_eigenvecs)
    
    if eta == 0:
        rho_eigenvecs[-1][0] = -rho_eigenvecs[-1][0]
        
    rho_eta_diff = make_rho_eta_diff(eta, rho_size, n_bar)
    
    qfi = 0
    for i in range(rho_size):
        for j in range(rho_size):
            if rho_eigenvals[i] + rho_eigenvals[j] != 0:
                qfi += 2 * abs(np.conj(rho_eigenvecs[i]) @ rho_eta_diff @ rho_eigenvecs[j])**2 / (rho_eigenvals[i] + rho_eigenvals[j])

    return qfi



####################################################################################################################################



# Making the qfi range eta array

rho_size, n_bar = 3, 0.02
eta_vals = np.arange(0, 1, 0.01)

qfi_range_eta_array = []
for eta in eta_vals:
    qfi_range_eta_array.append(compute_qfi(eta, rho_size, n_bar))

log_qfi = np.log10(qfi_range_eta_array)


# Comparing the SLD fisher info plot and the qfi plot vs eta!

sld_fisher_info_array = []
for eta in eta_vals:
    sld_fisher_info_array.append(compute_sld_fisher_info(eta, rho_size, n_bar))

fig, ax = plt.subplots(2, 1, figsize=(6, 10))

ax[0].plot(eta_vals, sld_fisher_info_array, label='SLD')
ax[0].plot(eta_vals, qfi_range_eta_array, label='QFI')
ax[0].set_title('SLD FI and QFI vs Eta')
ax[0].set_xlabel('Eta')
ax[0].set_ylabel('Fisher info')
ax[0].text(0.2, 3.7, f'rho size: {rho_size} \nn_bar: {n_bar}')
ax[0].legend()

ax[1].plot(eta_vals, np.log10(sld_fisher_info_array), label='SLD')
ax[1].plot(eta_vals, np.log10(qfi_range_eta_array), label='QFI')
ax[1].set_title('SLD FI and QFI vs Eta in logscale')
ax[1].set_xlabel('Eta')
ax[1].set_ylabel('log10(Fisher info)')
ax[1].text(0.2, 0.5, f'rho size: {rho_size} \nn_bar: {n_bar}')
ax[1].legend()

plt.savefig('sld-fi-and-qfi-vs-eta.jpg', dpi=150)
# plt.savefig('sld-fi-and-qfi-vs-eta.svg', dpi=1200)
plt.savefig('sld-fi-and-qfi-vs-eta.pdf')
plt.close()



# Plot FI of SLD povm at various etas as a function of eta, compare with QFI

# Making the FI arrays for 6 eta values of SLD POVMs

rho_size, n_bar = 3, 0.02
sld_povms_fi_dict = {}

for eta in np.arange(0, 1.2, 0.2): # 6 points
    sld_povms_fi_dict[f'{eta}'] = {'povm': make_sld_povm_array(eta, rho_size, n_bar), 'fi': []}

for key in sld_povms_fi_dict:
    for eta in eta_vals:
        sld_povms_fi_dict[key]['fi'].append(get_fisher_info_quad(eta, sld_povms_fi_dict[key]['povm'], rho_size, n_bar))



# Plotting

fig, axes = plt.subplots(2, 3, figsize=(15,8))
plt.suptitle(f'SLD FI at different etas against the QFI \nrho_size = {rho_size}, n_bar = {n_bar}', fontsize=16)

for i in range(6):
    key = list(sld_povms_fi_dict.keys())[i]
    axes.ravel()[i].plot(eta_vals, sld_povms_fi_dict[key]['fi'], label=f'eta = {key}')
    axes.ravel()[i].plot(eta_vals, qfi_range_eta_array, label='qfi')
    axes.ravel()[i].legend()
    axes.ravel()[i].set_xlabel('Eta')
    axes.ravel()[i].set_ylabel('FI')

plt.savefig('sld-fi-compare-various-etas-qfi.jpg', dpi=150)
plt.savefig('sld-fi-compare-various-etas-qfi.pdf')
plt.close()



# In log10scale

fig, axes = plt.subplots(2, 3, figsize=(15,8))
plt.suptitle(f'SLD FI at different etas against the QFI (logscale) \nrho_size = {rho_size}, n_bar = {n_bar}', fontsize=16)

for i in range(6):
    key = list(sld_povms_fi_dict.keys())[i]
    axes.ravel()[i].plot(eta_vals, np.log10(sld_povms_fi_dict[key]['fi']), label=f'eta = {key}')
    axes.ravel()[i].plot(eta_vals, np.log10(qfi_range_eta_array), label='qfi')
    axes.ravel()[i].legend()
    axes.ravel()[i].set_xlabel('Eta')
    axes.ravel()[i].set_ylabel('log10(FI)')
    axes.ravel()[i].set_ylim(bottom=-1.5)

plt.savefig('sld-fi-compare-qfi-various-etas-log.jpg', dpi=150)
plt.savefig('sld-fi-compare-qfi-various-etas-log.pdf')
plt.close()



# Plot QFI vs nbar

rho_size = 3
eta_vals = np.arange(0, 1, 0.01)
nbar_vals = np.arange(0, 0.04, 0.005)

for n_bar in nbar_vals:
    qfi_range_eta_array = []
    for eta in eta_vals:
        qfi_range_eta_array.append(compute_qfi(eta, rho_size, n_bar))
    
    plt.plot(eta_vals, qfi_range_eta_array, label=f'n_bar = {n_bar}')

plt.title('QFI vs eta for increasing n_bar')
plt.xlabel('Eta')
plt.ylabel('QFI')
plt.text(0.4, 6.5, f'rho_size = {rho_size}')
plt.legend()
plt.savefig('qfi-with-increasing-nbar.pdf')
plt.close()



# In log10scale

for n_bar in nbar_vals:
    qfi_range_eta_array = []
    for eta in eta_vals:
        qfi_range_eta_array.append(compute_qfi(eta, rho_size, n_bar))
    
    plt.plot(eta_vals, np.log10(qfi_range_eta_array), label=f'n_bar = {n_bar}')

plt.title('log10(QFI) vs eta for increasing n_bar')
plt.xlabel('Eta')
plt.ylabel('log10(QFI)')
plt.text(0.4, 0.75, f'rho_size = {rho_size}')
plt.legend()
plt.savefig('log-qfi-with-increasing-nbar.pdf')
plt.close()



# Comparing direct detection FI to QFI

rho_size, n_bar = 3, 0.02
eta_vals = np.arange(0, 1, 0.01)

direct_fi_range_eta = []
for eta in eta_vals:
    direct_fi_range_eta.append(get_fisher_info_eta(make_hadamard_povms(1, rho_size), eta, rho_size, n_bar))

plt.plot(eta_vals, np.log10(direct_fi_range_eta), label='Direct detection')
plt.plot(eta_vals, np.log10(qfi_range_eta_array), label='Quantum fisher info')
plt.title('Direct detection FI and QFI on log10scale')
plt.xlabel('Eta')
plt.ylabel('log10(FI)')
plt.text(0.5, 0.2, f'rho size: {rho_size} \nn_bar: {n_bar}')
plt.legend()
plt.savefig('direct-fi-compare-qfi-log.pdf')
plt.savefig('direct-fi-compare-qfi-log.jpg', dpi=150)
plt.close()



# Comparing homodyne detection to QFI

rho_size, n_bar = 3, 0.02
eta_vals = np.arange(0, 1, 0.01)

x_fi_range_eta_array = []

for eta in eta_vals:
    x_fi_range_eta_array.append(get_fisher_info_quad(eta, make_pos_povms(rho_size), rho_size, n_bar))

p_fi_range_eta_array = []

for eta in eta_vals:
    p_fi_range_eta_array.append(get_fisher_info_quad(eta, make_mom_povms(rho_size), rho_size, n_bar))



# DD and Homodyne detection FI and QFI normal and logscale

fig, ax = plt.subplots(1, 3, figsize=(20, 5))
plt.suptitle(f'FI for our POVMs and the QFI vs eta (rho size: {rho_size}, n_bar: {n_bar})', fontsize=16)

ax[0].plot(eta_vals, np.log10(direct_fi_range_eta), label='Direct detection')
ax[0].plot(eta_vals, log_qfi, label='qfi')
ax[0].set_title('direct detection log10(FI) vs eta')
ax[0].set_xlabel('Eta')
ax[0].set_ylabel('log10(FI)')
ax[0].set_ylim(ymin = -4.1, ymax=0.6)
# ax[0].text(0.5, 0.2, f'rho size: {rho_size} \nn_bar: {n_bar}')
ax[0].legend()

ax[1].plot(eta_vals, np.log10(x_fi_range_eta_array), label='x-op: {a^(dag) + a}')
ax[1].plot(eta_vals, log_qfi, label='qfi')
ax[1].set_title('x-op log10(FI) vs eta')
ax[1].set_xlabel('Eta')
ax[1].set_ylabel('log10(FI)')
ax[1].set_ylim(ymin = -4.1, ymax=0.6)
# ax[1].text(0.5, 0.2, f'rho size: {rho_size} \nn_bar: {n_bar}')
ax[1].legend()

# ax[1].plot(eta_vals, p_fi_range_eta_array, label='p-op: i{a^(dag) - a}')
ax[2].plot(eta_vals, np.log10(p_fi_range_eta_array), label='p-op: i{a^(dag) - a}')
ax[2].plot(eta_vals, log_qfi, label='qfi')
ax[2].set_title('p-op FI vs eta')
ax[2].set_xlabel('Eta')
ax[2].set_ylabel('log10(FI)')
ax[1].set_ylim(ymin = -4.1, ymax=0.6)
# ax[2].text(0.5, 0.2, f'rho size: {rho_size} \nn_bar: {n_bar}')
ax[2].legend()

plt.savefig('all-fi-compare-qfi.jpg', dpi=150)
plt.savefig('all-fi-compare-qfi.pdf')
plt.close()


# X-op FI and SLD FI at eta=0

fig, ax = plt.subplots(1, 3, figsize=(20, 5))
plt.suptitle(f'FI for SLD POVM at eta=0 and x-quadrature POVM in logscale, (rho_size = {rho_size}, n_bar = {n_bar})', fontsize=16)

ax[0].plot(eta_vals, np.log10(sld_povms_fi_dict['0.0']['fi']), label='SLD POVM at eta=0')
ax[0].plot(eta_vals, np.log10(x_fi_range_eta_array), label='x-quadrature POVM')
# ax[0].plot(eta_vals, np.log10(qfi_range_eta_array), label='QFI')
# ax[0].set_title('FI for SLD POVM at eta=0 and x-quadrature POVM')
ax[0].set_xlabel('eta')
ax[0].set_ylabel('log10(FI)')
ax[0].legend()

ax[1].plot(eta_vals, np.log10(sld_povms_fi_dict['0.0']['fi']), label='SLD POVM at eta=0')
ax[1].plot(eta_vals, np.log10(x_fi_range_eta_array), label='x-quadrature POVM')
ax[1].plot(eta_vals, np.log10(qfi_range_eta_array), label='QFI')
# ax[1].set_title('FI for SLD POVM at eta=0 and x-quadrature POVM')
ax[1].set_xlabel('eta')
ax[1].set_ylabel('log10(FI)')
ax[1].legend()

ax[2].plot(eta_vals, np.log10(sld_povms_fi_dict['0.0']['fi']), label='SLD POVM at eta=0')
ax[2].plot(eta_vals, np.log10(x_fi_range_eta_array), label='x-quadrature POVM')
ax[2].plot(eta_vals, np.log10(qfi_range_eta_array), label='QFI')
ax[2].set_title('Zooming in...')
ax[2].set_xlabel('eta')
ax[2].set_ylabel('log10(FI)')
ax[2].legend()
ax[2].set_ylim(ymin=-1.125, ymax=-0.925)
# plt.text(0.5, 0.0863, f'rho_size = {rho_size} \nn_bar = {n_bar}')

plt.savefig('sld-povm-eta0-and-x-op-fi.jpg', dpi=150)
plt.savefig('sld-povm-eta0-and-x-op-fi.pdf')
plt.close()