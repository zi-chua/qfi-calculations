# Computing the FI for projective and hadamard povms - a mess

import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import sympy as sym
import pandas as pd
sns.set()
from rho_eta_build import *


# Making projective povms

def make_proj_povms_old(rho_size): 
    """Adds extra povm element to normalise, doesn't affect answer much"""
    
    proj_povms = []
    fock_basis_array = make_fock_basis_array(rho_size)
    
    for i in range(rho_size + 1):
        if i < rho_size:
            proj_povms.append(np.outer(fock_basis_array[i], fock_basis_array[i]))
        elif i == rho_size:
            proj_povms.append(np.identity(rho_size, dtype=float) - np.sum(proj_povms, axis = 0)) # this is just a 0 matrix, but it's a placeholder w defined properties
    
    return proj_povms

def make_proj_povms(rho_size):
    proj_povms = []
    fock_basis_array = make_fock_basis_array(rho_size)
    
    for i in range(rho_size):
        proj_povms.append(np.outer(fock_basis_array[i], fock_basis_array[i]))
    
    return proj_povms

# Making Hadamard povms

def make_new_povms(alpha, rho_size):
    """Adds extra povm element to normalise, doesn't affect answer much"""
    
    fock_basis_array = make_fock_basis_array(rho_size)
    beta = np.sqrt(1 - alpha**2)
    
    psi_0 = (alpha * fock_basis_array[0]) + (beta * fock_basis_array[1])
    psi_1 = (beta * fock_basis_array[0]) - (alpha * fock_basis_array[1])

    E0 = np.outer(psi_0, psi_0)
    E1 = np.outer(psi_1, psi_1)
    E2 = np.outer(fock_basis_array[2], fock_basis_array[2])
    E3 = np.identity(rho_size, dtype = float) - (E0 + E1 + E2)
    
    new_povms = [E0, E1, E2, E3]
    
    return new_povms

def make_hadamard_povms(alpha, rho_size):
    fock_basis_array = make_fock_basis_array(rho_size)
    beta = np.sqrt(1 - alpha**2)
    
    psi_0 = (alpha * fock_basis_array[0]) + (beta * fock_basis_array[1])
    psi_1 = (beta * fock_basis_array[0]) - (alpha * fock_basis_array[1])

    E0 = np.outer(psi_0, psi_0)
    E1 = np.outer(psi_1, psi_1)
    E2 = np.outer(fock_basis_array[2], fock_basis_array[2])
    
    hadamard_povms = [E0, E1, E2]
    
    return hadamard_povms

# Fisher info - many versions, need to clean up

def get_fisher_info_eta(povm_array, eta, rho_size, n_bar): 
    """Only works for povms that aren't functions of eta"""
    
    eta_var = sym.Symbol('eta_var')
    
    rho_diff = make_rho_eta_diff(0, rho_size, n_bar)

    fisher_info_expr = 0
    for povm in povm_array:
        prob = np.trace(povm @ make_rho_eta(eta_var, rho_size, n_bar))
        fisher_info_expr += (np.trace(povm @ rho_diff))**2 / prob
    
    fisher_info_func = sym.lambdify(eta_var, fisher_info_expr, 'numpy')
    fisher_info = fisher_info_func(eta)

    return fisher_info

# rho_size, n_bar = 3, 0.01
# eta_vals = np.arange(0, 1, 0.01)
# direct_fi_range_eta = []
# for eta in eta_vals:
#     direct_fi_range_eta.append(get_fisher_info_eta(make_hadamard_povms(1, rho_size), eta, rho_size, n_bar))

# plt.plot(eta_vals, direct_fi_range_eta)

# def are_same_square_matrix(A, B):
#    for i in range(np.size(A, axis = 0)):
#       for j in range(np.size(A, axis = 0)):
#          if (A[i, j] != B[i, j]):
#             return False
   
#    return True

# def get_prob_roundabout(phot_count, eta, povm_array, rho_size, n_bar): 
#     """Only works for povm arrays where I define an extra E = 1 - (sum of other povms), 
#     bc I need to collect the povms in an array to be able to calculate the prob for the last povm"""
    
#     rho_eta = make_rho_eta(eta, rho_size, n_bar)
    
#     if phot_count < rho_size: # 0, 1, 2
#         prob_outcome = np.trace(povm_array[phot_count].dot(rho_eta))
    
#     elif phot_count == rho_size: # phot_count = 3
#         if are_same_square_matrix(povm_array[phot_count], np.zeros([rho_size, rho_size])) == False:
#             prob_outcome = np.trace(povm_array[phot_count].dot(rho_eta))
        
#         else:
#             prob_outcome = 1
#             for i in range(rho_size): # 0, 1, 2
#                 prob_outcome -= np.trace(povm_array[i].dot(rho_eta))

#     return prob_outcome

# def get_fisher_info(eta, povm_array, rho_size, n_bar): # ver 0
#     eta_var = sym.symbols('x')
#     fisher_info = 0
    
#     for phot_count in range(rho_size + 1): # 0, 1, 2, 3
#         fisher_info += sym.diff(get_prob_old(phot_count, eta_var, povm_array, rho_size, n_bar), eta_var)**2 / get_prob_old(phot_count, eta, povm_array, rho_size, n_bar)

#     return fisher_info


#### fisher info vs alpha and eta ################################################################################################################################


# Fisher info vs eta

def make_fisher_range_eta_array(povm_array, eta_start, eta_stop, rho_size, n_bar):
    eta_vals = np.arange(eta_start, (eta_stop + 1/20), (eta_stop-eta_start)/20)
    fisher_range_eta_array = []

    for eta_val in eta_vals:
        fisher_range_eta_array.append(get_fisher_info(eta_val, povm_array, rho_size, n_bar))

    fisher_range_eta_array = np.array(fisher_range_eta_array)

    return eta_vals, fisher_range_eta_array

# rho_size, n_bar = 3, 0.02
# fig, ax = plt.subplots(2, 2, figsize=(10, 10))

# alpha = 1
# proj_povm_array = make_new_povm_array(alpha, rho_size)
# proj_eta_vals, proj_fisher_range_eta_array = make_fisher_range_eta_array(proj_povm_array, 0, 1, rho_size, n_bar)
# ax[0, 0].plot(proj_eta_vals, proj_fisher_range_eta_array)
# ax[0, 0].set_title('Fisher Information vs Eta (Projective povms)')
# ax[0, 0].set_xlabel('Eta')
# ax[0, 0].set_ylabel('Fisher info')
# ax[0, 0].text(min(proj_eta_vals), 0.9*max(proj_fisher_range_eta_array), f'rho size: {rho_size} \nn_bar: {n_bar} \nalpha: {alpha}')

# log_proj_fisher_range_eta_array = [math.log10(f) for f in proj_fisher_range_eta_array]
# ax[0, 1].plot(proj_eta_vals, log_proj_fisher_range_eta_array)
# ax[0, 1].set_title('log(Fisher Information) vs Eta (Projective povms)')
# ax[0, 1].set_xlabel('Eta')
# ax[0, 1].set_ylabel('log(Fisher info)')
# ax[0, 1].text(min(proj_eta_vals), -3.75, f'rho size: {rho_size} \nn_bar: {n_bar} \nalpha: {alpha}')

# alpha = 0.5
# new_povm_array = make_new_povm_array(alpha, rho_size)
# new_eta_vals, new_fisher_range_eta_array = make_fisher_range_eta_array(new_povm_array, 0, 1, rho_size, n_bar)
# ax[1, 0].plot(new_eta_vals, new_fisher_range_eta_array)
# ax[1, 0].set_title('Fisher Information vs Eta (New povms)')
# ax[1, 0].set_xlabel('Eta')
# ax[1, 0].set_ylabel('Fisher info')
# ax[1, 0].text(0.7, 0.073, f'rho size: {rho_size} \nn_bar: {n_bar} \nalpha: {alpha}')

# log_new_fisher_range_eta_array = [math.log10(f) for f in new_fisher_range_eta_array]
# ax[1, 1].plot(new_eta_vals, log_new_fisher_range_eta_array)
# ax[1, 1].set_title('log(Fisher Information) vs Eta (New povms)')
# ax[1, 1].set_xlabel('Eta')
# ax[1, 1].set_ylabel('log(Fisher Info)')
# ax[1, 1].text(0.7, -1.135, f'rho size: {rho_size} \nn_bar: {n_bar} \nalpha: {alpha}')

# fig.savefig('fisher_vs_eta.jpg')

# plt.close()

# nbar_ratio_array = proj_eta_vals[:-1] / (1 - proj_eta_vals[:-1])
# x, y = nbar_ratio_array, proj_fisher_range_eta_array[:-1]
# print (nbar_ratio_array)
# plt.plot(x, y)
# plt.title('Fisher Information vs Ratio of n_bars (Direct Detection)')
# plt.xlabel('n_bar_coh/n_bar_th')
# plt.ylabel('Fisher information')
# plt.text(0.75*max(x), 0.9*max(y), f'Size of density matrix: {rho_size} \nSignal strength: {n_bar}')



# Fisher info vs alpha

# def make_fisher_range_alpha_array(alpha_start, alpha_stop, eta, rho_size, n_bar):
#     alpha_vals = np.arange(alpha_start, (alpha_stop+1/20), (alpha_stop-alpha_start)/20)
#     fisher_range_alpha_array = []

#     for alpha in alpha_vals:
#         new_povm_array = make_new_povm_array(alpha, rho_size)
#         fisher_range_alpha_array.append(get_fisher_info(eta, new_povm_array, rho_size, n_bar))

#     return alpha_vals, fisher_range_alpha_array

# eta, rho_size, n_bar = 0.5, 3, 0.02
# alpha_vals, fisher_range_alpha_array = make_fisher_range_alpha_array(0, 1, eta, rho_size, n_bar)



# Fisher vs alpha and eta

# Making a DataFrame

def make_fisher_new_povms_dict(alpha_start, alpha_stop, eta_start, eta_stop, n_bar_start, n_bar_stop, rho_size):
    alpha_vals = np.arange(alpha_start, (alpha_stop + 1/20), (alpha_stop-alpha_start)/20)
    eta_vals = np.arange(eta_start, (eta_stop + 1/20), (eta_stop-eta_start)/20)
    n_bar_vals = np.arange(n_bar_start, (n_bar_stop + (n_bar_stop-n_bar_start)/5), (n_bar_stop-n_bar_start)/5)
    
    fisher_dict = {'alpha_index': [], 'eta_column': [], 'n_bar_column': [], 'fisher_column': []}
    
    for alpha in alpha_vals:
        new_povm_array = make_new_povm_array(alpha, rho_size)

        for eta in eta_vals:
            
            for n_bar in n_bar_vals:
                fisher_dict['alpha_index'].append(alpha)
                fisher_dict['eta_column'].append(eta)
                fisher_dict['n_bar_column'].append(n_bar)
                
                fisher_info = get_fisher_info(eta, new_povm_array, rho_size, n_bar)
                fisher_dict['fisher_column'].append(float(fisher_info))
                # fisher_range_eta_array = make_fisher_range_eta_array(new_povm_array, eta_start, eta_stop, rho_size, nbar)[1]

    return fisher_dict

# alpha_start, alpha_stop, eta_start, eta_stop, n_bar_start, n_bar_stop, rho_size = 0, 1, 0, 1, 0.005, 0.02, 3
# fisher_dict = make_fisher_new_povms_dict(alpha_start, alpha_stop, eta_start, eta_stop, n_bar_start, n_bar_stop, rho_size)
# fisher_df = pd.DataFrame(fisher_dict) # n_bar = 0.02

# x = fisher_df[fisher_df['alpha_index'] == 1][fisher_df['eta_column'] == 0.5]['n_bar_column']
# y = fisher_df[fisher_df['alpha_index'] == 1][fisher_df['eta_column'] == 0.5]['fisher_column']
# plt.plot(x, y)

# Making the 3D plot

# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm

# x = fisher_df['alpha_index']
# y = fisher_df['eta_column']
# z = fisher_df['fisher_column']

# p = fisher_df[fisher_df['alpha_index'] == 1]['eta_column']
# q = fisher_df[fisher_df['alpha_index'] == 1]['n_bar_column']
# r = fisher_df[fisher_df['alpha_index'] == 1]['fisher_column']


# from matplotlib.colors import ListedColormap

# cmap = ListedColormap(sns.color_palette("coolwarm", 7))
# cmap = sns.cubehelix_palette(8, as_cmap = True)

# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, z, c=z, cmap = cm.jet)
# ax.set_title('Fisher info vs eta and alpha')
# ax.set_xlabel('Alpha')
# ax.set_ylabel('Eta')
# ax.set_zlabel('Fisher info')
# ax.text(0.1, 0.1, 0.2, f'n_bar = {n_bar}')
# # plt.show()
# plt.savefig('fisher_vs_alpha_eta.jpg')
# plt.close()

# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(p, q, r, c=r, cmap = cm.jet)
# ax.set_title('Fisher info vs eta and n_bar')
# ax.set_xlabel('Eta')
# ax.set_ylabel('N_bar')
# ax.set_zlabel('Fisher info')
# ax.text(min(p), min(q), max(r), 'alpha = 1')
# plt.show()
# plt.savefig('fisher_vs_alpha_eta.jpg')
# plt.close()
