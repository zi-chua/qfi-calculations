# Homodyne detection

import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import sympy as sym
import pandas as pd
sns.set()
from rho_eta_build import *

# Quadrature POVMs

def make_annihilation_op(rho_size):
    op = np.zeros([rho_size, rho_size], dtype=float)
    op[0, 1] = np.sqrt(1)
    op[1, 2] = np.sqrt(2)

    return op

def make_creation_op(rho_size):
    op = np.zeros([rho_size, rho_size], dtype=float)
    op[1, 0] = np.sqrt(1)
    op[2, 1] = np.sqrt(2)

    return op

def make_pos_povms(rho_size):
    """{a^(dag) + a}"""

    povms = []
    pos_op = make_annihilation_op(rho_size) + make_creation_op(rho_size)
    pos_eigvals, pos_eigvecs = np.linalg.eigh(pos_op)
    pos_eigvecs = np.transpose(pos_eigvecs)

    for eigvec in pos_eigvecs:
        povms.append(np.outer(eigvec, eigvec))

    return povms

def make_mom_povms(rho_size):
    """i{a^(dag) - a}"""
    
    povms = []
    mom_op = complex(0, 1) * (make_creation_op(rho_size) - make_annihilation_op(rho_size))
    mom_eigvals, mom_eigvecs = np.linalg.eigh(mom_op)
    mom_eigvecs = np.transpose(mom_eigvecs)

    for eigvec in mom_eigvecs:
        povms.append(np.outer(eigvec, np.conj(eigvec)))

    return povms

# Fisher info

def get_fisher_info_quad(eta, povms, rho_size, n_bar):
    fisher_info = 0
    for povm in povms:
        # print (np.trace(povm @ make_rho_eta_diff(eta, rho_size, n_bar)))
        fisher_info += (np.trace(povm @ make_rho_eta_diff(eta, rho_size, n_bar)))**2 / np.trace(povm @ make_rho_eta(eta, rho_size, n_bar))

    return fisher_info