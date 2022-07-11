# -*- coding: utf-8 -*-

import numpy as np
import math
import scipy as sp
import matplotlib.pyplot as plt
from numpy import  matlib
from scipy.linalg import sqrtm
import routines
def hermitian(A, **kwargs):
    return np.transpose(A, **kwargs).conj()
# Make some shortcuts for transpose,hermitian:
#    np.transpose(A) --> T(A)
#    hermitian(A) --> H(A)
T = np.transpose
C = np.conj
H = hermitian

def compute_tyler_joint_estimator(X, Sigma_init, max_iter_fp = 250):
    # Computes tyler's estimators for mu and Sigma with the data of the cluster (hard assigment)
    '''
input :  X the matrix which should be estimated
        Sigma_init: np.eye(N) with K,N=X.shape
        max_iter_fp: max number of iteration (250 by default )

        output:FP matrix estimator  R_Ty
                                    mu_Ty

    '''
    K, N = X.shape
    Sigma_fixed_point = Sigma_init.copy()
    sq_maha = np.empty((K, ))
    convergence_fp = False
    ite_fp = 1
    mu0 = np.zeros(N)
    z0=X
    while not(convergence_fp) and ite_fp < max_iter_fp:
        z = z0 - np.matlib.repmat(mu0, K, 1)
        inv_Sigma_fixed_point = np.linalg.inv(Sigma_fixed_point)
        sq_maha = ( (C(z) @ inv_Sigma_fixed_point) * z).sum(1)
        sq_maha=np.real(sq_maha)
        sq_maha=sq_maha.reshape(sq_maha.size,1)

        r_inv =  sq_maha **  (-0.5)
        r_inv=r_inv.reshape(r_inv.size,1)

        mu = np.sum(z0*np.matlib.repmat(r_inv, 1, N),axis=0) / np.sum(r_inv)
        w = N / sq_maha
        A = np.matlib.repmat(w, 1, N)
        Sigma_fixed_point_new = T(z) @ (C(z) * A) / K

        Sigma_fixed_point_new = Sigma_fixed_point_new/Sigma_fixed_point_new[0,0]

        convergence_fp = True
        convergence_fp = convergence_fp and (np.linalg.norm(Sigma_fixed_point_new-Sigma_fixed_point, ord='fro')/N) < 10**(-6)
        mu0=mu

        Sigma_fixed_point = Sigma_fixed_point_new.copy()

        ite_fp += 1
    mu=T(mu)

    return Sigma_fixed_point,mu