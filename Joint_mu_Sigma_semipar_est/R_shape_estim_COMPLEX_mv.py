# -*- coding: utf-8 -*-
import math
import numpy as np
import scipy as sp
from scipy.stats.distributions import gamma
import routines
from scipy.stats.distributions import f
from scipy.linalg import sqrtm
def hermitian(A, **kwargs):
    return np.transpose(A, **kwargs).conj()
T = np.transpose
C = np.conj
H = hermitian
from numpy import  matlib


def R_estimator_VdW_score_mv(y, mu, S0, pert):
    """
    # -----------------------------------------------------------
    # This function implement the R-estimator for shape matrices

    # Input:
    #   y: (N, K)-dim real data array where N is the dimension of each vector and K is the number of available data
    #   mu: N-dim array containing a preliminary estimate of the location       (isn't used in this case)
    #   S0: (N, N)-dim array containing a preliminary estimator of the scatter matrix
    #   pert: perturbation parameter

    # Output:
    # S_est: Estimated shape matrix with the normalization [S_est]_{1,1} = 1
    # -----------------------------------------------------------
    """

    N, K = y.shape

    S0 = S0 / S0[0, 0]

    #print(m_index)
    # Generation of the perturbation matrix
    V = pert * (np.random.randn(N, N) +1j*np.random.randn(N, N))
    V = (V + H(V)) / 2

    V[0, 0] = 0
    #print(V)
    alpha_est, Delta_S, W = alpha_estimator_sub_vdw_mv(y, S0, V)

    beta_est = 1 / alpha_est

    N_vdw_mv = S0 + beta_est * (W - W[0, 0] * S0)

    return N_vdw_mv

# R estimator with t_nu-distribution

def R_estimator_F_score_mv(y, mu, S0, pert, nu):
    """
    # -----------------------------------------------------------
    # This function implement the R-estimator for shape matrices

    # Input:
    #   y: (N, K)-dim real data array where N is the dimension of each vector and K is the number of available data
    #   mu: N-dim array containing a preliminary estimate of the location
    #   S0: (N, N)-dim array containing a preliminary estimator of the scatter matrix
    #   pert: perturbation parameter
    #   nu: coefficient of the t_nu distribution


    # Output:
    # S_est: Estimated shape matrix with the normalization [S_est]_{1,1} = 1
    # ---
    """
    N, K = y.shape

    S0 = S0 / S0[0, 0]


    # Generation of the perturbation matrix
    V = pert * (np.random.randn(N, N) +1j*np.random.randn(N, N))
    V = (V + H(V)) / 2

    V[0, 0] = 0

    alpha_est, Delta_S, W = alpha_estimator_sub_F_mv(y, S0, V, nu)

    beta_est = 1 / alpha_est

    N_f_mv = S0 + beta_est * (W - W[0, 0] * S0)

    return N_f_mv




# -----------------------------------------------------------
# Functions that will be used in the estimator

# Make some shortcuts for transpose,hermitian:
#    np.transpose(A) --> T(A)
#    hermitian(A) --> H(A)


def alpha_estimator_sub_F_mv(y, S0, V, nu):

    N, K = y.shape

    Delta_S, w,inv_T = Delta_Psi_eval_F_mv(y, S0,nu)
    S_pert = S0 + V / np.sqrt(K)
    Delta_S_pert = Delta_only_eval_F_mv(y, S_pert,nu)

    Z = inv_T.dot(V.dot(inv_T)) - np.trace(inv_T.dot(V)) / N*inv_T

    Z_vecs = Z[1:Z.size]
    Z_vecs=Z_vecs.reshape(Z_vecs.size,1)

    alpha_est = np.linalg.norm(Delta_S_pert - Delta_S) / np.linalg.norm( Z_vecs)

    return alpha_est, Delta_S, w

def alpha_estimator_sub_vdw_mv(y, S0, V):
    N, K = y.shape

    Delta_S, w,inv_T = Delta_Psi_eval_vdw_mv(y, S0)
    S_pert = S0 + V / np.sqrt(K)
    Delta_S_pert = Delta_only_eval_vdw_mv(y, S_pert)

    Z = inv_T.dot(V.dot(inv_T)) - np.trace(inv_T.dot(V)) / N*inv_T

    Z_vecs = Z[1:Z.size]
    Z_vecs=Z_vecs.reshape(Z_vecs.size,1)


    alpha_est = np.linalg.norm(Delta_S_pert - Delta_S) / np.linalg.norm( Z_vecs)

    return alpha_est, Delta_S, w



#S0=T


def Delta_only_eval_vdw_mv(y, S):
    N, K = y.shape

    score_vect, u, inv_sr_T ,inv_T = kernel_rank_sign_vdw_mv(y, S)

    score_mat = np.matlib.repmat(np.sqrt(score_vect), N, 1)
    U_appo = score_mat* u
    Score_appo_m = np.matmul(U_appo ,H(U_appo))

    D_T_m = inv_sr_T.dot( Score_appo_m.dot(inv_sr_T)) - sum(score_vect)*inv_T / N
    D_T_m=D_T_m.reshape(D_T_m.size,1)
    Delta_T = D_T_m[1:D_T_m.size] / np.sqrt(K)
    return Delta_T

def Delta_Psi_eval_vdw_mv(y, S):
    N, K = y.shape

    score_vect, u, inv_sr_T ,inv_T = kernel_rank_sign_vdw_mv(y, S)
    score_mat = np.matlib.repmat(np.sqrt(score_vect), N, 1)
    U_appo = score_mat* u
    Score_appo_m = np.matmul(U_appo ,H(U_appo))

    sr_T = sqrtm(S)
    W = sr_T.dot(Score_appo_m.dot(sr_T)) / K

    D_T_m = inv_sr_T.dot( Score_appo_m.dot(inv_sr_T)) - sum(score_vect)*inv_T / N
    D_T_m=D_T_m.reshape(D_T_m.size,1)
    Delta_T = D_T_m[1:D_T_m.size] / np.sqrt(K)

    return Delta_T,W,inv_T

def Delta_only_eval_F_mv(y, S,nu):
    N, K = y.shape

    score_vect, u, inv_sr_T ,inv_T = kernel_rank_sign_F_mv(y,S,nu,)

    score_mat = np.matlib.repmat(np.sqrt(score_vect), N, 1)
    U_appo = score_mat* u
    Score_appo_m = np.matmul(U_appo ,H(U_appo))

    D_T_m = inv_sr_T.dot( Score_appo_m.dot(inv_sr_T)) - sum(score_vect)*inv_T / N
    D_T_m=D_T_m.reshape(D_T_m.size,1)
    Delta_T = D_T_m[1:D_T_m.size] / np.sqrt(K)

    return Delta_T

def Delta_Psi_eval_F_mv(y, S,nu):
    N, K = y.shape

    score_vect, u, inv_sr_T ,inv_T = kernel_rank_sign_F_mv(y,S,nu)
    score_mat = np.matlib.repmat(np.sqrt(score_vect), N, 1)
    U_appo = score_mat* u
    Score_appo_m = np.matmul(U_appo ,H(U_appo))

    sr_T = sqrtm(S)
    W = sr_T.dot(Score_appo_m.dot(sr_T)) / K

    D_T_m = inv_sr_T.dot( Score_appo_m.dot(inv_sr_T)) - sum(score_vect)*inv_T / N
    D_T_m=D_T_m.reshape(D_T_m.size,1)
    Delta_T = D_T_m[1:D_T_m.size] / np.sqrt(K)

    return Delta_T,W,inv_T



def kernel_rank_sign_vdw_mv(y, S0):
    N, K = y.shape

    IN_S = sp.linalg.inv(S0)
    SR_IN_S = sp.linalg.sqrtm(IN_S)
    A_appo = SR_IN_S @ y
    Rq = np.real(np.linalg.norm(A_appo, axis=0))
    u = A_appo / Rq
    temp = Rq.argsort()
    ranks = np.arange(len(Rq))[temp.argsort()] + 1
    kernel_vect = gamma.ppf(ranks / (K + 1), N, scale=1)

    return kernel_vect, u, SR_IN_S, IN_S

def kernel_rank_sign_F_mv(y, S0,nu):
    N, K = y.shape

    IN_S = sp.linalg.inv(S0)
    SR_IN_S = sp.linalg.sqrtm(IN_S)
    A_appo = SR_IN_S @ y
    Rq = np.real(np.linalg.norm(A_appo, axis=0))
    u = A_appo / Rq
    temp = Rq.argsort()
    ranks = np.arange(len(Rq))[temp.argsort()] + 1

    f_inv_appo= f.ppf(ranks/ (K + 1), 2*N, nu)
    kernel_vect= N*f_inv_appo * ( 2 * N + nu)/ (nu +2 * N * f_inv_appo)
    return kernel_vect, u, SR_IN_S, IN_S