# -*- coding: utf-8 -*-

import math
import numpy as np
import scipy as sp
from scipy.special import gamma
import matplotlib.pyplot as plt


def hermitian(A, **kwargs):
    return np.transpose(A, **kwargs).conj()
# Make some shortcuts for transpose,hermitian:
#    np.transpose(A) --> T(A)
#    hermitian(A) --> H(A)
T = np.transpose
C = np.conj
H = hermitian

import routines
import Ty_estim_REAL
import R_shape_estim_REAL_mv
from scipy.linalg import toeplitz

Ns = 10 ** 6
N = 8
perturbation_par = 10**(-2)
nu_par=5
rho = 0.8
sigma2 = 4

svect = np.linspace(0.1, 2, 20)
Nl = len(svect)

K = 5 * N
n = np.arange(0, N, 1)

rx = rho ** n
Sigma = sp.linalg.toeplitz(rx)
Ls = T(sp.linalg.cholesky(Sigma))
Shape_S = N*Sigma/np.trace(Sigma)
Ln = routines.L(N)
Dn = routines.Dup(N)
Inv_Shape_S = sp.linalg.inv(Shape_S)

DIM = int(N*(N+1)/2)
Fro_MSE_SCM = np.empty(Nl)
Fro_MSE_Ty = np.empty(Nl)
Fro_MSE_Rm = np.empty(Nl)
Fro_MSE_Rf = np.empty(Nl)

CRBn = np.empty(Nl)
SCRBn = np.empty(Nl)
L2_bias_SCM=np.empty(Nl)
L2_bias_Ty=np.empty(Nl)
L2_bias_Rm=np.empty(Nl)
L2_bias_Rf=np.empty(Nl)


J_phi = routines.jacobian_constraint_real(N)
U = sp.linalg.null_space(T(J_phi))
for il in range(Nl):
    s = svect[il]
    print(s)

    b = ( sigma2*N*math.gamma(N/(2*s))/( 2**(1/s)*math.gamma( (N+2)/(2*s) ) ) )**s

    MSE_SCM = np.zeros((DIM,DIM))
    MSE_Ty = np.zeros((DIM,DIM))
    MSE_Rm = np.zeros((DIM,DIM))
    MSE_Rf = np.zeros((DIM,DIM))



    bias_SCM = np.zeros((1,DIM))
    bias_Ty = np.zeros((1,DIM))
    bias_Rm= np.zeros((1,DIM))
    bias_Rf= np.zeros((1,DIM))
    for i in range(Ns):
        # Generation of the GG data
        w = np.random.randn(N, K)
        #w_norm = np.linalg.norm(w, axis=0)
        w_n = w / np.linalg.norm(w, axis=0)
        x = Ls @ w_n
        R = np.random.gamma(N/(2*s), 2*b, size=K)
        y = (R ** (1/(2*s))) * x

        # Sample Mean and Sample Covariance Matrix (SCM)
        SCM = y @ T(y) / K
        Scatter_SCM = N*SCM/np.trace(SCM)
        err_s = routines.vech(Scatter_SCM-Shape_S)
        err_SCM = np.outer(err_s, err_s)
        MSE_SCM = MSE_SCM + err_SCM/Ns
        #bias_SCM=np.add(bias_SCM,err_s/Ns)
        bias_SCM = bias_SCM+err_s/Ns


        # Tyler matrix estimator
        Ty1 = Ty_estim_REAL.compute_tyler_shape_estimator(T(y), np.eye(N))
        Ty = N * Ty1  / np.trace(Ty1)
        
        # MSE mismatch on sigma
        err_v = routines.vech(Ty-Shape_S)
        err_Ty = np.outer(err_v, err_v)
        MSE_Ty = MSE_Ty + err_Ty/Ns
        bias_Ty = bias_Ty + err_v / Ns

        # R-estimator with the van der Waerden score function
        mu0 = np.zeros((N,))
        Rm1 = R_shape_estim_REAL_mv.R_estimator_VdW_score_mv(y, mu0, Ty1, perturbation_par)
        Rm = N * Rm1  / np.trace(Rm1)
        
        # MSE mismatch on sigma
        err_rm = routines.vech(Rm-Shape_S)
        err_RM = np.outer(err_rm, err_rm)
        MSE_Rm = MSE_Rm + err_RM/Ns
        bias_Rm = bias_Rm + err_rm/Ns
        # R-estimator with  t_nu-distribution
        mu1 = np.zeros((N,))
        Rf1 = R_shape_estim_REAL_mv.R_estimator_F_score_mv(y, mu0, Ty1, perturbation_par,nu_par)
        Rf = N * Rf1 / np.trace(Rf1)

        # MSE mismatch on sigma
        err_rf = routines.vech(Rf - Shape_S)
        err_RF = np.outer(err_rf, err_rf)
        MSE_Rf = MSE_Rf + err_RF / Ns
        bias_Rf = bias_Rf + err_rf / Ns


    #print(bias_SCM)

    # Semiparametric CRB
    #define CRB
    a1 = (s-1)/(2*(N+2))
    a2 = (N + 2*s)/(2*(N+2))
    FIM_Sigma = K * T(Dn) @ (a1*np.outer(routines.vec(Inv_Shape_S), routines.vec(Inv_Shape_S))+a2*np.kron(Inv_Shape_S,Inv_Shape_S))@ Dn
    CRB = U @ sp.linalg.inv(T(U) @ FIM_Sigma @ U) @ T(U)
    CRBn[il] = np.linalg.norm(CRB, ord='fro')

    # define SCRB
    SFIM_Sigma = K * a2 * T(Dn) @ (np.kron(Inv_Shape_S,Inv_Shape_S) - (1/N) * np.outer(routines.vec(Inv_Shape_S), routines.vec(Inv_Shape_S)) ) @ Dn
    SCRB = U @ sp.linalg.inv(T(U) @ SFIM_Sigma @ U) @ T(U)
    SCRBn[il] = np.linalg.norm(SCRB, ord='fro')

    Fro_MSE_SCM[il] = np.linalg.norm(MSE_SCM, ord='fro')
    Fro_MSE_Ty[il] = np.linalg.norm(MSE_Ty, ord='fro')
    Fro_MSE_Rm[il] = np.linalg.norm(MSE_Rm, ord='fro')
    Fro_MSE_Rf[il] = np.linalg.norm(MSE_Rf, ord='fro')

    L2_bias_SCM[il]=np.linalg.norm(bias_SCM)
    L2_bias_Ty[il]=np.linalg.norm(bias_Ty)
    L2_bias_Rm[il]=np.linalg.norm(bias_Rm)
    L2_bias_Rf[il]=np.linalg.norm(bias_Rf)


fig = plt.figure(1)
plt.title('MSE in Frobenus norm')
plt.plot(svect, Fro_MSE_SCM, svect, Fro_MSE_Ty, svect, Fro_MSE_Rm,svect, Fro_MSE_Rf,  svect, SCRBn,svect, CRBn)
plt.legend(['SCM','Ty','Rvdw','Rf','SCRB','CRB'])
plt.ylim((0.2,0.6))

plt.ylabel('MSE and Bound')
plt.xlabel('Shape parameter: s')
plt.show()

fig2 = plt.figure(2)
plt.title('Bias in Euclidean norm')
plt.plot(svect, L2_bias_SCM, svect, L2_bias_Ty, svect, L2_bias_Rm,svect, L2_bias_Rf)
plt.legend(['SCM','Ty','Rvdw','Rf'])
plt.ylabel('Frobenius norm')
plt.xlabel('Shape parameter: s')
plt.show()

