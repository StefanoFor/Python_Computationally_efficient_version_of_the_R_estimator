# -*- coding: utf-8 -*-

import math
import cmath
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
from scipy.linalg import sqrtm

import routines
import Ty_joint_estim_COMPLEX
import R_shape_estim_COMPLEX_mv


#from scipy.linalg import toeplitz

Ns = 10**6
N = 8
perturbation_par = 10**(-2)
nu_par=5

rho = 0.8*cmath.exp( 1j*2*math.pi/5 )
sigma2 = 4

svect = np.linspace(0.1, 2, 20)

Nl = len(svect)

K = 5 * N
n = np.arange(0, N, 1)

rx = rho ** n
Sigma = C(sp.linalg.toeplitz(rx))
mu_t = 0.5*np.exp(1j*2*np.pi/7*H(np.linspace(0,N-1,N)))

Ls = H(sp.linalg.cholesky(Sigma))
#Shape_S = N*Sigma/np.trace(Sigma)
Shape_S = Sigma/Sigma[0,0]
Inv_Shape_S = sp.linalg.inv(Shape_S)

T2 = np.kron(T(Inv_Shape_S),Inv_Shape_S)
sr_T2 = sqrtm(T2)

DIM = int(N**2)

In=np.eye(N)
J_n_per = np.eye(N**2) - np.matmul(In.reshape(In.size,1),T(In.reshape(In.size,1)))/N
In2=np.eye(N**2)
P = T(In2[:,1:In2.shape[1]])
K_V = np.matmul(P,sr_T2.dot(J_n_per))

J_phi = routines.vec(np.eye(N)).reshape(-1, 1)
U = sp.linalg.null_space(T(J_phi))
Fro_MSE_SM = np.empty(Nl)
Fro_MSE_mu_Ty = np.empty(Nl)
Fro_MSE_SCM = np.empty(Nl)
Fro_MSE_Ty = np.empty(Nl)
Fro_MSE_Rm = np.empty(Nl)
Fro_MSE_Rf = np.empty(Nl)
SCR_Bound = np.empty(Nl)
SCR_Bound_mean = np.empty(Nl)

CRBn = np.empty(Nl)
SCRBn = np.empty(Nl)
L2_bias_SM=np.empty(Nl)
L2_bias_mu_Ty=np.empty(Nl)
L2_bias_SCM=np.empty(Nl)
L2_bias_Ty=np.empty(Nl)
L2_bias_Rm=np.empty(Nl)
L2_bias_Rf=np.empty(Nl)

for il in range(Nl):
    s = svect[il]
    print(s)
    
    b = ( sigma2*N*math.gamma(N/s)/(math.gamma( (N+1)/s ) ) )**s

    MSE_SM = np.zeros((2*N,2*N))
    MSE_mu_Ty = np.zeros((2*N,2*N))
    MSE_SCM = np.zeros((DIM,DIM))
    MSE_Ty = np.zeros((DIM,DIM))
    MSE_Rm = np.zeros((DIM,DIM))
    MSE_Rf = np.zeros((DIM,DIM))



    bias_SM = np.zeros((1,DIM))
    bias_mu_Ty = np.zeros((1,DIM))
    bias_SCM = np.zeros((1,DIM))
    bias_Ty = np.zeros((1,DIM))
    bias_Rm= np.zeros((1,DIM))
    bias_Rf= np.zeros((1,DIM))
    
    for i in range(Ns):
        # Generation of the GG data
        w = (np.random.randn(N, K) + 1j*np.random.randn(N, K))/math.sqrt(2)
        #w_norm = np.linalg.norm(w, axis=0)
        w_n = w / np.linalg.norm(w, axis=0)
        x = Ls @ w_n
        R = np.random.gamma(N/s, b, size=K)
        mu_t=mu_t.reshape((mu_t.size,1))
        y =mu_t+ (R ** (1/(2*s))) * x
        # Sample Mean and Sample Covariance Matrix (SCM)
        SM = np.mean(y,1)
        SM=SM.reshape(SM.size,1)
        SCM = (y-SM) @ H(y-SM) / K
        Scatter_SCM = SCM/SCM[0,0]

        err_vect = np.append(SM - mu_t, C(SM - mu_t)).reshape((SM - mu_t).size * 2, 1)
        bias_SM = bias_SM + (SM - mu_t) / Ns
        err_SM = err_vect.dot(H(err_vect))
        MSE_SM = MSE_SM + err_SM / Ns


        err_s = routines.vec(Scatter_SCM-Shape_S)
        err_SCM = np.outer(err_s, C(err_s))
        MSE_SCM = MSE_SCM + err_SCM/Ns
        bias_SCM = bias_SCM+err_s/Ns

        # Tyler matrix estimator
        R_TY,mu_TY = Ty_joint_estim_COMPLEX.compute_tyler_joint_estimator(T(y), np.eye(N))

        mu_TY=mu_TY.reshape(mu_TY.size,1)
        A=mu_TY-mu_t
        err_vect = np.append(mu_TY - mu_t, C(mu_TY - mu_t)).reshape((mu_TY - mu_t).size * 2, 1)
        mu_TY=mu_TY.reshape(mu_TY.shape[0],1)

        bias_mu_Ty = bias_mu_Ty + (mu_TY - mu_t) / Ns
        err_mu_TY = err_vect.dot(H(err_vect))
        MSE_mu_Ty = MSE_mu_Ty + err_mu_TY / Ns

        # MSE mismatch on sigma
        err_v = routines.vec(R_TY-Shape_S)
        err_Ty = np.outer(err_v, H(err_v))
        MSE_Ty = MSE_Ty + err_Ty/Ns
        bias_Ty = bias_Ty + err_v / Ns

        # R-estimator with the van der Waerden score function
        mu0 = np.zeros((N,))
        Rm = R_shape_estim_COMPLEX_mv.R_estimator_VdW_score_mv(y-mu_TY, mu0, R_TY, perturbation_par)

        # MSE mismatch on sigma
        err_rm = routines.vec(Rm-Shape_S)
        err_RM = np.outer(err_rm, C(err_rm))
        MSE_Rm = MSE_Rm + err_RM/Ns
        bias_Rm = bias_Rm + err_rm/Ns

        # R-estimator with  t_nu-distribution
        mu1 = np.zeros((N,))
        Rf = R_shape_estim_COMPLEX_mv.R_estimator_F_score_mv(y-mu_TY, mu0, R_TY, perturbation_par, nu_par)

        # MSE mismatch on sigma
        err_rf = routines.vec(Rf - Shape_S)
        err_RF = np.outer(err_rf, err_rf)
        MSE_Rf = MSE_Rf + err_RF / Ns
        bias_Rf = bias_Rf + err_rf / Ns

        
    # Semiparametric CRB
    #define CRB
    a_inv_mean = s**2*gamma((N+2*s-1)/s)/(N*b**(1/s)*gamma(N/s))
    a1 = (s-1)/(2*(N+2))
    a2 = (N + s)/(N + 1)
    e1 = np.zeros(N ** 2).reshape(N ** 2, 1)
    e1[0] = 1

    D = np.eye(N**2)-T(Shape_S).reshape(Shape_S.size,1)*T(e1)

    A=( D *np.kron(T(Shape_S),Shape_S) *H(D) )
    SCRBn = (1/(K*a2))*( D.dot(np.kron(T(Shape_S),Shape_S).dot(H(D)) ))
    matrice=np.hstack((Sigma,np.zeros([N,N]),np.zeros([N,N]),C(Sigma)))
    SCRB_mean = (1 / (a_inv_mean * K)) * matrice

    SCR_Bound[il] = np.linalg.norm(SCRBn, ord='fro')
    SCR_Bound_mean[il] = np.linalg.norm(SCRB_mean, ord='fro')

    Fro_MSE_SM[il] = np.linalg.norm(MSE_SM, ord='fro')
    Fro_MSE_mu_Ty[il] = np.linalg.norm(MSE_mu_Ty, ord='fro')
    Fro_MSE_SCM[il] = np.linalg.norm(MSE_SCM, ord='fro')
    Fro_MSE_Ty[il] = np.linalg.norm(MSE_Ty, ord='fro')
    Fro_MSE_Rm[il] = np.linalg.norm(MSE_Rm, ord='fro')
    Fro_MSE_Rf[il] = np.linalg.norm(MSE_Rf, ord='fro')

    L2_bias_SM[il] = np.linalg.norm(bias_SM)
    L2_bias_mu_Ty[il] = np.linalg.norm(bias_mu_Ty)
    L2_bias_SCM[il] = np.linalg.norm(bias_SCM)
    L2_bias_Ty[il] = np.linalg.norm(bias_Ty)
    L2_bias_Rm[il] = np.linalg.norm(bias_Rm)
    L2_bias_Rf[il] = np.linalg.norm(bias_Rf)

fig = plt.figure(1)
plt.title('Bias in L2 norm')
plt.plot(svect, L2_bias_SCM, svect, L2_bias_mu_Ty)
plt.legend(['SM','mu_TY'])
plt.ylabel('L2 norm')
plt.xlabel('Shape parameter: s')
plt.show()

fig2 = plt.figure(2)
plt.title('MSE in Frobenus norm')
plt.plot(svect, SCR_Bound_mean, svect, Fro_MSE_SM, svect, Fro_MSE_mu_Ty)
plt.legend(['SCRB','SM','mu_TY'])
plt.ylabel('Frobenius norm')
plt.xlabel('Shape parameter: s')
plt.show()

fig3 = plt.figure(3)
plt.title('Bias in L2 norm')
plt.plot(svect, L2_bias_SCM, svect, L2_bias_Ty,svect,L2_bias_Rm,svect,L2_bias_Rf)
plt.legend(['SCM','TY','Rvdw','Rf'])
plt.ylabel('L2 norm')
plt.xlabel('Shape parameter: s')
plt.show()

fig4 = plt.figure(4)
plt.title('MSE in Frobenus norm')
plt.plot(svect, SCR_Bound, svect, Fro_MSE_SCM, svect, Fro_MSE_Ty,svect,Fro_MSE_Rm,svect,Fro_MSE_Rf)
plt.legend(['CSRB','SCM','TY','Rvdw','Rf'])
plt.ylabel('Frobenius norm')
plt.xlabel('Shape parameter: s')
plt.show()

