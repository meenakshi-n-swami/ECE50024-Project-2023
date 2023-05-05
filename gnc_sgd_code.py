
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 18:31:01 2023

@author: Meenakshi
"""
#importing required packages
import numpy as np
import scipy.stats as stats
from scipy.special import expit
import matplotlib.pyplot as plt
import time

#function for logit regression residuals, loss and loss gradient
def residuals(weights,X,y):
    z = y*X.dot(weights)
    f = -(1 / (1 + np.exp(z))) * (y * X)
    p = expit(z)
    res = y-p
    var = (res**2).mean()
    return z,f,p,res,var

#function for updating weights in gnc
def weight_update(l,mu,res,cbar2):
    weights = np.ones((l,1))
    mult = mu*cbar2
    for i in range(l):
        weights[i] = ((mult)/(res[i]+mult))**2
    return weights

#functions for stochastic gradient descent
def sgd_logreg(X, y, lr, max_iters, batch_size=32):
    start_time = time.time()
    w = np.zeros((X.shape[1], 1))
    for i in range(max_iters):
        indices = np.random.choice(X.shape[0], batch_size, replace=False)
        X_batch = X[indices]
        y_batch = y[indices]
        z,f,p,res,var_0 = residuals(w,X_batch,y_batch)
        grad = np.mean(f, axis=0).reshape((-1, 1))
        w -= lr * grad
    end_time = time.time()
    return w,i,end_time-start_time

#function to perform graduated non convexity
def gnc_logreg(X, y, epsilon, max_iters):
    start_time = time.time()
    l = X.shape[1]
    weight_0 = np.ones((l, 1))
    mis_his = [1]
    cbar2 = epsilon
    z,f,p,res,var_0 = residuals(weight_0,X,y)
    res_max=np.max(res)
    mu = 2*res_max/cbar2
    i = 1
    while i < max_iters and mu > 1:
        weights = weight_update(l,mu,res,cbar2)
        z,f,p,res,var= residuals(weights,X,y)
        mis_his.append(np.sum(np.abs(np.diff(weights))))
        mu = mu / 1.4
        i+=1
    end_time = time.time()
    return weights,i,end_time-start_time

#function to perform newton-raphson
def nr_logreg(X, y, epsilon, max_iters):
    start_time = time.time()
    w = np.ones((X.shape[1], 1))
    for i in range(max_iters):
        z,f,p,res,var_0 = residuals(w,X,y)
        z = X.dot(w)
        p = expit(z)
        grad = np.mean(f, axis=0).reshape((-1, 1))
        diagonal = (np.diagflat(p * (1 - p)))
        hess = X.T.dot(diagonal).dot(X)
        eigenvalues = np.linalg.eigvals(hess)
        if np.min(eigenvalues) < epsilon:
            hess += np.eye(X.shape[1]) * (epsilon - np.min(eigenvalues))
        w -= np.linalg.inv(hess).dot(grad)
    end_time = time.time()
    return w,i,end_time-start_time

#function to generate problem data
def problem(N0,outlier_frac):
    mu0 = 0
    mu1 = 10
    sigma0 = 1
    sigma1 = 1
    
    N1 = N0
    N  = N0+N1

    x0 = stats.norm.rvs(mu0,sigma0,N0)
    x1 = stats.norm.rvs(mu1,sigma1,N1)
    y0 = np.zeros(N0)
    y1 = np.ones(N1)
    
    num_outliers = int(N * outlier_frac)
    outlier_x = np.hstack((stats.uniform.rvs(-20, 30, num_outliers),stats.uniform.rvs(-20, 30, num_outliers)))
    outlier_y = np.hstack((np.zeros(num_outliers),np.ones(num_outliers)))
    x0 = np.hstack((x0, outlier_x))
    y0 = np.hstack((y0, outlier_y))
    x = np.hstack((x0, x1))
    y = np.hstack((y0, y1)).reshape((N + 2*num_outliers, 1))
    X = np.vstack((x, np.ones(N + 2*num_outliers))).T
    N=X.shape[0]
    return X,y,N

#global initializations
epsilon =stats.chi2.ppf (0.99, 2) * (0.2)**2
max_iters=1000
lr=0.1

outlier_min = 10
outlier_max = 110
N0 = 3000
plots_per_row = 2
plots_per_column = 5
u,v = 0,0
fig,axs = plt.subplots(plots_per_column,plots_per_row,figsize=(70,120))

#looping through iterations of outlier ratios
for i in range(outlier_min,outlier_max,10):
    j=i/100
    #generate problem for specific outliers
    X,y,N = problem(N0,j)
    #fit logistic regression using SGD
    w_sgd,i_sgd,time_sgd = sgd_logreg(X, y, lr, max_iters, batch_size=32)
    print(f"SGD weight vector for {i}% outliers: ", w_sgd.ravel())
    print(f"SGD iterations for {i}% outliers: ", i_sgd)
    print(f"SGD time for {i}% outliers: {time_sgd:,.3f}")
    #fit logistic regression using GNC
    w_gnc,i_gnc,time_gnc = gnc_logreg(X, y, epsilon,max_iters)
    print(f"GNC weight vector for {i}% outliers: ", w_gnc.ravel())
    print(f"GNC iterations for {i}% outliers: ", i_gnc)
    print(f"GNC time for {i}% outliers: {time_gnc:,.3f}")
    #fit logistic regression using NR
    # w_nr,i_nr,time_nr= nr_logreg(X, y, epsilon,max_iters)
    # print(f"NR weight vector for {i}% outliers: ", w_nr.ravel())
    # print(f"NR iterations for {i}% outliers: ", i_nr)
    # print(f"NR time for {i}% outliers: {time_nr:,.3f}")
    
    #plotting in one figure
    x_vals = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
    X_vals = np.vstack((x_vals, np.ones(x_vals.shape[0]))).T
    #calculating fit
    y_gnc = expit(X_vals @ w_gnc)
    y_sgd = expit(X_vals @ w_sgd)
    #y_nr = expit(X_vals @ w_nr)
    #plotting data
    axs[u][v].plot(X[:, 0], y, '*', markersize = '1', label = 'Data')
    #plotting fit curves
    axs[u][v].plot(x_vals, y_gnc,linewidth=2, color='green', label="Logit Regression with GNC")
    axs[u][v].plot(x_vals, y_sgd,linewidth=2, color= 'red', label="Logit Regression with SGD")
    #axs[u][v].plot(x_vals, y_nr,linewidth=2, color= 'yellow', label="Logit Regression with NR")
    # axs[u][v].set_xlabel('X',fontsize = 20.0)
    # axs[u][v].set_ylabel('y',fontsize = 20.0)  
    axs[u][v].set_title(f'Logistic Regression fit with {i}% outliers', size=55)
    axs[u][v].tick_params(axis='both', which='major', labelsize=22)
    j+=1
    v+=1
    if v%plots_per_row==0:
        u+=1
        v=0
handles, labels = axs[1][1].get_legend_handles_labels()
fig.legend(handles, labels, loc=(0.80,0.93), fontsize = 51)
plt.suptitle('Comparing Logistic Regression Fit with GNC and SGD for Various Outlier Ratios', fontsize = 70,x=0.50,y=0.92)
plt.show()

#plot GNC SGD NR fit curve individually
plt.plot(X[:, 0], y, '*', markersize = '1', label = 'Data')
x_vals = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
X_vals = np.vstack((x_vals, np.ones(x_vals.shape[0]))).T
y_vals_gnc = expit(X_vals @ w_gnc)
y_vals_sgd = expit(X_vals @ w_sgd)
plt.plot(x_vals, y_vals_gnc,linewidth=2, color='green', label="Logit Regression with GNC")
plt.plot(x_vals, y_vals_sgd,linewidth=2, color= 'red', label="Logit Regression with SGD")
#plt.plot(x_vals, y_vals_nr,linewidth=2, color= 'yellow', label="Logit Regression with NR")
plt.xlabel('X')
plt.ylabel('y')
plt.legend(fontsize='8')
plt.title('Logistic Regression Fit for 69% Outlier Ratio')
plt.show()