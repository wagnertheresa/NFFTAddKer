"""
Author: Theresa Wagner <theresa.wagner@math.tu-chemnitz.de>

Corresponding publication:
"Fast Evaluation of Additive Kernels: Feature Arrangement, Fourier Methods and Kernel Derivatives"
by T. Wagner, F. Nestler, M. Stoll (2024)
"""
####################################################################################
####################################################################################
"""
Compute the NFFT approximation error for multiplying the additive kernel
with a one vector for different values of ell,
where the feature windows are determined consecutively via MIS.
"""
import prescaledfastadj

import numpy as np
import pandas as pd
import time

from feature_engineering import feature_grouping

from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

#####################
## CHOOSE PARAMETERS

# number of train and test data included
Ndata = [500, 1000, 5000, 10000] 
# number of data points for feature grouping
Nfg = 1000

fastadj_setup = ["fine", "default", "rough"]

ell = [0.01, 0.1, 1, 10, 100]

dmax = 3

#####################
# choose data set
data = "keggundir"
#data = "protein"

# type of prediction task
pred_type = "regression"
#####################

# initialize dict for error and relative error and runtime
dict_err = {k: [] for k in fastadj_setup}
dict_rel_err = {k: [] for k in fastadj_setup}
dict_time_std = {k: [] for k in fastadj_setup}
dict_time_nfft = {k: [] for k in fastadj_setup}

####################################################################################
# compute kernel matrix traditionally
def kermat(X, ell):
    """
    Compute the Gaussian kernel matrix.
    
    Parameters
    ----------
    X : ndarray
        The data matrix.
    ell : float, default=1.0
        The length-scale parameter.

    Returns
    -------
    K : ndarray
        The generated kernel matrix.
    """
    pairwise_dists = [squareform(pdist(X[:,wind[i]], 'euclidean')) for i in range(len(wind))]
    K = [np.exp(- (pairwise_dists[i] ** 2) /(2* ell ** 2)) for i in range(len(wind))]
    K = (weight**2) * np.sum(K, axis=0)
    
    return K

#####################################################################################

#######################
    
for N in Ndata:
    print("\nSolving for N =", N)    

    if data == "keggundir": # 26 features
        
        # https://archive.ics.uci.edu/ml/datasets/KEGG+Metabolic+Reaction+Network+(Undirected)
        
        # read dataset
        df = pd.read_csv('/data/KEGGUndir.txt', sep=",", header=None)
        
        df.drop(df[df[4] == "?"].index, inplace=True)
        df[4] = df[4].astype(float)
        df.drop(df[df[21] > 1].index, inplace=True)
        df.drop(columns=[10], inplace=True)
        
        X = df.iloc[:,1:-1]
        y = df.iloc[:,-1]
    
        X = X.to_numpy()
        Y = y.to_numpy()
    
        X = X[:N,:]
        Y = Y[:N]
        
        ###################
        # https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html
        # Transform outputs -> use mean of Y_train only to prevent train-test-contamination
        Y = np.log1p(Y)
        Y_mean = np.mean(Y, axis=0)
        Y = Y - Y_mean
        ###################
        
        ###################
    
    elif data == "protein": # 9 features
        
        # https://archive.ics.uci.edu/dataset/265/physicochemical+properties+of+protein+tertiary+structure
        
        # read dataset
        df = pd.read_csv('/data/protein.csv', sep=",", header=0)
        
        X = df.iloc[:,1:]
        y = df.iloc[:,0]
        
        X = X.to_numpy()
        Y = y.to_numpy()
        
        X = X[:N,:]
        Y = Y[:N]
        
        ###################
        # https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html
        # Transform outputs -> use mean of Y_train only to prevent train-test-contamination
        Y = np.log1p(Y)
        Y_mean = np.mean(Y, axis=0)
        Y = Y - Y_mean
        ###################
        
        ###################
        
    print("\nDataset:", data)
    print("--------------\nShape data:", X.shape)
    
        ###################
        
    ####################################################################################
        
    #########
    # Z-score normalize data
    scaler = StandardScaler()
    X_fit = scaler.fit(X)
    
    X = X_fit.transform(X)
    
    #########
    
    # prescale data points and sigma for usage of prescaledfastadj
    
    # scale data points equally
    points_center = np.mean(X, axis=0)
    points = X - points_center
    
    # scale features such that abs(x[:,j]) <= 0.25
    # scale values in range [-0.25, 0.25]
    for j in range(X.shape[1]):
        m = np.max(np.abs(points[:,j]))
        points[:,j] = points[:,j] / m * 0.25
        
    # compute maximum radius possible in dmax dimensions
    scaling = np.sqrt(dmax)
    # ensure max radius 0.25 for points
    points = points / scaling
    # scale ell accordingly
    scaled_ell = ell / scaling

    #########
    
    # determine windows with MIS
    # initialize feature-grouping class object
    fgm = feature_grouping(X[:Nfg,:], Y[:Nfg], dmax, pred_type)
    wind = fgm.mis(mode="consec")
    weight = np.sqrt(1/len(wind))
    print("windows:", wind)
    print("weight:", weight)
    
    ######################################################################################
    ######################################################################################
    
    # define vector kernel matrix shall be multiplied with
    vec = np.ones((X.shape[0]))
    
    for setup in fastadj_setup:
        print("\nSolving for setup =", setup)
        err = []
        rel_err = []
        time_std = []
        time_nfft = []
        
        for l in scaled_ell:
            print("\nSolving for scaled_ell =", l)

            ###########################
            # Standard approach
            print('run std approach!')
            start_std = time.time()
            K = kermat(points, ell=l)
            Kvec_std = K@vec
            time_std.append(time.time() - start_std)
            #print("Kvec_std:", Kvec_std)
            
            ###########################
            # NFFT approach
            print('run NFFT approach!')
            start_nfft = time.time()
            adj_mats1 = [prescaledfastadj.AdjacencyMatrix(points[:,wind[i]], np.sqrt(2)*l, kernel=1, setup=setup, diagonal=1.0) for i in range(len(wind))]
            Kvec_nfft = [adj_mats1[i].apply(vec) for i in range(len(wind))]
            Kvec_nfft = (weight**2) * np.sum(Kvec_nfft, axis=0)
            time_nfft.append(time.time() - start_nfft)
            #print("Kvec_nfft:", Kvec_nfft)
            
            ###########################
            # compute error and relative error
            err.append(np.linalg.norm(Kvec_std - Kvec_nfft))
            rel_err.append(np.linalg.norm(Kvec_std - Kvec_nfft)/np.linalg.norm(Kvec_std))
            print("error:", err)
            print("relative error:", rel_err)
            print("time_std:", time_std)
            print("time_nfft:", time_nfft)
        
        dict_err[setup].append(err)
        dict_rel_err[setup].append(rel_err)
        dict_time_std[setup].append(time_std)
        dict_time_nfft[setup].append(time_nfft)
        

print("\nRESULTS:")
print("----------\n")
print("data:", data)
print("dict_err:", dict_err)
print("dict_rel_err:", dict_rel_err)
print("dict_time_std:", dict_time_std)
print("dict_time_nfft:", dict_time_nfft)
