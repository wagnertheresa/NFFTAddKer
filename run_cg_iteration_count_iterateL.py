"""
Author: Theresa Wagner <theresa.wagner@math.tu-chemnitz.de>

Corresponding publication:
"Fast Evaluation of Additive Kernels: Feature Arrangement, Fourier Methods and Kernel Derivatives"
by T. Wagner, F. Nestler, M. Stoll (2024)
"""
####################################################################################
####################################################################################
"""
Compare the eigenvalue decay and number of CG iterations for different values ell,
fixed tol=1e-3, fixed N=1000, fixed beta=0.0001, where the windows are determined
consecutively via MIS ranking.
"""
import numpy as np
import pandas as pd
import time

from feature_engineering import feature_grouping

from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from scipy.sparse.linalg import cg

#####################
## CHOOSE PARAMETERS

# number of data included
N = 1000 
# number of data points for feature selection
Nfg = 1000

#ell = 100
ell = [0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 25, 50, 75, 100]
# we choose weight = 1 (does not matter for error)
sf = 1
# regularization parameter
beta = 0.0001

dmax = 3

#####################
# choose data set
datasets = ["keggundir", "protein"]

# type of prediction task
pred_type = "regression"
#####################

####################################################################################
# compute kernel matrix traditionally
def kermat(X, ell):
    """
    Compute the Gaussian kernel matrix.
    
    Parameters
    ----------
    X : ndarray
        The data matrix.
    ell : float
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
# compute number of cg iterations
def cg_iterations(K, Y):
    """
    Return the number of CG iterations for solving the system: K @ alpha = Y.
    
    Parameters
    ----------
    K : ndarray
        The kernel matrix.
    Y : ndarray
        The vector on the right-hand side of the linear equation to be solved.

    Returns
    -------
    num_iters : int
        Number of CG iterations required to reach the relative residual tolerance 1e-3.
    """
    # initialize counter to get number of iterations needed in cg-algorithm
    num_iters = 0

    # function to count number of iterations needed in cg-algorithm
    def callback(xk):
        nonlocal num_iters
        num_iters += 1
    
    alpha, info = cg(K, Y, tol=1e-4, callback=callback)
    
    return num_iters

#####################################################################################

#######################
for data in datasets:

    print("\nSolving for data = ", data)
    print("\nSolving for N = ", N)    
    
    if data == "keggundir": # 26 features

        # https://archive.ics.uci.edu/ml/datasets/KEGG+Metabolic+Reaction+Network+(Undirected)
        
        # read dataset
        df = pd.read_csv('/data/KEGGUndir.txt', sep=",", header=None)
        
        # same preprocessing as Jonathan Wenger in "Posterior and Computational Uncertainty in Gaussian Processes"
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
    
    # determine windows with MIS
    # initialize feature-grouping class object
    fgm = feature_grouping(X[:Nfg,:], Y[:Nfg], dmax, pred_type)
    wind = fgm.mis(mode="consec")
    weight = np.sqrt(1/len(wind))
    print("windows:", wind)
    print("weight:", weight)

######################################################################################
######################################################################################
    
    # initialize dict for cg iteration count
    cg_iters_std = []
    cg_time_std = []
    eigs_std = []
    
    for l in ell:
        print("\nSolving for ell =", l)
    
        ###########################
        # Standard approach
        print('run std approach!')
        
        start_std = time.time()
        
        K = kermat(X, ell=l) + beta * np.eye(N)
        num_iters_std = cg_iterations(K, Y)
        
        cg_iters_std.append(num_iters_std)
        cg_time_std.append(time.time() - start_std)
        
        eigs = (np.linalg.eig(K)[0]).real
        sorted_eigs = (np.sort(eigs)[::-1]).tolist()
        eigs_std.append(sorted_eigs)
          
        print("cg_iters_std:", cg_iters_std)
        print("cg_time_std:", cg_time_std) 
        #print("cg_eigs_std:", cg_eigs_std) 
        
    if data == "keggundir":
        kegg_cg_iters = cg_iters_std
        kegg_cg_time = cg_time_std
        kegg_eigs = eigs_std
        #print("len kegg_eigs:", len(kegg_eigs))
    elif data == "protein":
        prot_cg_iters = cg_iters_std
        prot_cg_time = cg_time_std
        prot_eigs = eigs_std
        #print("len prot_eigs:", len(prot_eigs))
        
print("\nRESULTS:")
print("----------\n")
print("data:", data)
print("cg_iters_std:", cg_iters_std)
print("cg_time_std:", cg_time_std)
#print("eigs_std:", eigs_std)

#########################################################################################
# PLOT

import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter

from matplotlib import font_manager as fm, rcParams

# set font size
fs = 80

# use a font with serifs
plt.rc( 'font', size=fs, family="serif" )   

# activate LaTeX text rendering
plt.rc( 'text', usetex=True ) 

# use bold font
plt.rc('font', weight='bold')

# all mathematical expressions shall be bold
#plt.rcParams['text.latex.preamble'] = [r'\boldmath']

#####################################################################################
    
# set up figure
rows = 1
cols = 4
#fig, axs = plt.subplots(rows, cols, figsize=(70,20))
fig, axs = plt.subplots(rows, cols, figsize=(65,25))
#fig, axs = plt.subplots(rows, cols, figsize=(0,5))

##########################
## log transform axes

# some axes are logarithmic: x-axis for length-scale and y-axis for eigenvalues
axs[1].set_xscale('log')
axs[3].set_xscale('log')
axs[0].set_yscale('log')
axs[2].set_yscale('log')
# change the formatter since a logarithmic scale uses a LogFormatter
axs[1].xaxis.set_major_formatter(LogFormatter)
axs[3].xaxis.set_major_formatter(LogFormatter)
axs[0].yaxis.set_major_formatter(LogFormatter)
axs[2].yaxis.set_major_formatter(LogFormatter)
# set range for axes
axs[1].set_xlim(8*1e-3, 1.5*1e+2)
axs[3].set_xlim(8*1e-3, 1.5*1e+2)
axs[0].set_ylim(8*1e-5, 2*1e+3)
axs[2].set_ylim(8*1e-5, 2*1e+3)

##########################
##########################
# PROTEIN
##########################
##########################
## plot eigenvalues

idx = list(range(N))

for i in range(len(ell)):
    #print("len prot_eigs[i]:", len(prot_eigs[i]))
    axs[0].semilogy(idx, prot_eigs[i], linewidth=2.0)
#axs[0].set_title(r'\textbf{Protein}', fontsize=fs+10)
axs[0].text(1150, 1e+4, r'\textbf{Protein}', size=fs+10, horizontalalignment='center', verticalalignment='center')
axs[0].set_title(r'\textbf{eigenvalue}', fontsize=fs)
axs[0].set_xlabel(r'\textbf{index}', fontsize=fs)

##########################
## plot cg iterations

axs[1].semilogx(ell, prot_cg_iters, linewidth=2.0)
axs[1].set_title(r'\textbf{cg iteration count}', fontsize=fs)
axs[1].set_xlabel(r'\textbf{length-scale}', fontsize=fs)
axs[1].set_xticks([1e-2, 1e-1, 1, 1e+1, 1e+2])
#axs[1].set_xticklabels(labels=['$1e-2$','$1e-1$','$1$','$1e+1$','$1e+2$'])

##########################
##########################
# KEGGUNDIR
##########################
##########################
## plot eigenvalues

for i in range(len(ell)):
    axs[2].semilogy(idx, kegg_eigs[i], linewidth=2.0)
#axs[2].set_title(r'\textbf{KEGGundir}', fontsize=fs+10)
axs[2].text(1150, 1e+4, r'\textbf{KEGGundir}', size=fs+10, horizontalalignment='center', verticalalignment='center')
axs[2].set_title(r'\textbf{eigenvalue}', fontsize=fs)
axs[2].set_xlabel(r'\textbf{index}', fontsize=fs)

##########################
## plot cg iterations

axs[3].semilogx(ell, kegg_cg_iters, linewidth=2.0)
axs[3].set_title(r'\textbf{cg iteration count}', fontsize=fs)
axs[3].set_xlabel(r'\textbf{length-scale}', fontsize=fs)
axs[3].set_xticks([1e-2, 1e-1, 1, 1e+1, 1e+2])
#axs[3].set_xticklabels(labels=['$1e-2$','$1e-1$','$1$','$1e+1$','$1e+2$'])

####################################################
## Create figure

plt.rcParams.update({"font.size": fs, "font.weight": "bold"})

fig = plt.gcf()

#plt.show()
fig_name = 'plot_cg_iteration_eigs.pdf'
plt.savefig(fig_name, dpi=300)
