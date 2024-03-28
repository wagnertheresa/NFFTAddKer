"""
Author: Theresa Wagner <theresa.wagner@math.tu-chemnitz.de>

Corresponding publication:
"Fast Evaluation of Additive Kernels: Feature Arrangement, Fourier Methods and Kernel Derivatives"
by T. Wagner, F. Nestler, M. Stoll (2024)
"""
####################################################################################
####################################################################################
"""
Compare the RMSE, window size and time for fitting and predicting the model
with the corresponding windows for different GSI scores, fixed Nfeat=d and fixed dmax=3.
"""
import numpy as np
import pandas as pd
import itertools

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from gsi import GSI
from FE_nfft_kernel_ridge import GridSearch, NFFTKernelRidgeFE

#####################
## CHOOSE PARAMETERS
# number of train and test data included
Ndata = [1000]
# number of data points for feature engineering
Nfg = 1000
# kernel parameter ell
ell = 1
# regularization parameter beta
beta = 1
# type of prediction task
pred_type = "regression"
# kernel type
kernel = "gaussian"

## GridSearch with NFFTKernelRidge
param_grid_nfft = {
    "sigma": [0.01, 0.1, 1, 10, 100],
    "beta": [0.01, 0.1, 1, 10, 100],
}

## GridSearch with sklearn KRR
param_grid_krr = {
    "alpha": [0.01, 0.1, 1, 10, 100],
    "gamma": [1/((0.01)**2), 1/((0.1)**2), 1/((1)**2), 1/((10)**2), 1/((100)**2)],
}
#####################
# define list of gsi scores
gsi_scores = [0.5, 0.6, 0.7, 0.8, 0.9, 0.925, 0.95, 0.975, 0.98, 0.985, 0.99, 0.995, 0.999]
#####################

# choose data set
data = "keggundir"
#data = "protein"
#####################

# initialize keys for dicts
keys = gsi_scores
# initialize dicts with results, windows and time for running model with corresponding windows
fg_dict_res = {k: [] for k in keys}
fg_dict_wind = {k: [] for k in keys}
fg_dict_time_fit = {k: [] for k in keys}
fg_dict_time_pred = {k: [] for k in keys}
fg_dict_res["sklearnKRR"] = []
fg_dict_time_fit["sklearnKRR"] = []
fg_dict_time_pred["sklearnKRR"] = []
# initialize dict for saving best fg method and corresponding results
fg_best = {}
    
#####################################################################################

#######################
    
for N in Ndata:
    print("Solving for N =", N)    

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
        
        # split data in train and test split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.5, random_state=42)
    
        X_train = X_train[:N,:]
        Y_train = Y_train[:N]
        X_test = X_test[:N,:]
        Y_test = Y_test[:N]
        
        ###################
        # https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html
        # Transform outputs -> use mean of Y_train only to prevent train-test-contamination
        Y_train = np.log1p(Y_train)
        Y_train_mean = np.mean(Y_train, axis=0)
        Ytrain = Y_train - Y_train_mean
        Y_test = np.log1p(Y_test)
        Ytest = Y_test - Y_train_mean
        ###################
    
    elif data == "protein": # 9 features
        
        # https://archive.ics.uci.edu/dataset/265/physicochemical+properties+of+protein+tertiary+structure
        
        # read dataset
        df = pd.read_csv('/data/protein.csv', sep=",", header=0)
        
        X = df.iloc[:,1:]
        y = df.iloc[:,0]
        
        X = X.to_numpy()
        Y = y.to_numpy()
      
        # split data in train and test split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.5, random_state=42)
        
        X_train = X_train[:N,:]
        Y_train = Y_train[:N]
        X_test = X_test[:N,:]
        Y_test = Y_test[:N]
        
        ###################
        # https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html
        # Transform outputs -> use mean of Y_train only to prevent train-test-contamination
        Y_train = np.log1p(Y_train)
        Y_train_mean = np.mean(Y_train, axis=0)
        Ytrain = Y_train - Y_train_mean
        Y_test = np.log1p(Y_test)
        Ytest = Y_test - Y_train_mean
        ###################
        
    print("\nDataset:", data)
    print("--------------\nShape train data:", X_train.shape)
    print("Shape test data:", X_test.shape)
    
        ###################
        
    ####################################################################################
        
    #########
    # Z-score normalize data
    scaler = StandardScaler()
    X_fit = scaler.fit(X_train)
    
    X_train = X_fit.transform(X_train)
    X_test = X_fit.transform(X_test)
    
    ######################################################################################
    ######################################################################################
    ## determine the feature windows for the corresponding GSI scores
    
    # initialize wind0 and determine alpha0 for gsi model
    wind0 = list(itertools.combinations(list(range(X_train.shape[1])), 2))
    
    clf = NFFTKernelRidgeFE(sigma=1, beta=1, pred_type=pred_type)
    alpha0 = clf.train_NFFT_KRR(X_train[:Nfg,:], Ytrain[:Nfg], wind0)
    
    # initialize gsi object
    gsi = GSI(X_train[:Nfg,:], ell=1, windows=wind0)

    gsi_sorted = gsi.gsi_sorted(alpha0)
    print("gsi_sorted:", gsi_sorted)
    gsi_keys = list(gsi_sorted.keys())
    print("gsi_keys:", gsi_keys)
    
    # initialize gsi sum
    gsi_sum = 0
    # intialize dict for gsi-based windows according to gsi-scores with None entries
    #gsi_wind = dict(zip(gsi_scores, [None]*len(gsi_scores)))
    gsi_wind = {}
    print("gsi_wind:", gsi_wind)
    # intialize list for subsets in gsi_based windows
    gsi_windl = []
    cnt = 0
    for l in range(len(gsi_scores)-1):
        while (gsi_sum + gsi_sorted[gsi_keys[cnt]]) < gsi_scores[l]:
            gsi_windl.append(gsi_keys[cnt])
            gsi_sum += gsi_sorted[gsi_keys[cnt]]
            cnt += 1
        gsi_wind[gsi_scores[l]] = gsi_windl[:] # make copy of gsi_windl to not change all dict values
    # add wind for gsi-score = 1.0 (equals all subsets) by hand
    gsi_wind[gsi_scores[-1]] = gsi_keys
    print("gsi_wind:", gsi_wind)
    
    ######################################################################################
    print("\nGridSearch for NFFTKernelRidge:")
    print("-----------------------------------")
    
    # initialize list with results yielded with fg methods
    fg_res = []
    # initialize list with windows yielded by fg methods
    fg_wind = []
    # initialize list with time needed for determining windows
    fg_time = []
    
    ###################################################################################### 
  
    # iterate over gsi-scores
    for l in range(len(gsi_wind)):
        score = gsi_scores[l]
        print("\nSolving for gsi_score = ", score)
        wind = gsi_wind[score]
        # initialize NFFTKernelRidge model
        model = GridSearch(classifier="NFFTKernelRidge", param_grid=param_grid_nfft, balance=False, norm=None, pred_type=pred_type, pre_wind=wind, kernel=kernel)
        
        results = model.tune(X_train, Ytrain, X_test, Ytest)
    
        print("\n########################")
        print("FG via gsi-score: %f" %score)
        print("Windows:", results[7])
        print("Time wind:", results[8])
        print("Best Parameters:", results[0])
        print("Best Result:", results[1])
        print("Best Runtime Fit:", results[2])
        print("Best Runtime Predict:", results[3])
        print("Mean Runtime Fit:", results[4])
        print("Mean Runtime Predict:", results[5])
        print("Mean Total Runtime:", results[6])
        print("########################\n")
        fg_dict_res[score].append(results[1])
        fg_dict_wind[score].append(results[7])
        fg_dict_time_fit[score].append(results[2])
        fg_dict_time_pred[score].append(results[3])
        fg_res.append(results[1])
        fg_wind.append(results[7])
        fg_time.append(results[2]+results[3])
        
    ######################################################################################
    # full kernel KRR with sklearn KRR
    model = GridSearch(classifier="sklearn KRR", param_grid=param_grid_krr, balance=False, norm=None, pred_type=pred_type, kernel=kernel)
    results = model.tune(X_train, Ytrain, X_test, Ytest)
    
    print("\n########################")
    print("\nGridSearch for sklearn KRR:")
    print("Best Parameters:", results[0])
    print("Best Result:", results[1])
    print("Best Runtime Fit:", results[2])
    print("Best Runtime Predict:", results[3])
    print("Mean Runtime Fit:", results[4])
    print("Mean Runtime Predict:", results[5])
    print("Mean Total Runtime:", results[6])
    print("########################\n")
    fg_dict_res["sklearnKRR"].append(results[1])
    fg_dict_time_fit["sklearnKRR"].append(results[2])
    fg_dict_time_pred["sklearnKRR"].append(results[3])
    fg_res.append(results[1])
    fg_time.append(results[2]+results[3])
    
    ######################################################################################
    ######################################################################################
    print("\nResults for N=", N)
    print("-----------------\n")
    print("fg_res:", fg_res)
    print("fg_wind:", fg_wind)
    print("fg_time:", fg_time)
    
    best_idx = np.argmin(fg_res)
    best_res = np.min(fg_res)
    best_wind = fg_wind[best_idx]
    best_time = fg_time[best_idx]
    best = list(fg_dict_res.keys())[best_idx]
    if best in list(fg_dict_wind.keys()):
        print("best method: %s" %best, best_res, best_wind)
    else:
        print("best method: %s" %best, best_res)
    
    # save best method and corresponding results for N
    fg_best[N] = [best, best_res, best_wind, best_time]
    
    #######################################################################################
    ######################################################################################

print("\nRESULTS:")
print("----------\n")
print("data:", data)
print("fg_dict_res:", fg_dict_res)
print("fg_dict_wind:", fg_dict_wind)
print("fg_dict_time_fit:", fg_dict_time_fit)
print("fg_dict_time_pred:", fg_dict_time_pred)

print("\nfg_best:", fg_best)

