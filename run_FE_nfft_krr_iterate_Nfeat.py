"""
Author: Theresa Wagner <theresa.wagner@math.tu-chemnitz.de>

Corresponding publication:
"Fast Evaluation of Additive Kernels: Feature Arrangement, Fourier Methods and Kernel Derivatives"
by T. Wagner, F. Nestler, M. Stoll (2024)
"""
####################################################################################
####################################################################################
"""
Compare the RMSE, window setup time and time for fitting and predicting the model
with the corresponding windows for different feature arrangement techniques and strategies,
different values Nfeat and fixed dmax=3.
"""
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from FE_nfft_kernel_ridge import GridSearch

#####################
## CHOOSE PARAMETERS
# kernel parameter ell
ell = 1
# regularization parameter beta
beta = 1
###############################
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
    "gamma": [1/((0.01)**2), 1/((0.1)**2), 1/((1)**2), 1/((10)**2), 1/((100)**2)],
    "alpha": [0.01, 0.1, 1, 10, 100],
}
#####################
#####################

# choose data set
#data = "keggundir"
data = "protein"

#N = 1000

#######################  

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
    
# =============================================================================
#     X_train = X_train[:N,:]
#     Y_train = Y_train[:N]
#     X_test = X_test[:N,:]
#     Y_test = Y_test[:N]
# =============================================================================
        
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
        
# =============================================================================
#     X_train = X_train[:N,:]
#     Y_train = Y_train[:N]
#     X_test = X_test[:N,:]
#     Y_test = Y_test[:N]
# =============================================================================
        
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
    
# choose dmax
dmax = 3
# iterate over Nfeat
if data == "keggundir":
    Nfeat = [9, 18, 26]
elif data == "protein":
    Nfeat = [3, 6, 9]

# feature grouping methods
fg_consecdistr = ["dt", "mis", "fisher", "reliefFfilt", "reliefFwrap", "lasso", "elastic_net", "fc_cc"]
fg_direct = ["lasso", "elastic_net"]
fg_single = ["fc_cc"]

# other feature grouping methods
fg_other = ["consec"]

# initialize keys for dicts
keys = [i+"_consec" for i in fg_consecdistr] + [i+"_distr" for i in fg_consecdistr] + [i+"_direct" for i in fg_direct] + [i+"_single" for i in fg_single] + fg_other
print("keys:", keys)
# initialize dicts with results, windows and time for determining windows, and total time
fg_dict_res = {k: [] for k in keys}
fg_dict_wind = {k: [] for k in keys}
fg_dict_time = {k: [] for k in keys}
fg_dict_total_time = {k: [] for k in keys}
# initialize dict for saving best fg method and corresponding results
fg_best = {}
    
for Nf in Nfeat:
    print("############################")
    print("Solving for Nfeat = ", Nf)
    print("############################")

    ######################################################################################
    print("\nGridSearch for NFFTKernelRidge:")
    print("-----------------------------------")
    
    # initialize list with results yielded with fg methods
    fg_res = []
    # initialize list with windows yielded by fg methods
    fg_wind = []
    # initialize list with time needed for determining windows
    fg_time = []
    # initialize list with total time
    fg_total_time = []

    ###################################################################################### 
    # initialize NFFTKernelRidge model
    model = GridSearch(classifier="NFFTKernelRidge", param_grid=param_grid_nfft, balance=False, norm=None, Nfg=1000, threshold=0.0, Nfeat=Nf, dmax=dmax, pred_type=pred_type, kernel=kernel)
    
    # loop over feature grouping methods based on importance scores for fgmode="consec"
    for method in fg_consecdistr:
        results = model.tune(X_train, Ytrain, X_test, Ytest, window_scheme=method, fgmode="consec")
    
        print("\n########################")
        print("FG via %s" %method)
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
        fg_dict_res[method+"_consec"].append(results[1])
        fg_dict_wind[method+"_consec"].append(results[7])
        fg_dict_time[method+"_consec"].append(results[8])
        fg_dict_total_time[method+"_consec"].append(results[6])
        
        fg_res.append(results[1])
        fg_wind.append(results[7])
        fg_time.append(results[8])
        fg_total_time.append(results[6])
        
    # loop over feature grouping methods based on importance scores for fgmode="distr"
    for method in fg_consecdistr:
        results = model.tune(X_train, Ytrain, X_test, Ytest, window_scheme=method, fgmode="distr")
    
        print("\n########################")
        print("FG via %s" %method)
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
        fg_dict_res[method+"_distr"].append(results[1])
        fg_dict_wind[method+"_distr"].append(results[7])
        fg_dict_time[method+"_distr"].append(results[8])
        fg_dict_total_time[method+"_distr"].append(results[6])
        fg_res.append(results[1])
        fg_wind.append(results[7])
        fg_time.append(results[8])
        fg_total_time.append(results[6])
        
    # loop over feature grouping methods based on importance scores for fgmode="direct"
    for method in fg_direct:
        results = model.tune(X_train, Ytrain, X_test, Ytest, window_scheme=method, fgmode="direct")
    
        print("\n########################")
        print("FG via %s" %method)
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
        fg_dict_res[method+"_direct"].append(results[1])
        fg_dict_wind[method+"_direct"].append(results[7])
        fg_dict_time[method+"_direct"].append(results[8])
        fg_dict_total_time[method+"_direct"].append(results[6])
        fg_res.append(results[1])
        fg_wind.append(results[7])
        fg_time.append(results[8])
        fg_total_time.append(results[6])
        
    # loop over feature grouping methods based on importance scores for fgmode="single"
    for method in fg_single:
        results = model.tune(X_train, Ytrain, X_test, Ytest, window_scheme=method, fgmode="single")
    
        print("\n########################")
        print("FG via %s" %method)
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
        fg_dict_res[method+"_single"].append(results[1])
        fg_dict_wind[method+"_single"].append(results[7])
        fg_dict_time[method+"_single"].append(results[8])
        fg_dict_total_time[method+"_single"].append(results[6])
        fg_res.append(results[1])
        fg_wind.append(results[7])
        fg_time.append(results[8])
        fg_total_time.append(results[6])
            
    # loop over other feature grouping methods
    for method in fg_other:
        results = model.tune(X_train, Ytrain, X_test, Ytest, window_scheme=method)
    
        print("\n########################")
        print("FG via %s" %method)
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
        fg_dict_res[method].append(results[1])
        fg_dict_wind[method].append(results[7])
        fg_dict_time[method].append(results[8])
        fg_dict_total_time[method].append(results[6])
        fg_res.append(results[1])
        fg_wind.append(results[7])
        fg_time.append(results[8])
        fg_total_time.append(results[6])
        
    ######################################################################################
    ######################################################################################
    print("\nResults for dmax = ", dmax, ", Nfeat = ", Nf)
    print("-----------------\n")
    print("fg_wind:", fg_wind)
    print("fg_res:", fg_res)
    print("fg_time:", fg_time)
    print("fg_total_time:", fg_total_time)
    
    best_idx = np.argmin(fg_res)
    best_res = np.min(fg_res)
    best_wind = fg_wind[best_idx]
    best_time = fg_time[best_idx]
    best_total_time = fg_total_time[best_idx]
    best = list(fg_dict_res.keys())[best_idx]
    if best in list(fg_dict_wind.keys()):
        print("best method: %s" %best, best_res, best_wind)
    else:
        print("best method: %s" %best, best_res)

    # save best method and corresponding results for N
    fg_best[dmax,Nf] = [best, best_res, best_wind, best_time, best_total_time]
            
#####################################################################################
print("\nRESULTS:")
print("----------\n")
print("data:", data)
print("fg_dict_res:", fg_dict_res)
print("fg_dict_wind:", fg_dict_wind)
print("fg_dict_time:", fg_dict_time)
print("fg_dict_total_time:", fg_dict_total_time)

print("\nfg_best:", fg_best)

