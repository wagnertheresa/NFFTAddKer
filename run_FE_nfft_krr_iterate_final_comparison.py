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
with the corresponding windows for the superior feature importance ranking based techniques
with FGO, GSI, sklearn KRR and sklearn SVR, with fixed dmax=3, and fixed Nfeat and L1_reg
individually for the particular data set.
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

# data sets
data_sets = ["protein", "keggundir", "bike", "housing"]

#N = 10000
N_KRR = 10000

for data in data_sets:
    print("############################")
    print("Solving for data = ", data)
    print("############################")

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
#         X_train = X_train[:N,:]
#         Y_train = Y_train[:N]
#         X_test = X_test[:N,:]
#         Y_test = Y_test[:N]
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
#         X_train = X_train[:N,:]
#         Y_train = Y_train[:N]
#         X_test = X_test[:N,:]
#         Y_test = Y_test[:N]
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
        
    elif data == "bike": # 14 features
        
        # https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
        
        # read dataset
        df = pd.read_csv('/data/bike.csv', sep=",", header=0)
        
        X = df.iloc[:,2:-1]
        y = df.iloc[:,-1]
        
        X = X.to_numpy()
        Y = y.to_numpy()
      
        # split data in train and test split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.5, random_state=42)
            
# =============================================================================
#         X_train = X_train[:N,:]
#         Y_train = Y_train[:N]
#         X_test = X_test[:N,:]
#         Y_test = Y_test[:N]
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
        
    elif data == "housing": # 8 features
        
        # http://lib.stat.cmu.edu/datasets/ --> houses.zip
    
        # read dataset
        df = pd.read_csv('/data/housing.txt', sep="\s+", header=None)
        
        X = df.iloc[:,1:]
        y = df.iloc[:,0]
        
        X = X.to_numpy()
        Y = y.to_numpy()
    
        # split data in train and test split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.5, random_state=42)
            
# =============================================================================
#         X_train = X_train[:N,:]
#         Y_train = Y_train[:N]
#         X_test = X_test[:N,:]
#         Y_test = Y_test[:N]
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
    
    # set dmax
    dmax = 3
    # set Nfeat: for large feature space 2d/3, d else
    # set L1 regularization parameter for FGO
    if data == "keggundir":
        Nfeat = 18
        L1_reg = 0.5
    elif data == "protein":
        Nfeat = 9
        L1_reg = 2.5
    elif data == "bike":
        Nfeat = 9
        L1_reg = 1.0
    elif data == "housing":
        Nfeat = 8
        L1_reg = 1.5
    
    # feature grouping methods
    fg_consec = ["elastic_net"]
    fg_distr = ["mis"]
    fg_direct = ["lasso"]
    
    # other feature grouping methods
    fg_other = ["fg_opt", "gsi"]
    
    # initialize keys for dicts
    keys = [i+"_consec" for i in fg_consec] + [i+"_distr" for i in fg_distr] + [i+"_direct" for i in fg_direct] + fg_other
    print("keys:", keys)
    # initialize dicts with results, windows and time for determining windows, and total time
    fg_dict_res = {k: [] for k in keys}
    fg_dict_wind = {k: [] for k in keys}
    fg_dict_time = {k: [] for k in keys}
    fg_dict_total_time = {k: [] for k in keys}
    # initialize dict for saving best fg method and corresponding results
    fg_best = {}
        
    print("############################")
    print("Solving for Nfeat = ", Nfeat, ", dmax = ", dmax)
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
    model = GridSearch(classifier="NFFTKernelRidge", param_grid=param_grid_nfft, balance=False, norm=None, Nfg=1000, threshold=0.0, Nfeat=Nfeat, dmax=dmax, L1_reg=L1_reg, pred_type=pred_type, kernel=kernel)
    
    # loop over feature grouping methods based on importance scores for fgmode="consec"
    for method in fg_consec:
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
        fg_dict_res[method+"_consec"] = results[1]
        fg_dict_wind[method+"_consec"] = results[7]
        fg_dict_time[method+"_consec"] = results[8]
        fg_dict_total_time[method+"_consec"] = results[6]
        
        fg_res.append(results[1])
        fg_wind.append(results[7])
        fg_time.append(results[8])
        fg_total_time.append(results[6])
        
    # loop over feature grouping methods based on importance scores for fgmode="distr"
    for method in fg_distr:
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
        fg_dict_res[method+"_distr"] = results[1]
        fg_dict_wind[method+"_distr"] = results[7]
        fg_dict_time[method+"_distr"] = results[8]
        fg_dict_total_time[method+"_distr"] = results[6]
        
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
        fg_dict_res[method+"_direct"] = results[1]
        fg_dict_wind[method+"_direct"] = results[7]
        fg_dict_time[method+"_direct"] = results[8]
        fg_dict_total_time[method+"_direct"] = results[6]
        
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
        fg_dict_res[method] = results[1]
        fg_dict_wind[method] = results[7]
        fg_dict_time[method] = results[8]
        fg_dict_total_time[method] = results[6]
        fg_res.append(results[1])
        fg_wind.append(results[7])
        fg_time.append(results[8])
        fg_total_time.append(results[6])
        
    #####################################################################################
    # full kernel KRR with sklearn KRR
    model = GridSearch(classifier="sklearn KRR", param_grid=param_grid_krr, balance=False, norm=None, pred_type=pred_type, kernel=kernel)
    results = model.tune(X_train[:N_KRR,:], Ytrain[:N_KRR], X_test[:N_KRR,:], Ytest[:N_KRR])
    
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
    fg_dict_res["sklearnKRR"] = results[1]
    fg_dict_total_time["sklearnKRR"] = results[6]
    fg_res.append(results[1])
    fg_total_time.append(results[6])
    
    #####################################################################################
    # full kernel SVM with sklearn SVR
    # (note that SVR does not allow for the "matern" or laplacian kernel)
    model = GridSearch(classifier="sklearn SVC", param_grid=param_grid_krr, balance=False, norm=None, pred_type=pred_type, kernel=kernel)
    results = model.tune(X_train, Ytrain, X_test, Ytest)
    
    print("\n########################")
    print("\nGridSearch for sklearn SVR:")
    print("Best Parameters:", results[0])
    print("Best Result:", results[1])
    print("Best Runtime Fit:", results[2])
    print("Best Runtime Predict:", results[3])
    print("Mean Runtime Fit:", results[4])
    print("Mean Runtime Predict:", results[5])
    print("Mean Total Runtime:", results[6])
    print("########################\n")
    fg_dict_res["sklearnSVR"] = results[1]
    fg_dict_total_time["sklearnSVR"] = results[6]
    fg_res.append(results[1])
    fg_total_time.append(results[6])
    
    ######################################################################################
    ######################################################################################
    # save results
    if data == "protein":
        prot_rmse = fg_dict_res
        prot_wind = fg_dict_wind
        prot_time = fg_dict_time
        prot_total_time = fg_dict_total_time
    elif data == "keggundir":
        kegg_rmse = fg_dict_res
        kegg_wind = fg_dict_wind
        kegg_time = fg_dict_time
        kegg_total_time = fg_dict_total_time
    elif data == "bike":
        bike_rmse = fg_dict_res
        bike_wind = fg_dict_wind
        bike_time = fg_dict_time
        bike_total_time = fg_dict_total_time
    elif data == "housing":
        housing_rmse = fg_dict_res
        housing_wind = fg_dict_wind
        housing_time = fg_dict_time
        housing_total_time = fg_dict_total_time
    
    print("\nResults:")
    print("-----------------\n")
    print("fg_wind:", fg_wind)
    print("fg_res:", fg_res)
    print("fg_time:", fg_time)
    print("fg_total_time:", fg_total_time)
    
    best_idx = np.argmin(fg_res)
    best_res = np.min(fg_res)
    best_total_time = fg_total_time[best_idx]
    best = list(fg_dict_res.keys())[best_idx]
    if best in list(fg_dict_wind.keys()):
        best_wind = fg_wind[best_idx]
        best_time = fg_time[best_idx]
        print("best method: %s" %best, best_res, best_wind)
        # save best method and corresponding results for N
        fg_best = [best, best_res, best_wind, best_time, best_total_time]
    else:
        print("best method: %s" %best, best_res)
        # save best method and corresponding results for N
        fg_best = [best, best_res, best_total_time]
                
    #####################################################################################
    print("\nRESULTS:")
    print("----------\n")
    print("data:", data)
    print("fg_dict_res:", fg_dict_res)
    print("fg_dict_wind:", fg_dict_wind)
    print("fg_dict_time:", fg_dict_time)
    print("fg_dict_total_time:", fg_dict_total_time)
    
    print("\nfg_best:", fg_best)

#########################################################################################
#########################################################################################
print("\nRESULTS:")
print("----------\n")
print("\ndata = Protein")
print("prot_rmse = ", prot_rmse)
print("prot_wind = ", prot_wind)
print("prot_time = ", prot_time)
print("prot_total_time = ", prot_total_time)

print("\ndata = KEGGundir")
print("kegg_rmse = ", kegg_rmse)
print("kegg_wind = ", kegg_wind)
print("kegg_time = ", kegg_time)
print("kegg_total_time = ", kegg_total_time)

print("\ndata = Bike Sharing")
print("bike_rmse = ", bike_rmse)
print("bike_wind = ", bike_wind)
print("bike_time = ", bike_time)
print("bike_total_time = ", bike_total_time)

print("\ndata = Housing")
print("housing_rmse = ", housing_rmse)
print("housing_wind = ", housing_wind)
print("housing_time = ", housing_time)
print("housing_total_time = ", housing_total_time)
    
    