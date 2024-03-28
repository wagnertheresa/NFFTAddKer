"""
Author: Theresa Wagner <theresa.wagner@math.tu-chemnitz.de>

Corresponding publication:
"Additive kernels for learning tasks"
by T. Wagner, F. Nestler, M. Stoll (2024)
"""
####################################################################################
####################################################################################

import numpy as np
import prescaledfastadj
import time
import random
import itertools

from feature_engineering import feature_grouping

from sklearn.preprocessing import StandardScaler
from scipy.sparse.linalg import cg, LinearOperator

# import sklearn classifier for comparison
from sklearn import svm
from sklearn.kernel_ridge import KernelRidge


class NFFTKernelRidgeFE:
    """
    NFFT-based Additive Kernel Ridge Regression with Feature Grouping
    
    Parameters
    ----------
    sigma : float, default=1.0
         Length-scale parameter for the Gaussian kernel.
    beta : float, default=1.0
        The regularization parameter within the learning task for the kernel ridge regression, where beta > 0.
    setup : str, default='default'
        The setup argument loads presets for the parameters of the NFFT fastsum method. It is one of the strings 'fine', 'default' or 'rough'.
    tol : float, default=1e-03
        The tolerance of convergence within the CG-algorithm.
    pred_type : str, default="regression"
        The argument determining the type of the prediction.
        If "bin_class" is passed, binary classification shall be performed.
        If "regression" is passed, regression shall be performed.
    kernel : str, default="gaussian"
        The kernel definition that shall be used.
        If "gaussian" the Gaussian kernel is used.
        If "matern" the Matérn(1/2) kernel is used.
    
    Attributes
    ----------
    trainX : ndarray
        The training data used to fit the model.
    windows : list
        The list of feature windows.
    weights : list
        The kernel weights.
    alpha : ndarray
        The dual-variable for the KRR model.
    
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import train_test_split
    >>> N, d = 25000, 15
    >>> rng = np.random.RandomState(0)
    >>> X = rng.randn(N, d)
    >>> y = np.sign(rng.randn(N))
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=42)
    >>> windows = [[0,1,2],[3,4,5],[6,7,8],[9,10,11],[12,13,14]]
    >>> weights = np.sqrt(1/len(windows)))
    >>> clf = NFFTKernelRidge
    >>> clf.fit(X_train, y_train, windows, weights)
    >>> clf.predict(X_test)
    """
    
    def __init__(self, sigma=1, beta=1, setup='default', tol=1e-03, pred_type="regression", kernel="gaussian"):
        self.sigma = sigma
        self.beta = beta
        self.setup = setup
        self.tol = tol
        self.pred_type = pred_type
        self.kernel = kernel
    
    
    def init_fast_matvec(self, trainX):
        """
        Set up computations with the adjacency matrix and create adjacency matrix object.
            
        Parameters
        ----------
        trainX : ndarray
            The data matrix.
        
        Returns
        -------
        adj_mats : object
            The adjacency matrix object.
        """
        ## setup computations with the adjacency matrices
        # set diagonal=1.0, since FastAdjacency package is targeted at graph Laplacian with zeros at the diagonal, but we need 1 at the diagonal
        if self.kernel == "gaussian":
            adj_mats = [prescaledfastadj.AdjacencyMatrix(trainX[:,self.windows[l]], np.sqrt(2)*self.sigma, kernel=1, setup=self.setup, diagonal=1.0) for l in range(len(self.windows))]
        elif self.kernel == "matern":
            adj_mats = [prescaledfastadj.AdjacencyMatrix(trainX[:,self.windows[l]], self.sigma, kernel=3, setup=self.setup, diagonal=1.0) for l in range(len(self.windows))]
        
        return adj_mats
    
    
    def fast_adjacency(self, adj_mats, p):
        """
        Approximate the matrix-vector product A*p within the CG-algorithm used in train_NFFT_KRR, where A = w_1*K_1 + w_2*K_2 + ... + beta*I.
            
        Note
        ----
        Using the NFFT-approach, the kernel matrices are never computed explicitly. 
        Since this function serves as a LinearOperator for the cg-function from scipy, only one variable, the vector p, can be passed as input parameter.
        The variables which are needed additionally to approximate A*p, are therefore defined as global variables in train_NFFT_KRR, so that they can still be used within this function.
        
        Parameters
        ----------
        adj_mats : object
            The adjacency matrix object.
        p : ndarray
            The vector, whose product A*p with the matrix A shall be approximated.
        
        Returns
        -------
        Ap : ndarray
            The approximated matrix-vector product A*p.
        """
        # perform kernel-vector multiplication including weights
        Ap_i = np.asarray([adj_mats[l].apply(p) for l in range(len(self.windows))])
        
        # sum weighted sub-kernels up
        Ap = self.weights**2* np.sum(Ap_i, axis=0)
        
        # do not neglect: A = (w_1*K_1 + w_2*K_2 + ...) + beta*I
        Ap = Ap + self.beta*p
        
        return Ap
    
    
    def train_NFFT_KRR(self, X, y, wind0=[], weighting="equally weighted"):    
        """
        Train the model on the training data by solving the underlying system of linear equations using the CG-algorithm with NFFT-approach.
            
        Parameters
        ----------
        X : ndarray
            The training data matrix.
        y : ndarray
            The corresponding target values.
        wind0 : list, default=[]
            The initial feature windows for GSI.
        weighting : str, default="equally weighted"
            The weighting scheme for determining the kernel weights.
        
        Returns
        -------
        alpha : ndarray
            The dual-variable for the KRR-Model.
        """        
        if len(wind0) > 0:
            self.windows = wind0
            if weighting == "equally weighted":
                self.weights = np.sqrt(1/len(wind0))
            elif weighting == "no weights":
                self.weights = 1
        
        # set up computations with adjacency matrix
        adj_mats = self.init_fast_matvec(X)
        
        Ap = LinearOperator(shape=(X.shape[0],X.shape[0]), matvec= lambda p: self.fast_adjacency(adj_mats, p))
    
        # initialize counter to get number of iterations needed in cg-algorithm
        num_iters = 0
    
        # function to count number of iterations needed in cg-algorithm
        def callback(xk):
            nonlocal num_iters
            num_iters += 1
            
        # CG with NFFT
        alpha, info = cg(Ap, y, tol=self.tol, callback=callback)
        
        # print number of iterations needed in cg-algorithm
        #print('num_iters in CG for fitting:', num_iters)
        
        return alpha
           
    
    def fit(self, X, y, windows, weights):
        """
        Fit NFFT-based kernel ridge regression model on training data.

        Parameters
        ----------
        X : ndarray
            The training data matrix.
        y : ndarray
            The corresponding target values.
        windows : list
            The feature windows.
        weights : float
            The kernel weights.

        Returns
        -------
        self : returns an instance of self
        """
        self.trainX = X
        
        self.windows = windows
        self.weights = weights
            
        self.alpha = self.train_NFFT_KRR(X,y)
        
        return self
    
        
    def predict(self, X):
        """
        Predict class affiliations for the test data after the model has been fitted, using the NFFT-approach.
            
        Note
        ----
        To use the NFFT-based fast summation here, the test samples are appended to the training samples.
        In total, only a fraction of the approximated kernel evaluation is needed here.
        Because of this, the factor consists of alpha and the rest is padded with zeros.
        
        Parameters
        ----------
        X : ndarray
            The data, for which class affiliations shall be predicted.
        
        Returns
        -------
        YPred : ndarray
            The predicted class affiliations.
        windows : list
            The feature windows.
        alpha : ndarray
            The dual-variable for the KRR model.
        """
        N_sum = X.shape[0] + self.trainX.shape[0]
        arr = np.append(self.trainX, X, axis=None).reshape(N_sum,self.trainX.shape[1])
    
        if self.kernel == "gaussian":
            adjacency_vals = [prescaledfastadj.AdjacencyMatrix(arr[:,self.windows[l]], np.sqrt(2)*self.sigma, kernel=1, setup=self.setup) for l in range(len(self.windows))]
        elif self.kernel == "matern":
            adjacency_vals = [prescaledfastadj.AdjacencyMatrix(arr[:,self.windows[l]], self.sigma, kernel=3, setup=self.setup) for l in range(len(self.windows))]
        
        p = np.append(self.alpha, np.zeros(X.shape[0]), axis=None).reshape(N_sum,)
    
        # predict responses
        vals = [adjacency_vals[l].apply(p) for l in range(len(self.windows))]
        vals = self.weights**2 *np.sum(vals, axis=0)
        
        # select predicted responses for test data
        if self.pred_type == "bin_class":
            YPred = np.sign(vals[-X.shape[0]:])
        elif self.pred_type == "regression":
            YPred = vals[-X.shape[0]:]
        
        return YPred, self.windows, self.alpha
    
    
    
class GridSearch:
    """
    Exhaustive search on candidate parameter values for a classifier.
    
    Parameters
    ----------
    classifier : str, default='NFFTKernelRidge'
        The classifier parameter determines, for which classifier GridSearch shall be performed.
        It is either 'NFFTKernelRidge', 'sklearn KRR' or 'sklearn SVC'.
    param_grid : dict
        Dictionary with parameter names and lists of candidate values for the parameters to try as values.
    scoring : str, default='accuracy'
        The scoring parameter determines, which evaluation metric shall be used.
        It is either 'accuracy', 'precision' or 'recall'.
    balance : bool, default=True
        Whether the class distribution of the data, the model is fitted on, shall be balanced or not.
    n_samples : int, default=None
        Number of samples to include per class, when balancing the class distribution.
        If None, then the biggest possible balanced subset, i.e. a subset with min(#samples in class -1, #samples in class 1) samples, is used.
        Else, a subset with n_samples randomly chosen samples per class is constructed.
    norm : str, default='z-score'
        The normalization parameter determining how to standardize the features. It is either 'z-score' (z-score normalization) or None (the features are not standardized).
    setup : str, default='default'
        The setup argument loads presets for the parameters of the NFFT fastsum method. It is one of the strings 'fine', 'default' or 'rough'.
    tol : float, default=1e-03
        The tolerance of convergence within the CG-algorithm.
    Nfg : int, default=1000
        The subset size to perform the feature grouping techniques on.
    threshold : float, default=0.0
        Threshold determining, which features to include in the kernel. All features with a score below this threshold are dropped, the others are included.
    pred_type : str, default="regression"
        The argument determining the type of the prediction.
        If "bin_class" is passed, binary classification shall be performed.
        If "regression" is passed, regression shall be performed.
    Nfeat : int, default=9
        The total number of features involved.
    dmax : int, default=3
        The maximal window length.
    gsi_score : float, default=0.99
        The GSI score determining which feature subsets to add to the list of windows.
    L1_reg : float, default=1.5
        L1 regularization parameter for FGO.
    pre_list : list, default=[]
        If window_scheme=None, a predefined list of windows of features must be passed. Those windows are used to realize the kernel-vector multiplication.
        If window_scheme='mis', a predefined list of windows can be passed and the rest of the windows are determined by the features' mis.
    kernel : str, default="gaussian"
        The kernel definition that shall be used.
        If "gaussian" the Gaussian kernel is used.
        If "matern" the Matérn(1/2) kernel is used.
    
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import train_test_split
    >>> N, d = 25000, 15
    >>> rng = np.random.RandomState(0)
    >>> X = rng.randn(N, d)
    >>> y = np.sign(rng.randn(N))
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=42)
    >>> param_grid = {
        "sigma": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        "beta": [1, 10, 100, 1000],
    }
    >>> model = GridSearch(classifier="NFFTKernelRidge", param_grid=param_grid)
    >>> model.tune(X_train, y_train, X_test, y_test)
    """
    
    def __init__(self, classifier, param_grid, scoring="accuracy", balance=True, n_samples=None, norm='z-score', setup="default", tol=1e-03, Nfg=1000, threshold=0.0, pred_type="regression", Nfeat=9, dmax=3, gsi_score=0.99, L1_reg=1.5, pre_wind=[], kernel="gaussian"):
        
        self.classifier = classifier
        self.param_grid = param_grid
        self.scoring = scoring
        self.balance = balance
        self.n_samples = n_samples
        self.norm = norm
        self.setup = setup
        self.tol = tol
        self.Nfg = Nfg
        self.threshold = threshold
        self.pred_type = pred_type
        self.Nfeat = Nfeat
        self.dmax = dmax
        self.gsi_score = gsi_score
        self.L1_reg = L1_reg
        self.pre_wind = pre_wind
        self.kernel = kernel
        
        
    def evaluation_metrics(self, Y, YPred):
        """
        Evaluate the quality of a prediction for a binary classification task.
            
        Parameters
        ----------
        Y : ndarray
            The target vector incorporating the true labels.
        YPred : ndarray
            The predicted class affiliations.
            
        Returns
        -------
        accuracy : float
            Share of correct predictions in all predictions.
        precision : float
            Share of true positives in all positive predictions.
        recall : float
            Share of true positives in all positive values.
        """
        # initialize TP, TN, FP, FN (true positive, true negative, false positive, false negative)
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for j in range(len(Y)):
            if Y[j]==1.0:
                if YPred[j]==1.0:
                    TP += 1
                elif YPred[j]==-1.0:
                    FN += 1
            elif Y[j]==-1.0:
                if YPred[j]==1.0:
                    FP += 1
                elif YPred[j]==-1.0:
                    TN += 1
        if (TP+TN) == 0:
            accuracy = 0
        else:
            accuracy = np.divide((TP+TN), len(Y))
        if TP == 0:
            precision = 0
            recall = 0
        else:
            precision = np.divide(TP, (TP+FP))
            recall = np.divide(TP, (TP+FN))
            
        # return evaluation metrics
        return accuracy, precision, recall
        
    
    def under_sample(self, X, y):
        """
        Balance the class distribution of the data X by under-sampling the over-represented class.
            
        Parameters
        ----------
        X : ndarray
            The data which is to be under-sampled.
        y : ndarray
            The target vector.
            
        Returns
        -------
        X : ndarray
            The balanced data.
        y : ndarray
            The corresponding target values of the balanced data.
        """
        # save label for all indices
        idx_pos = []
        idx_neg = []
        for i in range(len(y)):
            if y[i] == -1:
                idx_neg.append(i)
            else:
                idx_pos.append(i)
        
        # determine maximal number of samples per class for balanced subset
        n_max = min(len(idx_pos), len(idx_neg))
        if self.n_samples == None:
            num = n_max
        elif self.n_samples > n_max:
            raise Warning("n_samples exceeds the number of samples per class for the biggest possible balanced subset for the input data. Therefore, the biggest possible balanced subset is constructed.")
            num = n_max
        else:
            num = self.n_samples
            
        r1 = random.sample(idx_pos, num)
        r2 = random.sample(idx_neg, num)
        r_samples = r1 + r2
        
        X = X[r_samples,:]
        y = y[r_samples]
            
        return X, y
        
    def z_score_normalization(self, X_train, X_test):
        """
        Z-score normalize the training and test data.
        
        Note
        ----
        Only the training data is included in fitting the normalizer.
        The boolean "dt_train" indicates, whether the input data serves as training or test data.
            
        Parameters
        ----------
        X_train : ndarray
            The training data, the z-score-normalization is fitted on.
        X_test : ndarray
            The test data, that shall be normalized for which the statistics from the training data are used.
            
        Returns
        -------
        X_train_norm : ndarray
            Z-score normalized training data.
        X_test_norm : ndarray
            Z-score normalized test data.
        """
        scaler = StandardScaler()
        X_fit = scaler.fit(X_train)

        X_train_norm = X_fit.transform(X_train)
        X_test_norm = X_fit.transform(X_test)
            
        return X_train_norm, X_test_norm
    
    def preprocess_prescaled(self, X_train, X_test, sigma):
        """
        Prescale the data points and kernel parameter for before using prescaledfastadj.
        
        Parameters
        ----------
        X_train : ndarray
            The training data.
        X_test : ndarray
            The test data.
        sigma : ndarray
            The length-scale kernel parameter.

        Returns
        -------
        points_train : ndarray
            The scaled training data points.
        points_test : ndarray
             The scaled test data points.
        scaled_sigma : list
            The scaled list of candidate kernel parameters.
        """
        # scale data points equally
        points_center = np.mean(X_train, axis=0)
        points_train = X_train - points_center
        points_test = X_test - points_center

        # scale features such that abs(x[:,j]) <= 0.25
        # scale values in range [-0.25, 0.25]
        for j in range(X_train.shape[1]):
            m_train = np.max(np.abs(points_train[:,j]))
            points_train[:,j] = points_train[:,j] / m_train * 0.25
            m_test = np.max(np.abs(points_test[:,j]))
            points_test[:,j] = points_test[:,j] / m_test * 0.25
            
        ############
        # determine maximum number of features in window/kernel
        if self.classifier == "NFFTKernelRidge":
            if len(self.pre_wind) > 0:
                dmax = np.max([len(w) for w in self.pre_wind])
            else:
                dmax = self.dmax
            
        else:
            dmax = X_train.shape[1]
        ############
        
        # compute maximum radius possible in dmax dimensions
        scaling = np.sqrt(dmax)
        # ensure max radius 0.25 for points
        points_train = points_train / scaling
        points_test = points_test / scaling
        # scale sigma accordingly
        scaled_sigma = sigma / scaling
        
        return points_train, points_test, scaled_sigma
    
        
    def windows_weights(self, trainX, trainy, fg="mis", fgmode="distr", weighting="equally weighted", testX=[], testy=[]):
        """
        Determine feature windows and corresponding kernel weights and measure the window setup time.
        
        Parameters
        ----------
        trainX : ndarray
            The training data.
        trainy : ndarray
            The corresponding target values.
        fg : str, default="mis"
            The feature arrangement technique to be applied.
        fgmode : str, default="distr"
            The feature arrangement strategy to be applied.
        weighting : str, default="equally weighted"
            The weighting scheme for the kernel weights.
            If "equally weighted" the kernel weights are equal so that they sum up to 1.
            If "no weights" the kernel weights are all 1.
        testX : ndarray, default=[]
            The test data for the wrapper method RReliefFwrap.
        testy : ndarray, default=[]
            The corresponding target values.

        Returns
        -------
        wind : list
            List of determined feature windows.
        kweights : list
             List of corresponding kernel weights.
        twind : float
            Window setup time.
        """
        
        ##############################
        # FEATURE GROUPING
        
        # measure time needed for determining windows
        tstart_wind = time.time()
        
        # initialize feature-grouping class object
        fgm = feature_grouping(trainX[:self.Nfg,:], trainy[:self.Nfg], self.dmax, self.pred_type, self.kernel)
        
        # determine windows of features
        ##############
        if fg == "consec":
            wind = fgm.consec(Nfeat=self.Nfeat)
        ##############
        elif fg == "dt":
            wind = fgm.decision_tree(threshold=self.threshold, Nfeat=self.Nfeat, mode=fgmode)
        ##############
        elif fg == "mis":
            wind = fgm.mis(threshold=self.threshold, Nfeat=self.Nfeat, mode=fgmode)
        ##############
        elif fg == "fisher":
            wind = fgm.fisher(Nfeat=self.Nfeat, mode=fgmode)
        ##############
        elif fg == "reliefFfilt":
            wind = fgm.reliefFfilt(threshold=self.threshold, Nfeat=self.Nfeat, mode=fgmode)
        ##############
        elif fg == "reliefFwrap":
            wind = fgm.reliefFwrap(Xtest=testX, Ytest=testy, threshold=self.threshold, mode=fgmode)
        ##############
        elif fg == "lasso":
            wind = fgm.lasso(L1_reg=0.01, threshold=self.threshold, Nfeat=self.Nfeat, mode=fgmode, gs=False)
        ##############
        elif fg == "elastic_net":
            wind = fgm.elastic_net(L1_reg=0.01, threshold=self.threshold, Nfeat=self.Nfeat, mode=fgmode, gs=False)
        ##############
        elif fg == "fc_cc":
            wind = fgm.fc_cc()    
        ##############
        elif fg == "fg_opt":
            wind = fgm.fg_optimization(Nfg=500, l0=1, sy0=0.1, L1_reg=self.L1_reg)
        ##############
        elif fg == "gsi":
            wind0 = list(itertools.combinations(list(range(trainX.shape[1])), 2))
            clf = NFFTKernelRidgeFE(sigma=1, beta=1, setup=self.setup, tol=self.tol, pred_type=self.pred_type, kernel=self.kernel)
            alpha0 = clf.train_NFFT_KRR(trainX[:self.Nfg,:], trainy[:self.Nfg], wind0, weighting)
            wind = fgm.gsi(wind0, alpha0, self.gsi_score, l0=1)
        
        ##############################
        # overall time for determining windows
        twind = time.time() - tstart_wind
        ##############################
        # KERNEL WEIGHTING
            
        # determine kernel weights
        kweights = fgm.get_weights(weighting, wind)
        
        print("\n---------------")
        print("Feature grouping:", fg)
        print("Windows:", wind)
        print("----")
        print("Weighting:", weighting)
        print("Weights:", kweights)
        print("---------------\n")
        
        return wind, kweights, twind
    
    def tune(self, X_train, y_train, X_test, y_test, window_scheme="mis", fgmode="distr", weight_scheme="equally weighted"):
        """
        Tune over all candidate hyperparameter sets.
        
        Parameters
        ----------
        X_train : ndarray
            The training data.
        y_train : ndarray
            The corresponding labels for the training data.
        X_test : ndarray
            The test data.
        y_test : ndarray
            The corresponding labels for the test data.

        Returns
        -------
        best_params : list
            List of the parameters, which yield the highest value for the chosen scoring-parameter (accuracy, precision or recall).
        best_result : list
            List of the best results, where the chosen scoring-parameter is crucial.
        best_time_fit : float
            Fitting time of the run, that yielded the best result.
        best_time_pred : float
            Prediction time of the run, that yielded the best result.
        mean_total_time_fit : float
            Mean value over the fitting times of all candidate parameters.
        mean_total_time_pred : float
            Mean value over the prediction times of all candidate parameters.
        mean_total_time : float
            Mean value over the total times of all candidate parameters.
        windows : list
            List of feature windows.
        twind : float
            The window setup time.
        best_alpha : ndarray
            The dual-variable for the KRR model from the run yielding the best performance (depending on self.scoring).
        """
        param_names = list(self.param_grid)
        prod = []
        [prod.append(self.param_grid[param_names[i]]) for i in range(len(param_names))]
        params = list(itertools.product(*prod))
        #print("List of Candidate Parameters:", params)
        
        #####################################
        # data scaling
        
        # balance the class distribution of the data by under-sampling the over-represenetd class
        if self.balance == True:
            X_train, y_train = self.under_sample(X_train, y_train)
        
        # scale data with z-score-normalization
        if self.norm == 'z-score':
            X_train, X_test = self.z_score_normalization(X_train, X_test)
        
        # access candidate kernel parameters
        sig = [elem[0] for elem in params]
        #print("sig:", sig)
        
        # prescale data points and sigma for usage of prescaledfastadj
        points_train, points_test, scaled_sigma = self.preprocess_prescaled(X_train, X_test, sig)

        ######################################
        if self.classifier == "NFFTKernelRidge":
            # determine feature windows and weights
            if len(self.pre_wind) > 0:
                windows = self.pre_wind
                # initialize feature-grouping class object
                fgm = feature_grouping(X_train[:self.Nfg,:], y_train[:self.Nfg], self.dmax, self.pred_type, self.kernel)
                weights = fgm.get_weights(weight_scheme, windows)
                twind = 0
            else:
                windows, weights, twind = self.windows_weights(X_train, y_train, fg=window_scheme, fgmode=fgmode, weighting=weight_scheme, testX=X_test, testy=y_test)

        ######################################
        
        best_params = [0,0]
        if self.pred_type == "bin_class":
            best_result = [0,0,0]
        elif self.pred_type == "regression":
            best_result = 1000000
        
        total_time_fit = []
        total_time_pred = []
        total_time = []
        
        for j in range(len(params)):
            #print("j:", j)
            if self.classifier == "NFFTKernelRidge":
                
                # measure time needed for fitting
                start_fit = time.time()
                
                clf = NFFTKernelRidgeFE(sigma=scaled_sigma[j], beta=params[j][1], setup=self.setup, tol=self.tol, pred_type=self.pred_type, kernel=self.kernel)
            
                clf.fit(points_train, y_train, windows, weights)
                
                time_fit = time.time() - start_fit
                
                total_time_fit.append(time_fit)
                
                # measure time needed for predicting
                start_predict = time.time()
                
                YPred, windows, alpha = clf.predict(points_test)
                
                time_pred = time.time() - start_predict
                
                total_time_pred.append(time_pred)
                
                total_time.append((time_fit + time_pred))
            
            elif self.classifier == "sklearn KRR":
                # measure time needed for fitting
                start_fit = time.time()
                
                if self.kernel == "gaussian":
                    clf = KernelRidge(alpha=params[j][1], gamma=params[j][0], kernel='rbf')
                elif self.kernel == "matern":
                    clf = KernelRidge(alpha=params[j][1], gamma=params[j][0], kernel='laplacian')
                
                clf.fit(X_train, y_train)
                
                time_fit = time.time() - start_fit
                
                total_time_fit.append(time_fit)
                
                # measure time needed for predicting
                start_predict = time.time()
                
                YPred = np.sign(clf.predict(X_test))
                
                time_pred = time.time() - start_predict
                
                total_time_pred.append(time_pred)
                
                total_time.append((time_fit + time_pred))
                
            elif self.classifier == "sklearn SVC":
                # measure time needed for fitting
                start_fit = time.time()
                
                # note that sklearn.svm.SVC is not defined fot the laplacian kernel
                if self.pred_type == "bin_class":
                    clf = svm.SVC(C=params[j][1], gamma=params[j][0], kernel='rbf')
                elif self.pred_type == "regression":
                    clf = svm.SVR(C=params[j][1], gamma=params[j][0], kernel='rbf')
            
                clf.fit(X_train, y_train)
                
                time_fit = time.time() - start_fit
                
                total_time_fit.append(time_fit)
                
                # measure time needed for predicting
                start_predict = time.time()
                
                YPred = clf.predict(X_test)
            
                time_pred = time.time() - start_predict
                
                total_time_pred.append(time_pred)
                
                total_time.append((time_fit + time_pred))
            
            ########################################
            # prediction type: binary classification
            if self.pred_type == "bin_class":
                result = self.evaluation_metrics(y_test,YPred)
                 
                #print("Candidate Parameter:", params[j])
                #print("Result:", result)
                
                if self.scoring == "accuracy":
                    if result[0] > best_result[0]:
                        best_params = params[j]
                        best_result = result
                        best_time_fit = time_fit
                        best_time_pred = time_pred
                        if self.classifier == "NFFTKernelRidge":
                            best_alpha = alpha
                    
                elif self.scoring == "precision":
                    if result[1] > best_result[1]:
                        best_params = params[j]
                        best_result = result
                        best_time_fit = time_fit
                        best_time_pred = time_pred
                    
                elif self.scoring == "recall":
                    if result[2] > best_result[2]:
                        best_params = params[j]
                        best_result = result
                        best_time_fit = time_fit
                        best_time_pred = time_pred
            
            ########################################
            # prediction type: regression
            elif self.pred_type == "regression":
                # compute root mean square error
                result = np.sqrt(np.sum((y_test - YPred)**2)/len(y_test))
                
                if result < best_result:
                    best_params = params[j]
                    best_result = result
                    best_time_fit = time_fit
                    best_time_pred = time_pred
                    if self.classifier == "NFFTKernelRidge":
                        best_alpha = alpha
        
        if self.classifier == "NFFTKernelRidge":
            return best_params, best_result, best_time_fit, best_time_pred, np.mean(total_time_fit), np.mean(total_time_pred), np.mean(total_time), windows, twind, best_alpha
        
        else:
            return best_params, best_result, best_time_fit, best_time_pred, np.mean(total_time_fit), np.mean(total_time_pred), np.mean(total_time)