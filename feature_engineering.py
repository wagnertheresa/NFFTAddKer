"""
Author: Theresa Wagner <theresa.wagner@math.tu-chemnitz.de>

Corresponding publication:
"Fast Evaluation of Additive Kernels: Feature Arrangement, Fourier Methods and Kernel Derivatives"
by T. Wagner, F. Nestler, M. Stoll (2024)
"""
####################################################################################
####################################################################################

import numpy as np
import pandas as pd
import scipy
import warnings
import itertools
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import cg

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from skfeature.function.similarity_based import fisher_score
import sklearn_relief as sr
from sklearn.pipeline import Pipeline
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LassoLars
from sklearn.linear_model import ElasticNet

from gsi import GSI

class feature_grouping:
    """
    Feature Grouping Methods for Additive Kernel Ridge Regression Model.
    
    Parameters
    ----------
    X : ndarray
        The data based on which the feature arrangement shall be determined.
    y : ndarray
        The corresponding target values.
    dmax : int, default=3
        The maximal feature window length.
    pred_type : str, default="regression"
        The data type. Either regression or binary classification.
    kernel : str, default="gaussian"
        The kernel definition that shall be used.
        If "gaussian" the Gaussian kernel is used.
        If "matern" the Matérn(1/2) kernel is used.
    
    Examples
    -------
    
    """
    
    def __init__(self, X, y, dmax=3, pred_type="regression", kernel="gaussian"):
        self.X = X
        self.y = y
        self.dmax = dmax # max number of features in window
        self.pred_type = pred_type
        self.kernel = kernel
    
    def arrange_groups(self, scores, threshold=0.001, Nfeat=9, mode="distr"):
        """
        Construct group arrangement based on feature importance scores.
        
        Parameters
        ----------
        scores : list
            The feature importance scores.
        threshold : float, default=0.001
            The threshold for dropping features with an importance score below that value.
        Nfeat : int, default=9
            The total number of features to be included.
        mode : str, default="distr"
            The feature arrangement strategy.
        
        Returns
        -------
        windows : list
            The determined feature windows.
        
        """
        # sort scores in descending order and convert np arrays to list
        sorted_scores = (np.sort(scores)[::-1]).tolist()
        sorted_idx = (np.argsort(scores)[::-1]).tolist()
        #print("sorted_scores:", sorted_scores)
        #print("sorted_idx:", sorted_idx)
        
        # determine indices of non-zero coeffs
        nz_coeff = list(itertools.compress(itertools.count(), scores))
        #print("nz_coeff:", nz_coeff)
        d_coeff = len(nz_coeff)
        #print("Number non-zero Features:", d_coeff)
        
        # drop features with coeff zero and bigger index than Nfeat
        pre_idx = sorted_idx[:np.min((d_coeff, Nfeat))]
        #print("preliminary idx:", pre_idx)
            
        # adjust threshold, if not enough features have a score above threshold
        while len([i for i in pre_idx if scores[i] >= threshold]) < self.dmax:
            print('Too many features are discarded with the chosen threshold. The threshold will be halved in the following.')
            threshold = threshold * 0.5
        
        # drop features with importance below threshold
        idx = [i for i in pre_idx if scores[i] >= threshold]
        
        # number of feature indices
        d = len(idx)
        
        # construct groups following index ranking, so that first dmax features with highest score build first window and so on
        if mode == "consec":
            # create windows of length dmax
            windows = [idx[(l*self.dmax):(l*self.dmax)+self.dmax] for l in range(d//self.dmax)]
            
            # if |d| is not divisible by dmax, the last window contains only 1 or 2 indices, respectively
            if d%self.dmax != 0:
                windows.append([idx[i] for i in range(d - d%self.dmax,d)])
        
        # construct groups following index ranking, so that we iterate over feature groups and always assign feature next in ranking to corresponding groups
        elif mode == "distr":
            # determine number of feature groups
            Ngroups = d//self.dmax
            if d%self.dmax != 0:
                Ngroups += 1
            
            windows = [idx[i::Ngroups][:self.dmax] for i in range(Ngroups)]
            
        return windows
    
    ###################################################################################
    ###################################################################################
    # Consecutive windows
    
    def consec(self, Nfeat=9):
        """
        Construct feature windows based on consecutive order of feature indices.
        
        Parameters
        ----------
        Nfeat : int, default=9
            The total number of features to be included.

        Returns
        -------
        windows : list
            The determined feature windows.
            
        """
        d = Nfeat
        
        # add windows of length dmax               
        windows = [list(range((l*self.dmax),(l*self.dmax) + self.dmax)) for l in range(d//self.dmax)]
    
        # if |d| is not divisible by dmax, the last window contains only 1 or 2 indices                
        if d%self.dmax != 0:
            windows.append([l for l in range(d - d%self.dmax, d)])
        
        return windows
    
    ####################################################################################
    ####################################################################################
    # Decision tree
    
    def decision_tree(self, threshold=0.001, Nfeat=9, mode="distr"):
        """
        Determine feature importance scores based on a decision tree model and construct corresponding feature windows.
        
        Ref: https://machinelearningmastery.com/calculate-feature-importance-with-python/
        
        Parameters
        ----------
        threshold : float, default=0.001
            The threshold for dropping features with an importance score below that value.
        Nfeat : int, default=9
            The total number of features to be included.
        mode : str, default="distr"
            The feature arrangement strategy.

        Returns
        -------
        windows : list
            The determined feature windows.
            
        """
        if self.pred_type == "bin_class":
            dt = DecisionTreeClassifier(criterion="entropy")
        elif self.pred_type == "regression":
            dt = DecisionTreeRegressor(criterion="squared_error")
        dt.fit(self.X, self.y)
        # get feature importance scores
        scores = dt.feature_importances_
        #print("feature importance:", scores)
        
        windows = self.arrange_groups(scores, threshold=threshold, Nfeat=Nfeat, mode=mode)
        
        return windows
    
    ####################################################################################
    ####################################################################################
    # Mutual information score
    
    def mis(self, threshold=0.001, Nfeat=9, mode="distr"):
        """
        Determine feature importance scores based on mutual information score and construct corresponding feature windows.
        
        Parameters
        ----------
        threshold : float, default=0.001
            The threshold for dropping features with an importance score below that value.
        Nfeat : int, default=9
            The total number of features to be included.
        mode : str, default="distr"
            The feature arrangement strategy.

        Returns
        -------
        windows : list
            The determined feature windows.
            
        """
        if self.pred_type == "bin_class":
            mi_scores = mutual_info_classif(self.X, self.y)
        elif self.pred_type == "regression":
            mi_scores = mutual_info_regression(self.X, self.y)
        mi_scores = pd.Series(mi_scores, name="MI Scores")
        mi_scores = mi_scores.tolist()
        #print("mi_scores:", mi_scores)
        
        windows = self.arrange_groups(mi_scores, threshold=threshold, Nfeat=Nfeat, mode=mode)
            
        return windows
    
    ###################################################################################
    ###################################################################################
    # Fisher score
    
    def fisher(self, Nfeat=9, mode="distr"):
        """
        Determine feature ranking based on Fisher score and construct corresponding feature windows.
        
        Parameters
        ----------
        Nfeat : int, default=9
            The total number of features to be included.
        mode : str, default="distr"
            The feature arrangement strategy.

        Returns
        -------
        windows : list
            The determined feature windows.
            
        """
        sorted_idx = fisher_score.fisher_score(self.X, self.y, mode="index")
        #print("sorted_idx:", sorted_idx)
            
        # drop all but Nfeat features and convert into list
        f_idx = list(sorted_idx[:Nfeat])
        #print("pre_idx:", f_idx)
        
        # number of features
        d = len(f_idx)
        
        # construct groups following index ranking, so that first dmax features with highest score build first window and so on
        if mode == "consec":
            # create windows of length dmax
            windows = [f_idx[(l*self.dmax):(l*self.dmax)+self.dmax] for l in range(d//self.dmax)]
            
            # if |d| is not divisible by dmax, the last window contains only 1 or 2 indices, respectively
            if d%self.dmax != 0:
                windows.append([f_idx[i] for i in range(d - d%self.dmax,d)])
        # construct groups following index ranking, so that we iterate over feature groups and always assign feature next in ranking to corresponding groups
        elif mode == "distr":
            # determine number of feature groups
            Ngroups = d//self.dmax
            if d%self.dmax != 0:
                Ngroups += 1
            
            windows = [f_idx[i::Ngroups][:self.dmax] for i in range(Ngroups)]
        
        return windows
    
    ####################################################################################
    ####################################################################################
    # RReliefF as filter method for regression
    
    def reliefFfilt(self, threshold=0.001, Nfeat=9, mode="distr"):
        """
        Determine feature ranking based on RReliefF filter method and construct corresponding feature windows.
        
        Ref: https://www.kaggle.com/code/jorgesandoval/feature-selection-with-rrelieff-regression
        
        Parameters
        ----------
        threshold : float, default=0.001
            The threshold for dropping features with an importance score below that value.
        Nfeat : int, default=9
            The total number of features to be included.
        mode : str, default="distr"
            The feature arrangement strategy.

        Returns
        -------
        windows : list
            The determined feature windows.
            
        """
        if self.pred_type == "bin_class":
            r = sr.ReliefF(n_features=self.X.shape[1])
        elif self.pred_type == "regression":
            r = sr.RReliefF(n_features=self.X.shape[1])
        
        # fit model
        r.fit(self.X, self.y)
        # feature importance scores
        fscores = np.abs(r.w_)
        #print("fscores:", fscores)
        
        windows = self.arrange_groups(fscores, threshold=threshold, Nfeat=Nfeat, mode=mode)
            
        return windows
    
    
    ####################################################################################
    ####################################################################################
    # ReliefF as wrapper method
    
    def reliefFwrap(self, Xtest, Ytest, threshold=0.001, mode="distr"):
        """
        Determine feature ranking based on RReliefF wrapper method and construct corresponding feature windows.
        
        Ref: https://www.kaggle.com/code/jorgesandoval/feature-selection-with-rrelieff-regression
        
        Parameters
        ----------
        Xtest : ndarray
            The test data.
        Ytest : ndarray
            The corresponding target values.
        threshold : float, default=0.001
            The threshold for dropping features with an importance score below that value.
        mode : str, default="distr"
            The feature arrangement strategy.

        Returns
        -------
        windows : list
            The determined feature windows.
            
        """
        nof_list = np.arange(1,self.X.shape[1])            
        high_score = 0
        nof = 0           
        score_list = []
        for n in range(len(nof_list)):
            if self.pred_type == "bin_class":
                fs = sr.ReliefF(n_features = nof_list[n])
            elif self.pred_type == "regression":
                fs = sr.RReliefF(n_features = nof_list[n])
            if self.kernel == "gaussian":
                relief = Pipeline([('fs', fs), ('m', KernelRidge(kernel="rbf"))])
            elif self.kernel == "matern":
                relief = Pipeline([('fs', fs), ('m', KernelRidge(kernel="laplacian"))])
            relief.fit(self.X,self.y)
            score = relief.score(Xtest,Ytest)
            score_list.append(score)
            #print(f'NOF: {nof_list[n]}, Score: {score}')
            if(score > high_score):
                high_score = score
                nof = nof_list[n]
                best_w = fs.w_
            
        # feature importance scores for best nof
        fscores = np.abs(best_w)
        #print("fscores:", fscores)
        
        windows = self.arrange_groups(fscores, threshold=threshold, Nfeat=nof, mode=mode)
        
        return windows
    
    ####################################################################################
    ####################################################################################
    # Lasso regularization
    
    def lasso(self, L1_reg=0.01, threshold=0.001, Nfeat=9, mode="consec"):
        """
        Determine feature importance scores based on lasso regularization model and construct corresponding feature windows.
        
        Ref: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html
        
        Parameters
        ----------
        L1_reg : float, default=0.01
            L1 regularization parameter for lasso.
        threshold : float, default=0.001
            The threshold for dropping features with an importance score below that value.
        Nfeat : int, default=9
            The total number of features to be included.
        mode : str, default="consec"
            The feature arrangement strategy.

        Returns
        -------
        windows : list
            The determined feature windows.
        
        """   
        lasso = LassoLars(alpha=L1_reg)
        lasso.fit(self.X, self.y)
            
        coeff = np.abs(lasso.coef_)
        #print("coeff:", coeff)
        
        if mode == "direct":
            # determine indices of non-zero coeffs
            nz_coeff = list(itertools.compress(itertools.count(), coeff))
            #print("nz_coeff:", nz_coeff)
            d_coeff = len(nz_coeff)
            #print("Number non-zero Features:", d_coeff)
            
            # add windows of length dmax               
            windows = [nz_coeff[l*self.dmax:l*self.dmax+self.dmax] for l in range(d_coeff//self.dmax)]
        
            # if |d| is not divisible by dmax, the last window contains only 1 or 2 indices                
            if d_coeff%self.dmax != 0:
                windows.append([nz_coeff[l] for l in range(d_coeff - d_coeff%self.dmax, d_coeff)])
        
        else:
            windows = self.arrange_groups(coeff, threshold=threshold, Nfeat=Nfeat, mode=mode)
            
        return windows
    
    ####################################################################################
    ####################################################################################
    # Elastic net regularization
    
    def elastic_net(self, L1_reg=0.01, threshold=0.001, Nfeat=9, mode="consec"):
        """
        Determine feature importance scores based on elastic net regularization model and construct corresponding feature windows.
        
        Ref: https://www.kaggle.com/code/cast42/feature-selection-and-elastic-net
        
        Parameters
        ----------
        L1_reg : float, default=0.01
            L1 regularization parameter for lasso.
        threshold : float, default=0.001
            The threshold for dropping features with an importance score below that value.
        Nfeat : int, default=9
            The total number of features to be included.
        mode : str, default="consec"
            The feature arrangement strategy.

        Returns
        -------
        windows : list
            The determined feature windows.
            
        """
        elastic = ElasticNet(alpha=L1_reg, l1_ratio=0.5)
        elastic.fit(self.X, self.y)
        
        coeff = np.abs(elastic.coef_)
        #print("coeff:", coeff)
        
        if mode == "direct":
            # determine indices of non-zero coeffs
            nz_coeff = list(itertools.compress(itertools.count(), coeff))
            #print("nz_coeff:", nz_coeff)
            d_coeff = len(nz_coeff)
            #print("Number non-zero Features:", d_coeff)
            
            # add windows of length dmax               
            windows = [nz_coeff[l*self.dmax:l*self.dmax+self.dmax] for l in range(d_coeff//self.dmax)]
        
            # if |d| is not divisible by dmax, the last window contains only 1 or 2 indices                
            if d_coeff%self.dmax != 0:
                windows.append([nz_coeff[l] for l in range(d_coeff - d_coeff%self.dmax, d_coeff)])
        
        else:
            windows = self.arrange_groups(coeff, threshold=threshold, Nfeat=Nfeat, mode=mode)
            
        return windows
    
    ####################################################################################
    ####################################################################################
    # Feature clustering via connected components
    
    def fc_cc(self, mode="distr"):
        """
        Determine feature importance scores based on feature clustering via connected components and construct corresponding feature windows.
        
        Ref: https://mltechniques.com/2023/03/12/feature-clustering-a-simple-solution-to-many-machine-learning-problems/
             https://github.com/VincentGranville/Main/blob/main/featureClustering.py
        
        Parameters
        ----------
        mode : str, default="distr"
            The feature arrangement strategy.

        Returns
        -------
        windows : list
            The determined feature windows.
        
        """
        # create correlation matrix
        corrmat = np.corrcoef(self.X, rowvar=False)
        
        dim = len(corrmat)
        threshold = 0.4  # two features with |correl|>threshold are connected 
        pairs = {}
        
        for i in range(dim):
            for j in range(i+1,dim):
                dist = abs(corrmat[i][j])
                if dist > threshold:
                    pairs[(i,j)] = abs(corrmat[i][j])
                    pairs[(j,i)] = abs(corrmat[i][j])
        
        # connected components algo to detect feature clusters on feature pairs
        # PART 1: Initialization. 
        
        point=[]
        NNIdx={}
        idxHash={}
        
        n=0
        for key in pairs:
            idx = key[0]
            idx2 = key[1]
            if idx in idxHash:
                idxHash[idx] = idxHash[idx]+1
            else:
                idxHash[idx] = 1
            point.append(idx)
            NNIdx[idx] = idx2
            n = n+1
        
        hash = {}
        for i in range(n):
            idx = point[i]
            if idx in NNIdx:
                substring = [NNIdx[idx]]
            string = []
            if idx in hash:
                string = hash[idx]
            if set(substring).issubset(string) == False:
                if idx in hash:
                    hash[idx] = hash[idx] + substring 
                else:
                    hash[idx] = substring    
            substring = [idx]
            if NNIdx[idx] in hash: 
                string = hash[NNIdx[idx]]
            if set(substring).issubset(string) == False:
                if NNIdx[idx] in hash:
                    hash[NNIdx[idx]] = hash[NNIdx[idx]] + substring 
                else:
                    hash[NNIdx[idx]] = substring 

        # PART 2: Find the connected components 
        i = 0
        status = {}
        stack = {}
        onStack = {}
        cliqueHash = {}
        
        while i<n:
        
            while (i<n and point[i] in status and status[point[i]]==-1):    
                # point[i] already assigned to a clique, move to next point
                i = i+1
        
            nstack = 1
            if i<n:
                idx = point[i]
                stack[0] = idx    # initialize the point stack, by adding $idx 
                onStack[idx]=1
                #size = 1    # size of the stack at any given time
        
                while nstack>0:    
                    idx = stack[nstack-1]
                    if (idx not in status) or status[idx] != -1: 
                        status[idx] = -1    # idx considered processed
                        if i<n:
                            if point[i] in cliqueHash:
                                cliqueHash[point[i]] = cliqueHash[point[i]] + [idx]
                            else: 
                                cliqueHash[point[i]] = [idx]
                        nstack = nstack-1 
                        for idx2 in hash[idx]:
                            # loop over all points that have point idx as nearest neighbor
                            idx2 = int(idx2)
                            if idx2 not in status or status[idx2] != -1:     
                                # add point idx2 on the stack if it is not there yet
                                if idx2 not in onStack: 
                                    stack[nstack] = idx2
                                    nstack = nstack+1
                                onStack[idx2] = 1
                                
        #print("cliqueHash:", cliqueHash)
        
        clkeys = cliqueHash.keys()
        windows = []
        
        # construct groups following index ranking, so that first dmax features with highest score build first window and so on
        if mode == "consec":
            for key in clkeys:
                if len(cliqueHash[key]) > self.dmax:
                    subwind = cliqueHash[key][:self.dmax]
                else:
                    subwind = cliqueHash[key]
                windows.append(subwind)

        # construct groups following index ranking, so that we iterate over feature groups and always assign feature next in ranking to corresponding groups
        elif mode == "distr":
            # determine number of feature groups
            Ngroups = len(list(clkeys))
            
            vals = list(cliqueHash.values())
            #print("vals:", vals)
            
            lmax = np.min([len(l) for l in vals])
            pre_idx = [l[:lmax] for l in vals]
            #print("pre_idx:", pre_idx)
            idx = list(itertools.chain.from_iterable(pre_idx))
            #print("idx:", idx)
            
            windows = [idx[i::Ngroups][:self.dmax] for i in range(Ngroups)]
            
        # center points of connected components form the groups what yields windows of length 1
        elif mode == "single":
            windows = [[key] for key in list(clkeys)]

        return windows    
    
    ###################################################################################
    ###################################################################################
    # Global sensitivity indices
    
    def gsi(self, wind, alpha, gsi_score, l0):
        """
        Determine global sensitivity indices and construct corresponding feature windows.
        
        Parameters
        ----------
        wind : list
            Initial list of windows (all feature subsets of length 2).
        alpha : ndarray
            The solution vector obtained by training the model with all feature subsets of length 2.
        gsi_score : float
            The score until the feature subsets shall be added to the list of windows.
        l0 : float
            Initial length-scale kernel parameter for determining the global sensitivity indices.
        
        Returns
        -------
        windows : list
            The determined feature windows.
            
        """
        gsi = GSI(self.X, l0, wind)

        gsi_sorted = gsi.gsi_sorted(alpha)
        gsi_keys = list(gsi_sorted.keys())
        #print("gsi_keys:", gsi_keys)
        
        # initialize gsi sum
        gsi_sum = 0
        # intialize list for gsi_based windows
        windows = []
        cnt = 0
        while (gsi_sum + gsi_sorted[gsi_keys[cnt]]) < gsi_score:
            windows.append(gsi_keys[cnt])
            gsi_sum += gsi_sorted[gsi_keys[cnt]]
            cnt += 1
            
        return windows
    
    ####################################################################################
    ####################################################################################
    # Feature grouping optimization
    
    def fg_optimization(self, Nfg, l0, sy0, L1_reg):
        """
        Perform feature grouping optimization of all feature subsets of length 2 and construct corresponding feature windows.
        
        Parameters
        ----------
        Nfg : int
            The data subset size on that the FGO shall be performed.
        l0 : float
            The fixed length-scale parameter for the FGO.
        sy0 : float
            The fixed kernel noise parameter for the FGO.
        L1_reg : float
            The L1 regularization parameter.
        
        Returns
        -------
        windows : list
            The determined feature windows.
            
        """
        
        # define smaller dataset for feature grouping: first Nfg data points
        X_fg = self.X[:Nfg,:]
        y_fg = self.y[:Nfg]
        
        # define global variables for fmin_cgprox
        global Xfg
        Xfg = X_fg
        global yfg
        yfg = y_fg
        global l
        l = l0
        global sy
        sy = sy0
        global L1reg
        L1reg = L1_reg
        
        # approximate K as sum of k w.r.t all possible windows of 2 features
        # all combinations of features of length 2
        wind = list(itertools.combinations(list(range(0,X_fg.shape[1])),2))
        global windows
        windows = wind
        global nw
        nw = len(wind)
        
        ## set initial value for sf_fg (initial sf values for fg) and initialize parameter list of correct length
        sf0_fg = list(np.hstack(np.random.uniform(1e-1,1e+1,nw)))
        
        ## optimization to detect relevant feature windows
        # optimization with L1 regularizer (to make most weights 0)
        res_fg = self.fmin_cgprox(func_fg=self.func_fg, f=self.f, f_prime=self.f_prime, g_prox=self.g_prox, x0=sf0_fg)
        
        # optimized theta values
        sf_fg_hat = res_fg.x
        #print("sf_fg_hat:", sf_fg_hat)
        
        # determine indices of non-zero weights
        nz_w = list(itertools.compress(itertools.count(), sf_fg_hat))
        #print("nz_w:", nz_w)
        #print("Number non-zero Windows:", len(nz_w))
        # detected selection of windows
        windows = [wind[i] for i in nz_w]
        #print("Windows:", windows)
        ##############################
        ##############################
        
        if len(windows) == 0:
            raise Warning("The feature grouping optimization returned an empty list of windows!")
        return windows

    
    def fmin_cgprox(self, func_fg, f, f_prime, g_prox, x0, rtol=1e-6, maxiter=1000, verbose=0, default_step_size=1.):
        
        """
        proximal gradient-descent solver for optimization problems of the form
    
                           minimize_x f(x) + g(x)
    
        where f is a smooth function and g is a (possibly non-smooth)
        function for which the proximal operator is known.
    
        Parameters
        ----------
        f : callable
            f(x) returns the value of f at x.
    
        f_prime : callable
            f_prime(x) returns the gradient of f.
    
        g_prox : callable of the form g_prox(x, alpha)
            g_prox(x, alpha) returns the proximal operator of g at x
            with parameter alpha.
    
        x0 : array-like
            Initial guess
    
        maxiter : int
            Maximum number of iterations.
    
        verbose : int
            Verbosity level, from 0 (no output) to 2 (output on each iteration)
    
        default_step_size : float
            Starting value for the line-search procedure.
    
        Returns
        -------
        res : OptimizeResult
            The optimization result represented as a
            ``scipy.optimize.OptimizeResult`` object. Important attributes are:
            ``x`` the solution array, ``success`` a Boolean flag indicating if
            the optimizer exited successfully and ``message`` which describes
            the cause of the termination. See `scipy.optimize.OptimizeResult`
            for a description of other attributes.
        """
        xk = x0
        fk_old = np.inf
    
        fk, grad_fk = func_fg(xk)
        
        success = False
        for it in range(maxiter):
            # .. step 1 ..
            # Find suitable step size
            step_size = default_step_size  # initial guess
            grad_fk = f_prime(xk)
            while True:  # adjust step size
                xk_grad = xk - step_size * grad_fk
                prx = self.g_prox(xk_grad, step_size)
                Gt = (xk - prx) / step_size
                lhand = f(xk - step_size * Gt)
                rhand = fk - step_size * grad_fk.dot(Gt) + \
                    (0.5 * step_size) * Gt.dot(Gt)
                if lhand <= rhand:
                    # step size found
                    break
                else:
                    # backtrack, reduce step size
                    step_size *= .5
    
            xk -= step_size * Gt
            fk_old = fk
            
            fk, grad_fk = func_fg(xk)
    
            if verbose > 1:
                print("Iteration %s, Error: %s" % (it, scipy.linalg.norm(Gt)))
    
            if np.abs(fk_old - fk) / fk < rtol:
                if verbose:
                    print("Achieved relative tolerance at iteration %s" % it)
                    success = True
                break
        else:
            warnings.warn(
                "fmin_cgprox did not reach the desired tolerance level",
                RuntimeWarning)
    
        return scipy.optimize.OptimizeResult(x=xk, success=success, fun=fk, jac=grad_fk, nit=it)


    def kermat(self, X, l):
        """
        Compute the Gaussian kernel matrix.
        
        Parameters
        ----------
        X : ndarray
            The data matrix.
        l : float, default=1.0
            The length-scale parameter.

        Returns
        -------
        K : ndarray
            The generated kernel matrix.
            
        """
        pairwise_dists = squareform(pdist(X, "euclidean"))
        if self.kernel == "gaussian":
            K = np.exp(- (pairwise_dists ** 2) /(2* l ** 2))
        elif self.kernel == "matern":
            K = np.exp(- pairwise_dists/l)
        
        return K

    def setup_kernels(self, X, theta):
        """
        """
        K = [theta[i+1]**2 * self.kermat(X[:,windows[i]], theta[0]) for i in range(nw)]
        K_sum = np.sum(K, axis=0)
        K_tilde = K_sum + (nw*(theta[-1]**2)*np.eye(X.shape[0]))
        return K, K_sum, K_tilde
    
    def f(self, sf):
        """
        Compute the objective of the KRR model.
        
        Parameters
        ----------
        sf : list
            The weight parameters for the subkernels.

        Returns
        -------
        obj : float
            The KRR objective.
        
        """
        theta = list(np.hstack((l, sf, sy)))
        
        K, K_sum, K_tilde = self.setup_kernels(Xfg, theta)
        alpha = cg(K_tilde, yfg, x0=None, tol=1e-08, maxiter=500)[0]
        
        # objective
        obj = np.linalg.norm(yfg - K_sum@alpha)**2 + nw*sy**2*alpha.T.dot(K_sum@alpha)
        
        return obj
    
    
    def f_prime(self, sf):
        """
        Compute the derivative with respect to the kernel weights of the KRR model.
        
        Parameters
        ----------
        sf : list
            The weight parameters for the subkernels.

        Returns
        -------
        der2 : ndarray
            The KRR derivative with respect to the kernel weights.
        
        """
        theta = list(np.hstack((l, sf, sy)))
        
        K, K_sum, K_tilde = self.setup_kernels(Xfg, theta)
        alpha = cg(K_tilde, yfg, x0=None, tol=1e-08, maxiter=500)[0]
        
        # derivatives wrt sf
        dK2 = [(2/sf[i])*K[i] if sf[i] > 1e-10 else 0*K[i] for i in range(nw)]
        der2 = [-2*(dK2[i]@(alpha))@yfg + 2*np.dot(K_sum@(dK2[i]@alpha), alpha) + nw*sy**2*np.dot(alpha.T, dK2[i]@alpha) for i in range(nw)]
        der2 = (np.hstack(der2)).T
        
        return der2
    
    
    def func_fg(self, sf):
        """
        ""
        Compute the objective and derivative with respect to the kernel weights of the KRR model.
        
        Parameters
        ----------
        sf : list
            The weight parameters for the subkernels.

        Returns
        -------
        obj : float
            The KRR objective.
        der2 : ndarray
            The KRR derivative with respect to the kernel weights.
        
        """
        theta = list(np.hstack((l, sf, sy)))
         
        K, K_sum, K_tilde = self.setup_kernels(Xfg, theta)
        alpha = cg(K_tilde, yfg, x0=None, tol=1e-08, maxiter=500)[0]
        
        # objective
        obj = np.linalg.norm(yfg - K_sum@alpha)**2 + nw*sy**2* np.transpose(alpha)@(K_sum@alpha)
        
        # derivatives wrt sf
        dK2 = [(2/sf[i])*K[i] if sf[i] > 1e-10 else 0*K[i] for i in range(nw)]
        der2 = [-2*(dK2[i]@(alpha))@yfg + 2*np.dot(K_sum@(dK2[i]@alpha), alpha) + nw*sy**2*np.dot(alpha.T, dK2[i]@alpha) for i in range(nw)]
        der2 = (np.hstack(der2)).T
        
        return obj, der2
    
    
    def g_prox(self, sf, step_size):
        """
        Determine kernel weights for next optimization step.
        
        Parameters
        ----------
        sf : list
            The weight parameters for the subkernels.
        step_size : float
            Step size for optimization step.

        Returns
        -------
        prx : ndarray
            Kernel weights in next optimization step.

        """
        return np.fmax(sf - step_size*L1reg, 0) - np.fmax(-sf - step_size*L1reg, 0)
    
    ##################################################################################
    ##################################################################################
    ##################################################################################
    
    def get_weights(self, weighting, wind):
        """
        Determine the kernel weights.
        
        Parameters
        ----------
        weighting : str
            Scheme after which kernels shall be weighted.
            If "equally weighted" all kernels are weighted equally so that the weights sum up to 1.
            If "no weights" weight sf is 1.
        wind : list
            The feature windows.
        
        Returns
        -------
        kweights : list
            The determined feature windows.
        
        """
        # equally weighted kernels, so that weight sf = sqrt(1/len(wind))
        if weighting == "equally weighted":
            kweights = np.sqrt(1/len(wind))
        # no weighting, weight sf = 1
        elif weighting == "no weights":
            kweights = 1
        
        return kweights
    
#########################################################################################
#########################################################################################