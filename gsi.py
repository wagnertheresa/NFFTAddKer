"""
Authors: Theresa Wagner <theresa.wagner@math.tu-chemnitz.de>, Franziska Nestler <franziska.nestler@math.tu-chemnitz.de>

Corresponding publication:
"Fast Evaluation of Additive Kernels: Feature Arrangement, Fourier Methods and Kernel Derivatives"
by T. Wagner, F. Nestler, M. Stoll (2024)
"""
####################################################################################
####################################################################################

import numpy as np
import itertools
import operator

from pynufft import NUFFT

class GSI:
    """
    Compute the Global Sensitivity Indices for All Feature Subsets.
    
    Parameters
    ----------
    X_train : ndarray
        The training data.
    ell : float
         Length-scale parameter for the Gaussian kernel.
    windows : list
        The list of original feature windows.
    fastadj_setup : str, default="default"
        The NFFT parameter setting.
    
    Attributes
    ----------
    wind : list
        The list of feature windows.
    M : int
        The number of Fourier coefficients.
    
    Examples
    --------
    >>> 
    """
    
    def __init__(self, X_train, ell, windows, fastadj_setup="default"):
        self.X_train = X_train
        self.ell = ell
        self.windows = windows
        self.fastadj_setup = fastadj_setup
    
    ###################################################################################
    
    def modified_gaussian_function(self, X):
        """
        Generate modified Gaussian function on the grid.
        
        Parameters
        ----------
        X : ndarray
            The data points.

        Returns
        -------
        modified_gaussian_data : ndarray
            The modified Gaussian function on the grid.
        
        """
        dim = len(X)
        
        if dim == 1:
            X = X[0]
            
            return np.exp(- (X**2) / 2*self.ell**2) 
            
        elif dim == 2:
            X0 = X[0]
            X1 = X[1]
            condition = X0**2 + X1**2 <= 1/4
            gaussian_inside = np.exp(- (X0**2 + X1**2) / 2*self.ell**2) 
            gaussian_outside = np.exp(- (1/2)**2 / 2*self.ell**2)
            
            return np.where(condition, gaussian_inside, gaussian_outside)
            
        elif dim == 3:
            X0 = X[0]
            X1 = X[1]
            X2 = X[2]
            condition = X0**2 + X1**2 + X2**2 <= 1/4
            gaussian_inside = np.exp(- (X0**2 + X1**2 + X2**2) / 2*self.ell**2) 
            gaussian_outside = np.exp(- (1/2)**2 / 2*self.ell**2) 
            
            return np.where(condition, gaussian_inside, gaussian_outside)
        
    ###################################################################################
    
    def discrete_fourier_transform(self, data):
        """
        Compute the discrete Fourier coefficients for the modified Gaussian on the grid.
        
        Parameters
        ----------
        data : ndarray
            The feature windows.

        Returns
        -------
        fft_result : dict
            The discrete Fourier transform.
        
        """
        
        if data.ndim == 1:
            return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(data)))
            
        elif data.ndim == 2:
            return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(data)))
        
        elif data.ndim == 3:
            return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(data)))
    
    ###################################################################################
    
    def compute_fft_coeff(self, wind):
        """
        Compute the Fourier coefficients for the superposition dimensions present in the subsets of features.
        
        Parameters
        ----------
        wind : list
            The feature windows.

        Returns
        -------
        fft_coeff_dict : dict
            The dict of Fourier coefficients for the superposition dimensions.
        
        """
        #print("wind:", wind)
        # determine lengths of feature windows
        d_list = [len(elem) for elem in wind]
        #print("lengths of windows:", d_list)
        
        # remove duplicates from d_list
        d_list = list(dict.fromkeys(d_list))
        #print("unique lengths of windows:", d_list)
        
        # create dict for fft_coeffs for all lengths
        fft_coeff_dict = {}
                
        for d in d_list:
            # create meshgrid
            if d == 1:
                X = np.linspace(-1/2, 1/2, self.M, endpoint=False)
                X = [X]
            elif d == 2:
                x0 = np.linspace(-1/2, 1/2, self.M, endpoint=False) 
                x1 = np.linspace(-1/2, 1/2, self.M, endpoint=False) 
                X0, X1 = np.meshgrid(x0, x1)
                X = [X0,X1]
            elif d == 3:
                x0 = np.linspace(-1/2, 1/2, self.M, endpoint=False) 
                x1 = np.linspace(-1/2, 1/2, self.M, endpoint=False) 
                x2 = np.linspace(-1/2, 1/2, self.M, endpoint=False) 
                X0, X1, X2 = np.meshgrid(x0, x1, x2)
                X = [X0, X1, X2]
            
            # Generate modified Gaussian function on the grid
            modified_gaussian_data = self.modified_gaussian_function(X)
            
            # Compute discrete Fourier coefficients using FFT
            fft_result = self.discrete_fourier_transform(modified_gaussian_data)
            fft_coeff = fft_result * 1/(self.M**d) # devide by total number of coefficients
            
            #print(fft_coeff)
            
            # add fft_coeff to dict
            fft_coeff_dict[d] = fft_coeff
            
        #print("keys fft_coeff_dict:", fft_coeff_dict.keys())
        
        return fft_coeff_dict
    
    ###################################################################################
    
    def compute_nufft(self, windl, d, vec):
        """
        Compute the adjoint NFFT (pyNUFFT) for the corresponding feature subset to be considered.
        
        Parameters
        ----------
        windl : list
            The feature subset to be considered.
        d : int
            The superposition dimension.
        vec : ndarray
            The solution vector obtained by training the model with all feature subsets of length 2.
        
        Returns
        -------
        nufft_result_type2 : dict
            The type-2 NUFFT (type-2 means adjoint).
        
        """
        
        # Set up NUFFT object
        
        if d == 1:
            Nd = (self.M,) # grid size
            Kd = (2*self.M,) # oversampled grid size
            Jd = (8,) # maybe the same like m? TODO: find out
            
        elif d == 2:
            Nd = (self.M, self.M) # grid size
            Kd = (2*self.M, 2*self.M) # oversampled grid size
            Jd = (8,8) # maybe the same like m? TODO: find out
            
        elif d == 3:
            Nd = (self.M, self.M, self.M) # grid size
            Kd = (2*self.M, 2*self.M, 2*self.M) # oversampled grid size
            Jd = (8,8,8) # maybe the same like m? TODO: find out
        
        #########################
        
        #print("Create NUFFT object for windl=", windl)
        # create NUFFT object
        NufftObj = NUFFT()
        NufftObj.plan(self.X_train[:,windl], Nd, Kd, Jd)
        
        # Compute Type-2 NUFFT (type-2 means adjoint)
        nufft_result_type2 = NufftObj.adjoint(vec)
        
        #print(nufft_result_type2)
        
        return nufft_result_type2
        
    ###################################################################################
    
    def compute_ghat(self, vec):
        """
        Compute ghat, the product of the Fourier coefficients and the adjoint NFFT (pyNUFFT).
        
        Parameters
        ----------
        vec : ndarray
            The solution vector obtained by training the model with all feature subsets of length 2.
        
        Returns
        -------
        ghat_dict : dict
            The dict of the product of the Fourier coefficients and the adjoint NFFT (pyNUFFT) for all feature subsets.
        
        """
        #############################
        
        # initialize list of all subsets of window indices
        w_comb = []
        
        # create all subsets of window indices
        for i in range(len(self.wind)):
            w = self.wind[i]
            nwi = len(w)
            for j in range(nwi):
                w_comb.append(list(itertools.combinations(w, j+1)))
        
        #print("Windows:", self.wind)
        #print("All subsets windows:", w_comb)
        
        # flatten list of subsets of windows
        u_flat = [item for sublist in w_comb for item in sublist]
        #print("All subsets windows flat:", u_flat)
        
        # drop double subsets
        u = list(dict.fromkeys(u_flat))
        #print("All subsets windows unique:", u)
        
        ############################
        
        # create dict to save ghat for all possible feature combis
        ghat_dict = {}
        
        # initialize dict entries with zeros arrays of correct shape
        for l in range(len(u)):
            
            if len(u[l]) == 1:
                ghat_dict[u[l]] = np.zeros((self.M-1,))
                
            elif len(u[l]) == 2:
                ghat_dict[u[l]] = np.zeros((self.M-1,self.M-1))
            
            elif len(u[l]) == 3:
                ghat_dict[u[l]] = np.zeros([self.M-1,self.M-1,self.M-1])
        
        #############################
        
        # compute fft coefficients
        fft_coeff = self.compute_fft_coeff(u)
        
        for l in range(len(u)):
            
            w = u[l]
            
            d = len(w)
            
            # compute adjoint NFFT (pyNUFFT)
            nufft = self.compute_nufft(w, d, vec)
            
            ####################################################################################
            # fft_coeff do not depend on windows or data
            # nufft depends on windows and data
            pointwise_fkS = fft_coeff[d] * nufft
            
            # apply fft shift (rearrange rows and columns / tensors)
            # with shifted version allocation of zero frequency is in first row/column
            ghat = np.fft.fftshift(pointwise_fkS)
            
            ####################
            # assign ghat to subsets
        
            if len(w) == 1:
                ghat_dict[(w[0]),] = np.add(ghat_dict[(w[0],)], ghat[1:])
                
            elif len(w) == 2:
                # first index only
                ghat_dict[(w[0]),] = np.add(ghat_dict[(w[0]),], ghat[1:,0])
                # second index only
                ghat_dict[(w[1]),] = np.add(ghat_dict[(w[1]),], ghat[0,1:])
                # mixed/all
                ghat_dict[(w[0],w[1])] = np.add(ghat_dict[(w[0],w[1])], ghat[1:,1:])
                
            elif len(w) == 3:
                # first index only
                ghat_dict[(w[0],)] = np.add(ghat_dict[(w[0],)], ghat[1:,0,0])
                # second index only
                ghat_dict[(w[1],)] = np.add(ghat_dict[(w[1],)], ghat[0,1:,0])
                # third index only
                ghat_dict[(w[2],)] = np.add(ghat_dict[(w[2],)], ghat[0,0,1:])
                # first and second index
                ghat_dict[(w[0],w[1])] = np.add(ghat_dict[(w[0],w[1])], ghat[1:,1:,0])
                # first and third index
                ghat_dict[(w[0],w[2])] = np.add(ghat_dict[(w[0],w[2])], ghat[1:,0,1:])
                # second and third index
                ghat_dict[(w[1],w[2])] = np.add(ghat_dict[(w[1],w[2])], ghat[0,1:,1:])
                # all
                ghat_dict[(w[0],w[1],w[2])] = np.add(ghat_dict[(w[0],w[1],w[2])], ghat[1:,1:,1:])
    
    
        return ghat_dict
    
    ########################################################################################
    
    def gsi_sorted(self, vec):
        """ 
        Compute the Global Sensitivity Indices.

        Parameters
        ----------
        vec : ndarray
            The solution vector obtained by training the model with all feature subsets of length 2.
        
        Returns
        -------
        gsi_sorted : dict
            The dict of Global Sensitivity Indices for the feature subsets in descending order.
        
        """
        ##########################
        
        #print("Original windows:", self.windows)
        # sort windows so that indices are ascending
        self.wind = [sorted(item) for item in self.windows]
        #print("Sorted windows:", self.wind)
        
        ###########################
        
        # setup gsi computation

        if self.fastadj_setup == "fine":
            self.M = 64
        elif self.fastadj_setup == "default":
            self.M = 32
        elif self.fastadj_setup == "rough":
            self.M = 16
        
        ###########################
        
        # compute ghat
        ghat_dict = self.compute_ghat(vec)
        
        ###########################
        # compute GSI
        
        # initialize variance
        var = 0.0
        
        # initialize dict for GSI
        gsi_dict = {}
                
        for key in ghat_dict:
            gsi_dict[key] = np.sum(np.abs(ghat_dict[key])**2)
            var += gsi_dict[key]
            
        # divide every value in gsi_dict by var
        for key in gsi_dict:    
            gsi_dict[key] = gsi_dict[key] / var
        
        #print("gsi_dict:", gsi_dict)
        #print("sum gsi:", sum(gsi_dict.values()))
        
        # sort subsets by descending GSI ranking
        gsi_sorted = dict(sorted(gsi_dict.items(), key=operator.itemgetter(1), reverse=True))
        #print("Sorted GSI:", gsi_sorted)
        
        return gsi_sorted
                
