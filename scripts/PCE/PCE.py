import numpy as np
from ChaosPolynomials import ChaosPolynomial1d, ChaosPolynomialNd


class PolyChaosExp:
    """
    Class for the Polynomial Chaos Expansion.
    
    **Inputs**
    
    * **dist** (`UQpy distribution object`)
        1d or Nd probability distribution characterizing the random input 
        parameter(s)
        
    **Attributes**
    
    * **dist** (`UQpy distribution object`)
        1d or Nd probability distribution characterizing the random input 
        parameter(s)
        
    * **n_inputs** (`int`):
        Number of input parameters
    
    * **n_outputs** (`int`):
        Dimensions of the QoI
    
    * **midx_set** (`ndarray`):
        Multiindex set the rows of which correspond to the orders of the 1d 
        chaos polynomials that form the Nd chaos polynomial.
        
    * **n_polys** (`int`):
        Size of polynomial basis, equiv., number of PCE terms
        
    * **coefficients** (`ndarray`):
        PCE coefficients
        
    * **bias** (`float` or `1darray`):
        Bias term in case LASSO or ridge regression are employed to estimate 
        the PCE coefficients
        
    * **poly_basis** (`list`):
        Contains the 1d or Nd chaos polynomials that form the PCE basis
        
    * **design_matrix** (`ndarray`):
        Matrix containing the evaluations of the PCE basis on the experimental 
        design that has been used to fit the PCE coefficients
        
    * **exp_design_in** (`ndarray`):
        Realizations of the random parameter in the experimental design that 
        has been used to fit the PCE coefficients
    
    * **exp_design_out** (`ndarray`):
        Model outputs for the random parameter realizations of the 
        experimental design that has been used to fit the PCE coefficients
    
    **Methods**
    
    * **eval_basis**
        Evaluates the polynomials of the PCE basis on a given data set of 
        input parameter realizations
        
    * **predict**
        Evaluates the PCE on a given data set of input parameter realizations
    """
    def __init__(self, dist):
        self.dist = dist
        try:
            self.n_inputs = len(dist.marginals)
        except:
            self.n_inputs = 1
        self.n_outputs = None
        self.midx_set = None 
        self.n_polys = None 
        self.coefficients = None
        self.bias = None
        self.poly_basis = None
        self.design_matrix = None
        self.exp_design_in = None
        self.exp_design_out = None
    
    def eval_basis(self, xx):
        """
        Evaluate the polynomial basis given a data set xx with 
        dimensions (n_samples x n_inputs) which contains realizations of the 
        input random parameters. The dimensions of the resulting matrix 
        are (n_samples x n_polys).
        
        **Input**
        
        * **xx** (`ndarray`):
            Data set upon which the basis polynomials will be evaluated
            
        **Output**
            (n_samples x n_polys) matrix containing the evaluations of the PCE 
            polynomials on the data set points
        """
        
        if self.n_inputs == 1:
            n_samples = len(xx)
        else:
            try: # case: 2d array, n_samples x n_inputs
                n_samples, n_inputs = np.shape(xx)
            except: # case: 1d array, 1 x n_inputs
                n_samples = 1
                n_inputs = len(xx)
                xx = xx.reshape(n_samples, n_inputs)
            # check if dimensions agree
            if n_inputs != self.n_inputs:
                raise ValueError('Dimensions do not agree!')
        
        # construct polynomial basis if not available        
        if self.poly_basis is None:
            raise ValueError("PCE polynomial basis is empty!")
        
        # construct evaluation matrix
        eval_matrix = np.empty([n_samples, self.n_polys])
        for ii in range(self.n_polys):
            eval_matrix[:, ii] = self.poly_basis[ii].evaluate(xx)
        
        return eval_matrix

    
    def predict(self, xx):
        """
        PCE predictions for the input data xx.
        
        **Input**
        
        * **xx** (`ndarray`):
            Data set upon which the PCE will be evaluated
            
        **Output**
            (`1darray`) Vector containing the evaluations of the PCE on the  
            data set points
        """
        D = self.eval_basis(xx)
        if self.bias is None:
            return D.dot(self.coefficients)
        else:
            return D.dot(self.coefficients) + self.bias