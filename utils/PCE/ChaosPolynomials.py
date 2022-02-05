import math
import numpy as np
from scipy.special import eval_hermitenorm, eval_legendre 
from StandardizeData import standardize_normal, standardize_uniform


def hermite_eval(x, k, n_dist):
    """
    Evaluates the Hermite 'probabilistic' polynomial of order k on the data 
    set x. The Hermite polynomial is ORTHONORMAL in (-inf, inf) w.r.t. the 
    Gaussian PDF 1/sqrt(2*pi*std) * exp( -(x-mean)**2 / (2*std**2) ).
    
    **Inputs:**

    * **x** (`1darray`):
        Points upon which the Hermite polynomial will be evaluated.

    * **k** (`int`):
        polynomial order.
        
    * **n_dist** (`UQpy distribution object`):
        Normal distribution that generated the evaluation points in `x`
            
    **Output/Return:**
    
    * **h** (`1darray`):
        Evaluations of the Hermite polynomial.
    """

    # impose numpy array type
    x = np.array(x).flatten()
    
    # normalize data
    x_normed = standardize_normal(x, n_dist)
    
    # evaluate standard Hermite polynomial, orthogonal w.r.t. the PDF of N(0,1)
    h = eval_hermitenorm(k, x_normed)
    
    # normalization constant
    st_herm_norm = np.sqrt(math.factorial(k) )

    h = h/st_herm_norm
    
    return h


def legendre_eval(x, k, u_dist):
    """
    Evaluates the Legendre polynomial of order k on the data set x. The 
    Legendre polynomial is ORTHONORMAL in [a,b] w.r.t the uniform PDF 1/(b-a).
    
    **Inputs:**

    * **x** (`1darray`):
        Points upon which the Legendre polynomial will be evaluated.

    * **k** (`int`):
        polynomial order.
        
    * **u_dist** (`UQpy distribution object`):
        Uniform distribution that generated the evaluation points in `x`
            
    **Output/Return:**
    
    * **h** (`1darray`):
        Evaluations of the Legendre polynomial.
    """
    
    # impose numpy array type
    x = np.array(x).flatten()
    
    # normalize data 
    x_normed = standardize_uniform(x, u_dist)
    
    # evaluate standard Legendre polynomial, i.e. orthogonal in [-1,1] with 
    # PDF = 1 (NOT 1/2!!!)
    l = eval_legendre(k, x_normed)
    
    # normalization constant
    st_lege_norm = np.sqrt(2/(2*k+1))

    # multiply by sqrt(2) to take into account the pdf 1/2
    l = np.sqrt(2) * l/st_lege_norm

    return l
    

class ChaosPolynomial1d():
    """
    Class for univariate Wiener-Askey chaos polynomials.

    **Inputs:**

    * **dist** (`UQpy distribution object`):
        Probability distribution.

    * **order** (`int`):
        Polynomial degree.
    
    **Attributes:**
        
    * **order** (`int`):
        Polynomial order.
        
    * **dist** (`UQpy distribution`):
        Probability distribution that defines the orthonormality of the 1d
        chaos polynomial.
        
    * **dist_type** (`str`):
        Type of distribution, e.g. 'Uniform', 'Normal', etc.
        
    * **poly_type** (`str`)
        Type of polynomial, e.g. 'Hermite', 'Legendre', etc.
    
    **Methods:**
    
    * *evaluate*:
        Evaluates the 1d chaos polynomial on a set of data points.
    """
    
    def __init__(self, dist, order):
        self.order = order
        self.dist = dist
        self.dist_type = type(dist).__name__
        if self.dist_type == 'Uniform':
            self.poly_type = 'Legendre'
        elif self.dist_type == 'Normal':
            self.poly_type = 'Hermite'
        else:
            raise NameError('Distribution not supported!')
        
    def evaluate(self, eval_data):
        """
        Evaluates the 1d chaos polynomial on the given data set. 
        
        **Inputs:**
        
        * **eval_data** (`1darray`):
            Points upon which the 1d chaos polynomial will be evaluated.
        
        **Outputs:**
        
        * **evals** (`1darray`):
            Evaluations of the 1d chaos polynomial.
        """
        eval_data = np.array(eval_data).flatten()
        if self.dist_type == 'Uniform':
            evals = legendre_eval(eval_data, self.order, self.dist)
        elif self.dist_type == 'Normal': 
            evals =  hermite_eval(eval_data, self.order, self.dist)
        else: 
            raise ValueError('Distribution not supported!')
        return evals


class ChaosPolynomialNd():
    """
    Class for multivariate Wiener-Askey chaos polynomials.

    **Inputs:**

    * **jdist** (`UQpy distribution object`):
        Joint probability distribution.

    * **midx** (`list` or `1darray`):
        Polynomial multi-degree (multi-index).
            
    **Attibutes:**
    
    * **jdist** (`UQpy distribution object`):
        Joint probability distribution.
        
    * **midx** (`list` or `1darray`):
        Polynomial multi-degree (multi-index).
    
    * **marginals** (`list`):
        Marginal distributions of the joint distribution.
    
    * **polynomials1d** (`list`):
        1d chaos polynomials forming the Nd chaos polynomial.
        
    **Methods**
    
    * **evaluate**
        Evaluates the Nd chaos polynomial on a set of data points.
    """
    
    def __init__(self, jdist, midx):
        self.midx = midx
        self.jdist = jdist
        marginals = jdist.marginals
        N = len(midx) # dimensions
        self.polynomials1d = [ChaosPolynomial1d(marginals[n], midx[n]) 
                              for n in range(N)]
        
    def evaluate(self, eval_data):
        """
        Evaluate Nd chaos polynomial on the given data set.
        
        **Inputs:**
        
        * **eval_data** (`ndarray` or `1darray`):
            Points upon which the Nd chaos polynomial will be evaluated.
        
        **Outputs:**
        
        * **evals** (`1darray`):
            Evaluations of the Nd chaos polynomial.
        """
        try: # case: 2d array, K x N, N being the number of dimensions
            K, N = np.shape(eval_data)
        except: # case: 1d array, 1 x N, N being the number of dimensions
            K = 1
            N = len(eval_data)
            eval_data = eval_data.reshape(K,N)
        
        # Store evaluations of 1d polynomials in a KxN matrix. Each column has 
        # the evaluations of the n-th 1d polynomial on the n-th data column, 
        # i.e. on the values of the n-th parameter 
        eval_matrix = np.empty([K,N])
        for n in range(N):
            eval_matrix[:,n] = self.polynomials1d[n].evaluate(eval_data[:,n])
        
        # The output of the multivariate polynomial is the product of the 
        # outputs of the corresponding 1d polynomials
        return  np.prod(eval_matrix, axis=1)
