from ChaosPolynomials import ChaosPolynomial1d, ChaosPolynomialNd
from MultiIndexSets import td_multiindex_set, tp_multiindex_set


def construct_arbitrary_basis(pce, midx_set):
    """
    Create polynomial basis for a given multiindex set.
    
    **Inputs**
    
    * **midx_set** (`ndarray`):
        n_polys x n_inputs ndarray with the multiindices of the PCE basis
        
    * **pce** (`PolyChaosExp object`):
        Polynomial chaos expansion for which the multiindex set will be 
        generated.
        
    **Output**
    
    * **poly_basis** (`list`)
        List with the basis polynomials (ChaosPolynomial1d or 
        ChaosPolynomialNd objects)
    """
    # populate polynomial basis
    if pce.n_inputs == 1:
        poly_basis = [ChaosPolynomial1d(pce.dist, idx) for idx in midx_set]
    else:
        poly_basis = [ChaosPolynomialNd(pce.dist, idx) for idx in midx_set]
    # update attributes of PolyChaosExp object
    pce.midx_set = midx_set
    pce.n_polys = len(midx_set)
    pce.poly_basis = poly_basis


def construct_tp_basis(pce, max_degree):
    """
    Create tensor-product polynomial basis given the . 
    The size is equal to (max_degree+1)**n_inputs (exponential complexity).
    
    **Inputs**:
        
    * **max_degree** (`int`):
        Maximum polynomial degree of the 1d chaos polynomials
        
    * **pce** (`PolyChaosExp object`):
        Polynomial chaos expansion for which the multiindex set will be 
        generated.
        
    **Output**
    
    **poly_basis** (`list`)
        List with the basis polynomials (ChaosPolynomial1d or 
        ChaosPolynomialNd objects)
    """
    midx_set = tp_multiindex_set(pce.n_inputs, max_degree)
    construct_arbitrary_basis(pce, midx_set)
    
    
def construct_td_basis(pce, max_degree):
    """
    Create total-degree polynomial basis. 
    The size is equal to (total_degree+n_inputs)!/(total_degree!*n_inputs!) 
    (polynomial complexity).
    
    **Inputs**:
        
    * **max_degree** (`int`):
        Maximum polynomial degree of the 1d chaos polynomials
        
    * **pce** (`PolyChaosExp object`):
        Polynomial chaos expansion for which the multiindex set will be 
        generated.
        
    **Output**
    
    **poly_basis** (`list`)
        List with the basis polynomials (ChaosPolynomial1d or 
        ChaosPolynomialNd objects)
    """
    midx_set = td_multiindex_set(pce.n_inputs, max_degree) 
    construct_arbitrary_basis(pce, midx_set)
    