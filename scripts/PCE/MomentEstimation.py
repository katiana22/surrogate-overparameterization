import numpy as np

def pce_mean(pce):
    """
    Estimate the mean value of the QoI using its PCE approximation.
    
    **Inputs**
    
    * **pce** (`PCE object`)
        PCE approximation of the QoI.
    
    **Output**
        (`float` or `1darray`) Mean value of the QoI. 
    """
    if pce.coefficients is None:
        ValueError('PCE coefficients have not yet been computed.')
    else:
        if pce.bias is None:
            return pce.coefficients[0, :]
        else:
            return pce.coefficients[0, :] + pce.bias
        
def pce_variance(pce):
    """
    Estimate the variance of the QoI using its PCE approximation.
    
    **Inputs**
    
    * **pce** (`PCE object`)
        PCE approximation of the QoI.
    
    **Output**
        (`float` or `1darray`) Variance of the QoI. 
    """
    if pce.coefficients is None:
        ValueError('PCE coefficients have not yet been computed.')
    else:
        return np.sum(pce.coefficients[1:]**2, axis=0)