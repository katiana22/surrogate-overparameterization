import numpy as np
from MomentEstimation import pce_variance


def pce_sobol_first(pce):
    """
    PCE estimates for the first order Sobol indices.
    
    **Inputs**
    
    * **pce** (`PCE object`)
        PCE approximation of the QoI.
    
    **Output**
        (`ndarray` or `1darray`) First order Sobol indices. 
    """
    if pce.coefficients is None:
        ValueError('PCE coefficients have not yet been computed.')
    else:
        # compute number of outputs
        n_outputs = np.shape(pce.coefficients)[1]
        # compute variance estimate
        variance = pce_variance(pce)
        # compute first-order Sobol index estimates
        sobol_f = np.zeros([pce.n_inputs, n_outputs])
        # take all multi-indices except 0-index
        idx_no_0 = np.delete(pce.midx_set, 0, axis=0) 
        for nn in range(pce.n_inputs): 
            # remove nn-th column
            idx_no_0_nn = np.delete(idx_no_0, nn, axis=1) 
            # we want the rows with all indices (except nn) equal to zero
            sum_idx_rows = np.sum(idx_no_0_nn, axis=1)
            zero_rows = np.asarray(np.where(sum_idx_rows==0)).flatten() + 1 
            variance_contribution = np.sum(pce.coefficients[zero_rows,:]**2, 
                                           axis=0)
            sobol_f[nn,:] = variance_contribution/variance
    return sobol_f    
    
           
def pce_sobol_total(pce):
    """
    PCE estimates for the first order Sobol indices.
    
    **Inputs**
    
    * **pce** (`PCE object`)
        PCE approximation of the QoI.
    
    **Output**
        (`ndarray` or `1darray`) Total order Sobol indices. 
    """
    if pce.coefficients is None:
        ValueError('PCE coefficients have not yet been computed.')
    else:
        # compute number of outputs
        n_outputs = np.shape(pce.coefficients)[1]
        # compute variance estimate
        variance = pce_variance(pce)
        # compute first-order Sobol index estimates
        sobol_t = np.zeros([pce.n_inputs, n_outputs])
        for nn in range(pce.n_inputs):
            # we want all multi-indices where the nn-th index is NOT zero
            idx_column_nn = np.array(pce.midx_set)[:,nn]
            nn_rows = np.asarray(np.where(idx_column_nn!=0)).flatten()
            variance_contribution = np.sum(pce.coefficients[nn_rows,:]**2, 
                                           axis=0)
            sobol_t[nn,:] = variance_contribution/variance
    return sobol_t


def pce_generalized_sobol_first(pce):
    """
    PCE estimates of generalized first order Sobol indices, which characterize
    the sensitivity of a vector-valued quantity of interest on the random 
    inputs.
    
    **Inputs**
    
    * **pce** (`PCE object`)
        PCE approximation of the QoI.
    
    **Output**
        (`1darray`) Generalized first order Sobol indices.
    """
    if pce.coefficients is None:
        ValueError('PCE coefficients have not yet been computed.')
    elif pce.n_inputs == 1:
        ValueError('Not applicable for scalar model outputs.')
    else:
        # compute variance and first order Sobol indices (elementwise)
        variance = pce_variance(pce)
        sobol_f = pce_sobol_first(pce)
        # retrieve elementwise variance contributions
        variance_contributions = sobol_f*variance
        # compute total variance (scalar value)
        tot_var = np.sum(variance)
        # compute total variance contribution per random input 
        # 1d array with length equal to n_inputs
        tot_var_contr_per_input = np.sum(variance_contributions, axis=1)
        # compute generalized first-order Sobol indices
        sobol_f_gen = tot_var_contr_per_input/tot_var
        return sobol_f_gen
        

def pce_generalized_sobol_total(pce):
    """
    PCE estimates of generalized total order Sobol indices, which characterize
    the sensitivity of a vector-valued quantity of interest on the random 
    inputs.
    
    **Inputs**
    
    * **pce** (`PCE object`)
        PCE approximation of the QoI.
    
    **Output**
        (`1darray`) Generalized total order Sobol indices.
    """
    if pce.coefficients is None:
        ValueError('PCE coefficients have not yet been computed.')
    elif pce.n_inputs == 1:
        ValueError('Not applicable for scalar model outputs.')
    else:
        # compute variance and first order Sobol indices (elementwise)
        variance = pce_variance(pce)
        sobol_t = pce_sobol_total(pce)
        # retrieve elementwise variance contributions
        variance_contributions = sobol_t*variance
        # compute total variance (scalar value)
        tot_var = np.sum(variance)
        # compute total variance contribution per random input 
        # 1d array with length equal to n_inputs
        tot_var_contr_per_input = np.sum(variance_contributions, axis=1)
        # compute generalized first-order Sobol indices
        sobol_t_gen = tot_var_contr_per_input/tot_var
        return sobol_t_gen