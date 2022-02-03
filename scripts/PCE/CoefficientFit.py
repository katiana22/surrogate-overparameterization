import numpy as np


def fit_lstsq(pce, xx, yy):
    """
    Computes the PCE coefficients with the numpy.linalg.lstsq solver.
    xx are realizations of the input parameters and yy the 
    corresponding observations regarding the quantity of interest.
    
    **Inputs**
    
    * **pce** (`PCE object`)
        Polynomial Chaos Expansion the coefficients of which will be evaluated
        by means of least squares regression.
        
    * **xx** (`ndarray`)
        Realizations of the input random parameters
        
    * **yy** (`ndarray`)
        Evaluations of the original model on xx
    """
    xx = np.array(xx)
    yy = np.array(yy)
    # update experimental design
    pce.exp_design_in = xx
    pce.exp_design_out = yy
    # compute and update design matrix
    D = pce.eval_basis(xx)
    pce.design_matrix = D
    # fit coefficients
    c, res, rank, sing = np.linalg.lstsq(D, yy, rcond=None)
    if c.ndim == 1:
        c = c.reshape(-1, 1)
    # update n_outputs and coefficients
    pce.n_outputs = np.shape(c)[1]
    pce.coefficients = c
    

def fit_lasso(pce, xx, yy, learning_rate=0.01, iterations=1000, 
              penalty=1):
    """
    Compute the PCE coefficients with the LASSO method.
    xx are realizations of the input parameters and yy the 
    corresponding observations regarding the quantity of interest.
    
    **Inputs**
    
    * **pce** (`PCE object`)
        Polynomial Chaos Expansion the coefficients of which will be evaluated
        by means of LASSO regression.
        
    * **xx** (`ndarray`)
        Realizations of the input random parameters
        
    * **yy** (`ndarray`)
        Evaluations of the original model on xx
    """
    xx = np.array(xx)
    yy = np.array(yy)
    # update experimental design
    pce.exp_design_in = xx
    pce.exp_design_out = yy
    # compute and update design matrix
    D = pce.eval_basis(xx)
    pce.design_matrix = D
    
    m, n = D.shape
    # in some 1D output problems the y array in python might be (n,) or 
    # (n,1)
    if yy.ndim == 1 or yy.shape[1] == 1:
        yy = yy.reshape(-1, 1)
        w = np.zeros(n).reshape(-1, 1)
        dw = np.zeros(n).reshape(-1, 1)
        b = 0
        for _ in range(iterations):
            y_pred = (D.dot(w) + b)

            for i in range(n):
                if w[i] > 0:
                    dw[i] = (-(2 * (D.T[i, :]).dot(yy - y_pred)) \
                             + penalty) / m
                else:
                    dw[i] = (-(2 * (D.T[i, :]).dot(yy - y_pred)) \
                             - penalty) / m

            db = - 2 * np.sum(yy - y_pred) / m

            w = w - learning_rate * dw
            b = b - learning_rate * db
    else:
        n_out_dim = yy.shape[1]
        w = np.zeros((n, n_out_dim))
        b = np.zeros(n_out_dim).reshape(1, -1)
        for _ in range(iterations):
            y_pred = (D.dot(w) + b)

            dw = (-(2 * D.T.dot(yy - y_pred)) - penalty) / m
            db = - 2 * np.sum((yy - y_pred), axis=0).reshape(1, -1) / m

            w = w - learning_rate * dw
            b = b - learning_rate * db
    pce.bias = b
    pce.coefficients = w
    pce.n_outputs = np.shape(w)[1]
    

def fit_ridge(pce, xx, yy, learning_rate=0.01, iterations=1000,
             penalty=1):
    """
    Compute the PCE coefficients with ridge regression.
    xx are realizations of the input parameters and yy the 
    corresponding observations regarding the quantity of interest.
    
    **Inputs**
    
    * **pce** (`PCE object`)
        Polynomial Chaos Expansion the coefficients of which will be evaluated
        by means of ridge regression.
        
    * **xx** (`ndarray`)
        Realizations of the input random parameters
        
    * **yy** (`ndarray`)
        Evaluations of the original model on xx

    """
    xx = np.array(xx)
    yy = np.array(yy)
    # update experimental design
    pce.exp_design_in = xx
    pce.exp_design_out = yy
    # compute and update design matrix
    D = pce.eval_basis(xx)
    pce.design_matrix = D
    
    m, n = D.shape
    # in some 1D output problems the y array in python might be (n,) or 
    # (n,1)
    if yy.ndim == 1 or yy.shape[1] == 1:
        yy = yy.reshape(-1, 1)
        w = np.zeros(n).reshape(-1, 1)
        b = 0
        for _ in range(iterations):
            y_pred = (D.dot(w) + b).reshape(-1, 1)

            dw = (-(2 * D.T.dot(yy - y_pred)) + (2 * penalty * w)) / m
            db = - 2 * np.sum(yy - y_pred) / m

            w = w - learning_rate * dw
            b = b - learning_rate * db
    else:
        n_out_dim = yy.shape[1]
        w = np.zeros((n, n_out_dim))
        b = np.zeros(n_out_dim).reshape(1, -1)
        for _ in range(iterations):
            y_pred = (D.dot(w) + b)

            dw = (-(2 * D.T.dot(yy - y_pred)) + (2 * penalty * w)) / m
            db = - 2 * np.sum((yy - y_pred), axis=0).reshape(1, -1) / m

            w = w - learning_rate * dw
            b = b - learning_rate * db
    pce.bias = b
    pce.coefficients = w
    pce.n_outputs = np.shape(w)[1]