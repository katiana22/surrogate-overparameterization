def standardize_uniform(x, u_pdf):
        """
        Transform data set from the uniform distribution U(a,b) to the 
        uniform distribution U(-1,1).

        **Input:**

        * **x** (`ndarray`)
            Input data generated from a uniform distribution.

        * **u_pdf** (`UQpy distribution object`)
            Uniform distribution as defined in the UQpy package.

        **Output/Returns:**

        `ndarray`
            Standardized data following U(-1,1).

        """
        loc = get_loc(u_pdf) # loc = lower bound of uniform distribution
        scale = get_scale(u_pdf) 
        upper = loc + scale # upper bound = loc + scale
        return (2*x - loc - upper) / (upper-loc)
    

def standardize_normal(x, n_pdf):
        """
        Transform data set from a normal distribution N(m, s) to the 
        standard normal distribution N(0,1).

        **Input:**

        * **x** (`ndarray`)
            Input data generated from a normal distribution.

        * **mean** (`UQpy distribution object`)
            Normal distribution as defined in the UQpy package.


        **Output/Returns:**

        `ndarray`
            Standardized data  following N(0,1).

        """
        loc = get_loc(n_pdf) # mean of normal distribution
        scale = get_scale(n_pdf) # std of normal distribution
        return (x - loc) / scale


def get_loc(dist):
    """
    Returns a `float` with the location of the UQpy distribution object.
    """
    m = dist.__dict__['params']['loc']
    return m


def get_scale(dist):
    """
    Returns a `float` with the scale of the UQpy distribution object.
    """
    s = dist.__dict__['params']['scale']
    return s