def loss(density, z0, log_jacobians):

    '''Inputs:
       density (p_z0) : prior distribution density (the simple one)

       z0             : transformed real data sample x by sequence of
                        inverse
                        z_0= flows f_inv_1 * ... f_inv_(k-1)* f_inv_k (x)

       log_jacobians  : list [df_n/dz_(n-1),....,df_1/dz_0]
       '''

    sum_of_log_jacobians = sum(log_jacobians)
    return (sum_of_log_jacobians - density.log_prob(z0)).mean()
