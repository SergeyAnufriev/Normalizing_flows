import torch

def loss(density, z0, log_jacobians):

    '''Inputs:
       density (p_z0) : prior distribution density (the simple one)
       z0             : transformed real data sample x by sequence of
                        inverse flows f_inv_1 * ... f_inv_(k-1)* f_inv_k (x)
       log_jacobians  : list [df_k

       '''

    sum_of_log_jacobians = sum(log_jacobians)
    return (-sum_of_log_jacobians - torch.log(density(z0) + 1e-9)).mean()
