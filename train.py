import torch.distributions as distrib
import torch
import numpy as np
import torch.optim as optim

from densities import sample_gaus_mixture
from flows import AffineCouplingFlow,ReverseFlow,Norm_flow_model
from utils import loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs  = 10000
plot_it = 1000

Blocks    = [AffineCouplingFlow,ReverseFlow]
ref_distr = distrib.MultivariateNormal(torch.zeros(2), torch.eye(2))
model     = Norm_flow_model(2,Blocks,6,ref_distr)

w_list     = [0.2,0.1,0.7]
covar_list = [[[0.2,0],[0,0.2]]]*3
means      = [[10,10],[0,0],[-5,-5]]

n_samples  = 512

x = sample_gaus_mixture(w_list,means,covar_list,n_samples)
x = torch.tensor(x,device=device,dtype=torch.float32)

print('x',x)
model = AffineCouplingFlow(2)
print('model(x)',model(x))

print(model._inverse(x))
print(model.log_abs_det_jacobian(x))


'''
z, log_det = model(x)

print('x',x)
print('z',z)
print('log_det',log_det)

print(len(model.bijectors))

'''
