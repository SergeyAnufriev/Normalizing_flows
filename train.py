import torch.distributions as distrib
import torch
import torch.nn as nn
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


'''
optimizer = optim.Adam(model.parameters(), lr=2e-3)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9999)
for i in range(epochs):
    x = torch.tensor(sample_gaus_mixture(w_list,means,covar_list,512),dtype=torch.float32,device=device)
    z_0,list_log_det = model(x)
    l                = loss(ref_distr,z_0,list_log_det)
    optimizer.zero_grad()
    l.backward()
    optimizer.step()
    scheduler.step()
    if i%plot_it:
        print('Loss',l)
'''
