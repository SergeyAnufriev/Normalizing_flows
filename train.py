import torch.distributions as distrib
import torch
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from flows import AffineCouplingFlow,ReverseFlow,Norm_flow_model
from utils import loss

epochs  = 10000
plot_it = 1000
n_samples  = 128
learning_rate = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ref_distr    = distrib.MultivariateNormal(torch.zeros(2,device=device), torch.eye(2,device=device))
Blocks       = [AffineCouplingFlow,ReverseFlow]*5+[AffineCouplingFlow]
model        = Norm_flow_model(2,Blocks,ref_distr,device)
optimizer    = torch.optim.Adam(model.parameters(), lr=learning_rate)

print(model.bijectors[0].shift_log_scale)
'''
for i in range(epochs):
    optimizer.zero_grad()
    noisy_moons = datasets.make_moons(n_samples=128, noise=.05)[0].astype(np.float32)
    X = StandardScaler().fit_transform(noisy_moons)
    X = torch.FloatTensor(X,device=device)
    z_0,list_log_det = model(X)
    l                = loss(ref_distr,z_0,list_log_det)
    l.backward()
    optimizer.step()

    if i%plot_it==0:
        print('Loss',l)
'''
