import torch.distributions as distrib
import torch

from densities import sample_gaus_mixture
from flows import AffineCouplingFlow,ReverseFlow,Norm_flow_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

w_list     = [0.2,0.1,0.7]
covar_list = [[[0.2,0],[0,0.2]]]*3
means      = [[10,10],[0,0],[-5,-5]]

n_samples  = 32


x = torch.tensor(sample_gaus_mixture(w_list,means,covar_list,n_samples),device=device,dtype=torch.float32)


ref_distr    = distrib.MultivariateNormal(torch.zeros(2), torch.eye(2))
z            = ref_distr.sample(sample_shape=(32,))

model_affine = AffineCouplingFlow(2,n_hidden=256,n_layers=3)


print(model_affine.shift_log_scale)

Blocks       = [AffineCouplingFlow,ReverseFlow]*5+[AffineCouplingFlow]
print(Blocks)
model_mult   = Norm_flow_model(2,Blocks,ref_distr)


print('Single layer test')
print('Inverse affine call,f_inv(x)={}'.format(model_affine._inverse(x)))
print('Forward affine call,f(z)={}'.format(model_affine(z)))
print('log_det_df_z_0={}'.format(model_affine.log_abs_det_jacobian(z)))

'''

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

noisy_moons = datasets.make_moons(n_samples=128, noise=.05)[0].astype(np.float32)
X = StandardScaler().fit_transform(noisy_moons)

plt.scatter(X[:, 0], X[:, 1], s=10, color='red')
plt.show()


'''
print('Multi flow test')
z_0, list_logdet_jacob = model_mult(x)
print('mult flow z_0={}'.format(z_0))
print('mult flow, jacob list,f(z)={}'.format(list_logdet_jacob))


'''
print('test first mult_flow inverse iteration')
flow_k      = model_mult.bijectors[-1]
z_k_1       = flow_k._inverse(x)
log_abs_det_k_1 = flow_k.log_abs_det_jacobian(z_k_1)

print('z_k_1={}'.format(z_k_1))
print('x={}'.format(x))
print('log_abs_det={}'.format(log_abs_det_k_1))


print('test second mult_flow inverse iteration')
flow_k_1   = model_mult.bijectors[-2]
z_k_2      = flow_k_1._inverse(z_k_1)
log_abs_det_k_2 = flow_k_1.log_abs_det_jacobian(z_k_2)

print('z_k_2={}'.format(z_k_2))
print('log_abs_det_={}'.format(log_abs_det_k_2))

'''
