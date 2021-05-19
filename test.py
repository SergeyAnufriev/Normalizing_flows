import torch.distributions as distrib
import torch

from densities import sample_gaus_mixture
from flows import AffineCouplingFlow,ReverseFlow,Norm_flow_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

w_list     = [0.2,0.1,0.7]
covar_list = [[[0.2,0],[0,0.2]]]*3
means      = [[10,10],[0,0],[-5,-5]]

n_samples  = 32

'''Data sample'''
x = torch.tensor(sample_gaus_mixture(w_list,means,covar_list,n_samples),device=device,dtype=torch.float32)

'''Latent var sample'''
ref_distr    = distrib.MultivariateNormal(torch.zeros(2), torch.eye(2))
z            = ref_distr.sample(sample_shape=(32,))

'''Single Affine Layer'''
model_affine = AffineCouplingFlow(2)
'''20-22 Multi Affine Layer with reverse in between'''
Blocks       = [AffineCouplingFlow,ReverseFlow]
model_mult   = Norm_flow_model(2,Blocks,6,ref_distr)


print('Single layer test')
print('Inverse affine call,f_inv(x)={}'.format(model_affine._inverse(x)))
print('Forward affine call,f(z)={}'.format(model_affine(z)))
print('log_det_df_z_0={}'.format(model_affine.log_abs_det_jacobian(z)))

print('Multi flow test')
z_0, list_logdet_jacob = model_mult(x)
print('mult flow z_0={}'.format(z_0))
print('mult flow, jacob list,f(z)={}'.format(list_logdet_jacob))