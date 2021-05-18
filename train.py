from flows import AffineCouplingFlow
import torch.distributions as distrib
import torch
import numpy as np



from flows import ReverseFlow

flow = ReverseFlow(2)

z = torch.rand((4,2))

print('original flow',z)
print('reversed flow',flow(z))

print('go back',flow._inverse(flow(z)))

