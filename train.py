from flows import AffineCouplingFlow
import torch.distributions as distrib
import torch


Flow = AffineCouplingFlow(2)
q0 = distrib.MultivariateNormal(torch.zeros(2), torch.eye(2))
q1 = distrib.TransformedDistribution(q0,Flow)


print(q1.log_prob(torch.rand((2,2))))
