import torch
import torch.nn as nn
import torch.distributions as distrib
import torch.distributions.transforms as transform
from torch.distributions import constraints

'''Flow defined as following 
   z - random variable from known prob dist, usually multivar gaus with diagonal covar matrix
   x - target varibale from unknow distribution
   f - function z: -> x, flow transformation (bijective function)
   '''

class Flow(transform.Transform, nn.Module):
    '''Parent class to all flows classes
    allows:
    1) Transform prob. densities
    2) Training as nn_module'''
    def __init__(self):
        transform.Transform.__init__(self)
        nn.Module.__init__(self)
    # Init all parameters
    def init_parameters(self):
        '''Work in progress'''
        for param in self.parameters():
            param.data.uniform_(-0.01, 0.01)
    # Hacky hash bypass
    def __hash__(self):
        return nn.Module.__hash__(self)


class AffineCouplingFlow(Flow):
    '''paper - https://arxiv.org/pdf/1605.08803.pdf'''
    def __init__(self,dim, n_hidden=64, n_layers=3, activation=nn.ReLU):
        '''s and t share input only'''
        super(AffineCouplingFlow,self).__init__()
        '''d - first d dimentions go inside s and t networks, 
        the other dim-d get updated by s and t'''
        self.d  = dim//2
        ''' self.s - scale tranformation,  function from R^d to R^dim-d, where dim is the data size'''
        self.s = self.transform_net(self.d,dim-self.d,n_hidden,n_layers,activation)
        ''' self.t - translate transformation, function from R^d to R^dim-d'''
        self.t  = self.transform_net(self.d,dim-self.d,n_hidden,n_layers,activation)
        '''Init weight as in parent flow class'''
        self.init_parameters()
        '''domain and codomain of the transformation f'''
        self.domain   = constraints.real
        self.codomain = constraints.real

    def transform_net(self, nin, nout, nhidden, nlayer, activation):
        net = nn.ModuleList()
        for l in range(nlayer):
            net.append(nn.Linear(l==0 and nin or nhidden, l==nlayer-1 and nout or nhidden))
            net.append(activation())
        return nn.Sequential(*net)

    def _call(self,z):
        ''' Forward transformation
          Input:  Latent variable from known prior distr
          Output: Generated data like from target distr
          '''

        '''split input by dimention, Paper equation 4'''
        x_d, z_D = z[:, :self.d], z[:, self.d:]
        '''Paper equation 5'''
        z_transformed = z_D * torch.exp(self.s(x_d)) + self.t(x_d)
        return torch.cat((x_d, z_transformed), dim = 1)

    def _inverse(self,x):
        ''' Inverse transformation
          Input:  Data from target distribution
          Output: Data in latent space with known prior distr'''

        '''split input by dimention, paper equation 8a'''
        z_d, x_D = x[:, :self.d], x[:, self.d:]
        '''paper equation 8b'''
        x_transformed = (x_D-self.t(z_d))/self.s(z_d)
        return torch.cat((z_d, x_transformed), dim = 1)

    def log_abs_det_jacobian(self,*args):
        '''Input: Latent variable from known prior distr
          Output:log_det(df_dz), paper -> summation under equation 6'''
        z   = args[0]
        z_d = z[:,:self.d]
        '''summation under equation 6'''
        return torch.sum(self.s(z_d),dim=-1)


class ReverseFlow(Flow):

    def __init__(self, dim):
        super(ReverseFlow, self).__init__()
        self.permute = torch.arange(dim-1, -1, -1)
        self.inverse = torch.argsort(self.permute)

    def _call(self, z):
        return z[:, self.permute]

    def _inverse(self, z):
        return z[:, self.inverse]

    def log_abs_det_jacobian(self, *args):
        z = args[0]
        return torch.zeros(z.shape[0], 1)


class Norm_flow_model(nn.Module):

      '''Main model class, assambled
      by series of flows defined above'''
      '''dim - dimnention of z and x 
      blocks - reapeating flow functions (list of python classes, which represent
                                                            flow transformation)
      flow_length - number of blocks
      density - random number, i.e density function of z'''

      def __init__(self,dim,blocks,flow_length,density):
          super().__init__()

          '''List containing [f1,f2,f3,f4,...,f_n], where f is the flow transform'''
          bijectros = []

          for f in range(flow_length):
              for b_flow in blocks:
                  bijectros.append(b_flow(dim))

          '''list of flow objects
          represents [f_n,f_(n-1),...,f_1] sequence  '''
          self.bijectors = nn.Modulelist(bijectros.reverse())

          '''Normilizing flow probability transformation
          reprsents [f_1,f_2,...,f_n]'''
          self.transforms = transform.ComposeTransform(bijectros)

          '''density of variable z_0, with known density function, usually mult var gauss with diag covar matrix'''
          self.base_density  = density

          '''density of target variable z_n = f_n * f_(n_1) **** f_1(z_0)'''
          self.final_density = distrib.TransformedDistribution(density, self.transforms)

          '''list : [log_det_df_i/dz_(i-1) for i in range n] , where n is the flow number'''
          self.log_det = []

      def forward(self,x):
          '''Input: Real data sample X
             Output: z_0, [log_abs_det(df_i/df_z_(i-1) for i in range(k)],

             where z_0 transformed real data sample x by sequence of
             inverse flows z_0 = f_inv_1 * f_inv_2 * ... * f_inv_n (x)'''

          z = x
          for biject in self.bijectors:
              z = biject._inverse(z)
              self.log_det.append(biject.log_abs_det_jacobian(z))
          return z, self.log_det
