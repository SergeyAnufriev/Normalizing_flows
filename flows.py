import torch
import torch.nn as nn
import torch.nn.functional as F
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


    def _call(self,x):
        ''' Forward transformation
          Input:  Latent variable from known prior distr
          Output: Generated data like from target distr
          '''

        '''split input by dimention, Paper equation 4'''
        x_d, x_D = x[:, :self.d], x[:, self.d:]
        '''Paper equation 5'''
        X_transformed = x_D * torch.exp(self.s(x_d)) + self.t(x_d)
        return torch.cat((x_d, X_transformed), dim = 1)

    def _inverse(self,y):
        ''' Inverse transformation
          Input:  Data from target distribution
          Output: Data in latent space with known prior distr'''

        '''split input by dimention, paper equation 8a'''
        y_d, y_D = y[:, :self.d], y[:, self.d:]
        '''paper equation 8b'''
        y_transformed = (y_D-self.t(y_d))/self.s(y_d)
        return torch.cat((y_d, y_transformed), dim = 1)

    def log_abs_det_jacobian(self,x,y):
        '''Input: real data x
          Output:log_det(df_dz)'''
        x_d = x[:,:self.d]
        '''summation under equation 6'''
        return -torch.sum(torch.abs(self.s(x_d)))




    '''
    def log_abs_det_jacobian(self, x, y):
        """
        Computes the log det jacobian `log |dy/dx|` given input and output.
        """
        raise NotImplementedError
    '''

class ReverseFlow(Flow):

    def __init__(self, dim):
        super(ReverseFlow, self).__init__()
        self.permute = torch.arange(dim-1, -1, -1)
        self.inverse = torch.argsort(self.permute)

    def _call(self, z):
        return z[:, self.permute]

    def _inverse(self, z):
        return z[:, self.inverse]

    def log_abs_det_jacobian(self, z):
        return torch.zeros(z.shape[0], 1)


class Norm_flow_model(nn.Module):

      '''Main model class, assambled
      by series of flows defined above'''

      def __init__(self,bijectros,density):
          super().__init__()

          self.bijectors = nn.Modulelist(bijectros)
          '''list of flow objects
          representating f1,f2,f3,...,fn '''

          self.transform = transform(bijectros)
          '''Normilizing flow probability transformation'''

          self.base_density  = density
          '''density of variable z, with known density function, usually mult var gauss with diag covar matrix'''

          self.final_density = distrib.TransformedDistribution(density, self.transforms)
          '''density of target variable x,obtained by transfroming z WITH f1,f2,f3,...,fn'''

          self.log_det = []

      def forward(self,z):

          '''self.log_det is a list of
          df1_dz1,df2_dz2,...,df_n_dz_n
          where z_n = f_n(z_(n-1))
          '''

          return 'f1(f2(f3,...,fn(z))), self.log_det'


def loss(density, zk, log_jacobians):

    '''Nomilising flow Loss function
       Input: z_k         - trnsformed z by f1,f2,f3,...,fn
            log_jacobians - above self.log_det
            density       - density function of the target distribution
       Returns -log P_q0(z_0):
       '''

    sum_of_log_jacobians = sum(log_jacobians)
    return (-sum_of_log_jacobians - torch.log(density(zk) + 1e-9)).mean()


'''To do

   a) Implement:
   
   1) Affine Coupling Flow
   2) Radial flow 
   3) Reverse & shuffle flow 
   4) Plannar flow 
   5) Batch norm flow 
   6) Autoregressive flow 
   
   b) 
   Try to compose a flow from multiple above
   
   '''
