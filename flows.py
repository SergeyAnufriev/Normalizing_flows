import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distrib
import torch.distributions.transforms as transform

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


class Some_flow(Flow):
    def __init__(self,*args,**kwargs):
        super(Some_flow,self).__init__()
        '''Flow params, modules, e.t.c'''

    def _call(self,z):
        ''' Forward transformation
          Input: z
          Output: f(z)=x
          where f - the flow transformation'''
        return

    def _inverse(self,x):
        ''' Inverse transformation
          Input: x
          Output: f^-1(x)=z'''
        return

    def log_abs_det_jacobian(self, x,y):
        '''Input: z
          Output:log_det(df_dz)'''
        return


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

       Output:log P(X)'''

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
