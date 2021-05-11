import numpy as np

w_list     = [0.2,0.1,0.7]
covar_list = [[[0.2,0],[0,0.2]]]*3
means      = [[1,0],[0,0],[0,1]]

def sample_gaus_mixture(w_list,m_list,covar_list,n_samples):

    '''P(X) = w1*N(X|m1,var1)+w2*N(X|m2,var2)+w3*N(X|m3,var3)
    w_list   = [w1,w2,w3]
    var_list = [var1,var2,var3]
    n_sample - number samples from P(X)'''

    mixture_idx = np.random.choice(3,size=n_samples,p= w_list)

    samples_x = []
    samples_y = []

    for i in mixture_idx:
        xy = np.random.multivariate_normal(m_list[i], covar_list[i], (1, 1))[0][0]
        samples_x.append(xy[0])
        samples_y.append(xy[1])

    return np.array([samples_x,samples_y]).T

print(sample_gaus_mixture(w_list,means,covar_list,1000))



