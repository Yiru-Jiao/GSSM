'''
This script is reused from UnifiedConflictDetection https://github.com/Yiru-Jiao/UnifiedConflictDetection
'''

import os
import sys
import torch
import gpytorch
import numpy as np
from tqdm import tqdm
from scipy.special import erf
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src_data_preparation.represent_utils.coortrans import coortrans
coortrans = coortrans()


# SVGP model: Sparse Variational Gaussian Process
class SVGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):

        # Determine variational distribution and strategy
        # Number of inducing points is better to be smaller than 1000 to speed up the training
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
            )
        super(SVGP, self).__init__(variational_strategy)

        # Mean module
        self.mean_module = gpytorch.means.ConstantMean()

        # Kernel module
        mixture_kernel = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=10, ard_num_dims=10)
        rbf_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel(ard_num_dims=10))
        self.covar_module = mixture_kernel + rbf_kernel

        # To make mean positive
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        mean_x = self.mean_module(x)
        mean_x = self.softplus(mean_x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def define_model(num_inducing_points, device):
    # Create representative points for the input space
    # This is defined when training. Don't change it when applying the model.
    inducing_points = np.concatenate([np.random.uniform(4.,12.,(num_inducing_points,1)), # length_ego
                                      np.random.uniform(4.,12.,(num_inducing_points,1)), # length_sur
                                      np.random.uniform(-1,1,(num_inducing_points,1)), # hx_sur
                                      np.random.uniform(-1,1,(num_inducing_points,1)), # hy_sur
                                      np.random.uniform(0.,20.,(num_inducing_points,1)), # delta_v
                                      np.random.uniform(0.,400.,(num_inducing_points,1)), # delta_v2
                                      np.random.uniform(0.,3000.,(num_inducing_points,1)), # v_ego2
                                      np.random.uniform(0.,3000.,(num_inducing_points,1)), # v_sur2
                                      np.random.uniform(-5.5,5.5,(num_inducing_points,1)), # acc_ego
                                      np.random.uniform(-np.pi,np.pi,(num_inducing_points,1))], # rho
                                      axis=1)
    inducing_points = torch.from_numpy(inducing_points).float()

    # Define the model
    model = SVGP(inducing_points)
    # Define the likelihood of the model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    ## Load trained model
    model_path = 'src_safety_evaluation/reuse_ucd'
    model.load_state_dict(torch.load(f'{model_path}/model_52epoch.pth', map_location=torch.device(device), weights_only=True))
    likelihood.load_state_dict(torch.load(f'{model_path}/likelihood_52epoch.pth', map_location=torch.device(device), weights_only=True))
    model.eval()
    likelihood.eval()
    model = model.to(device)
    likelihood = likelihood.to(device)
    print('Model loaded successfully.')

    return model, likelihood


def lognormal_cdf(x, mu, sigma):
    return 1/2+1/2*erf((np.log(x)-mu)/sigma/np.sqrt(2))


def extreme_cdf(x, mu, sigma, n=10):
    return (1-lognormal_cdf(x,mu,sigma))**n


def UCD(data, device):
    # Mirror the coordinates as the model is trained on highD where the y-axis points downwards
    data = data.rename(columns={'x_ego':'y_ego', 'y_ego':'x_ego', 'x_sur':'y_sur', 'y_sur':'x_sur',
                                'vx_ego':'vy_ego', 'vy_ego':'vx_ego', 'vx_sur':'vy_sur', 'vy_sur':'vx_sur',
                                'hx_ego':'hy_ego', 'hy_ego':'hx_ego', 'hx_sur':'hy_sur', 'hy_sur':'hx_sur'})
    data['psi_ego'] = coortrans.angle(1, 0, data['hx_ego'], data['hy_ego'])
    data['psi_sur'] = coortrans.angle(1, 0, data['hx_sur'], data['hy_sur'])

    ## Transform coordinates and formulate input data
    data['delta_v'] = np.sqrt((data['vx_ego']-data['vx_sur'])**2 + (data['vy_ego']-data['vy_sur'])**2)
    data['delta_v2'] = data['delta_v']**2
    data['v_ego2'] = data['v_ego']**2
    data['v_sur2'] = data['v_sur']**2
    data_view_ego = coortrans.transform_coor(data, view='i')
    heading_sur = data_view_ego[['target_id','time','hx_sur','hy_sur']]
    data_relative = coortrans.transform_coor(data, view='relative')
    rho = coortrans.angle(1, 0, data_relative['x_sur'], data_relative['y_sur']).reset_index().rename(columns={0:'rho'})
    rho[['target_id','time']] = data_relative[['target_id','time']]
    interaction_context = data.drop(columns=['hx_sur','hy_sur']).merge(heading_sur, on=['target_id','time']).merge(rho, on=['target_id','time'])
    features = ['length_ego','length_sur','hx_sur','hy_sur','delta_v','delta_v2','v_ego2','v_sur2','acc_ego','rho']
    interaction_context = interaction_context[features+['event_id','target_id','time']].sort_values(['target_id','time'])
    data_relative = data_relative.merge(interaction_context[['target_id','time']], on=['target_id','time']).sort_values(['target_id','time'])
    proximity = np.sqrt(data_relative['x_sur']**2 + data_relative['y_sur']**2).values

    ## Load trained model
    model, likelihood = define_model(100, device)

    ## Compute mu_list, sigma_list
    chunk_size = 2000
    mu_list, sigma_list = [], []
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for chunk in tqdm(range(0, len(interaction_context), chunk_size), desc='Inferring', ascii=True, dynamic_ncols=False, miniters=100):
            chunk_data = interaction_context[features].values[chunk:chunk+chunk_size]
            f_dist = model(torch.Tensor(chunk_data).to(device))
            y_dist = likelihood(f_dist)
            mu, sigma = y_dist.mean.cpu().numpy(), y_dist.variance.sqrt().cpu().numpy()
            mu_list.append(mu)
            sigma_list.append(sigma)
    mu_list = np.concatenate(mu_list)
    sigma_list = np.concatenate(sigma_list)

    # Modify mu when ego and sur are leaving each other
    leaving = (interaction_context['rho'].values<0)
    mu_list[leaving] = proximity[leaving]

    max_intensity = np.log(0.5)/np.log(1-lognormal_cdf(proximity, mu_list, sigma_list)+1e-6)
    max_intensity = np.maximum(1., max_intensity)

    results = interaction_context[['event_id','target_id','time']].copy()
    results['proximity'] = proximity
    results['mu'] = mu_list
    results['sigma'] = sigma_list
    results['intensity'] = max_intensity
    return results

