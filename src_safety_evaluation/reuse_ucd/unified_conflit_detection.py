'''
This script is reused from UnifiedConflictDetection https://github.com/Yiru-Jiao/UnifiedConflictDetection
'''

import os
import sys
import torch
import gpytorch
import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.special import erf
from torch.utils.data import Dataset, DataLoader
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src_posterior_inference.model import LogNormalNLL, SmoothLogNormalNLL
from src_data_preparation.represent_utils.coortrans import coortrans
coortrans = coortrans()
small_eps = 1e-6


# Create dataloader
class DataOrganiser(Dataset):
    def __init__(self, dataset, path_input):
        self.dataset = dataset
        self.path_input = path_input
        self.read_data()

    def __len__(self,):
        return len(self.idx_list)

    def __getitem__(self, idx):
        # idx is the index of items in the dataset
        int_ctxt = self.interaction_context.loc[self.idx_list[idx]].values
        int_ctxt = torch.from_numpy(int_ctxt).float()
        cur_spac = self.current_spacing.loc[self.idx_list[idx]]
        return int_ctxt, cur_spac

    def read_data(self,):
        features = pd.read_hdf(self.path_input + 'current_features_SafeBaseline_' + self.dataset + '.h5', key='features')
        self.idx_list = features['scene_id'].values
        features = features.set_index('scene_id')
        variables = ['l_ego','l_sur','combined_width',
                     'vy_ego','vx_sur','vy_sur','v_ego2','v_sur2','delta_v2','delta_v',
                     'psi_sur','rho']
        self.interaction_context = features[variables].copy()
        # log-transform spacing, and the spacing must be larger than 0
        if np.any(features['s']<=small_eps):
            print('There are spacings smaller than or equal to 0.')
            features.loc[features['s']<=small_eps, 's'] = small_eps
        self.current_spacing = np.log(features['s']).copy()
        # print data descriptions for inspection
        print(features[variables+['s']].describe().to_string())
        del features


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
        mixture_kernel = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=10, ard_num_dims=12)
        rbf_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel(ard_num_dims=12))
        self.covar_module = mixture_kernel + rbf_kernel

        # To make mean positive
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        mean_x = self.mean_module(x)
        mean_x = self.softplus(mean_x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class train_val_test():
    def __init__(self, device, num_inducing_points, path_input='./', path_output='./'):
        self.device = device
        self.path_input = path_input
        self.path_output = path_output
        os.makedirs(self.path_output, exist_ok=True)

        # Define model and likelihood
        self.define_model(num_inducing_points)


    def create_dataloader(self, batch_size, beta=5):
        self.batch_size = batch_size
        self.beta = beta

        # Create dataloader
        self.train_dataloader = DataLoader(DataOrganiser('train', self.path_input), batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(DataOrganiser('val', self.path_input), batch_size=self.batch_size, shuffle=True)
        print(f'Dataloader created, number of training samples: {len(self.train_dataloader.dataset)}\n')
       # Determine loss function
        self.loss_func = gpytorch.mlls.PredictiveLogLikelihood(self.likelihood, self.model, num_data=len(self.train_dataloader.dataset), beta=self.beta)


    def define_model(self, num_inducing_points):
        self.inducing_points = self.create_inducing_points(num_inducing_points)
        self.inducing_points = torch.from_numpy(self.inducing_points).float()

        # Define the model
        self.model = SVGP(self.inducing_points)

        # Define the likelihood of the model
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def create_inducing_points(self, num_inducing_points):
        # Create representative points for the input space
        inducing_points = pd.DataFrame({'l_ego': np.random.uniform(0.5, 16.2,num_inducing_points),
                                        'l_sur': np.random.uniform(0.5, 16.2,num_inducing_points),
                                        'combined_width': np.random.uniform(0.,2.5,num_inducing_points),
                                        'vy_ego': np.random.uniform(0.,45.,num_inducing_points),
                                        'vx_sur': np.random.uniform(-45.,45.,num_inducing_points),
                                        'vy_sur': np.random.uniform(-45.,45.,num_inducing_points),
                                        'v_ego2': np.random.uniform(0.,2000.,num_inducing_points),
                                        'v_sur2': np.random.uniform(0.,2000.,num_inducing_points),
                                        'delta_v2': np.random.uniform(0.,2000.,num_inducing_points),
                                        'delta_v': np.random.uniform(-45.,45.,num_inducing_points),
                                        'psi_sur': np.random.uniform(-np.pi,np.pi,num_inducing_points),
                                        'rho': np.random.uniform(-np.pi,np.pi,num_inducing_points)})
        return inducing_points.values

    def get_inducing_out(self, x, noise=0.01):
        inducing_points = x + noise*torch.randn_like(x, device=x.device)
        f_dist = self.model(inducing_points)
        y_dist = self.likelihood(f_dist)
        mu, var = y_dist.mean, y_dist.variance
        return mu, torch.log(torch.clamp(var, min=small_eps))

    def train_model(self, num_epochs=100, initial_lr=0.1):
        self.initial_lr = initial_lr

        # Move model and likelihood to device
        self.model = self.model.to(self.device)
        self.likelihood = self.likelihood.to(self.device)
        self.loss_func = self.loss_func.to(self.device)

        # Training
        loss_records = np.zeros(num_epochs)
        val_loss_records = np.zeros(num_epochs)

        self.model.train()
        self.likelihood.train()
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.likelihood.parameters()),
            lr=self.initial_lr, amsgrad=True)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.6, patience=10, cooldown=15,
            threshold=1e-3, threshold_mode='rel', min_lr=self.initial_lr*0.6**30
        )

        progress_bar = tqdm(range(num_epochs), desc='Epoch', ascii=True, dynamic_ncols=False, miniters=5)
        for count_epoch in progress_bar:
            train_loss = torch.tensor(0., device=self.device, requires_grad=False)
            for batch, (interaction_context, current_spacing) in enumerate(self.train_dataloader, start=1):
                '''
                interaction_context: [batch_size, 12]
                current_spacing: [batch_size]
                '''
                self.optimizer.zero_grad()
                output = self.model(interaction_context.to(self.device))
                loss = -self.loss_func(output, current_spacing.to(self.device))

                loss.backward()
                self.optimizer.step()
                train_loss += loss
            loss_records[count_epoch] = train_loss.item()/batch

            val_loss = self.val_loop()
            if count_epoch > 20:
                self.scheduler.step(val_loss)
            val_loss_records[count_epoch] = val_loss

            progress_bar.set_postfix({'lr=': self.optimizer.param_groups[0]['lr'], 
                                      'loss=': loss_records[count_epoch], 
                                      'val_loss=': val_loss}, refresh=False)

            if (count_epoch>60) and np.all(abs(np.diff(val_loss_records[count_epoch-3:count_epoch+1])/val_loss_records[count_epoch-3:count_epoch])<5e-4):
                # early stopping if validation loss converges
                print('Validation loss converges and training stops at Epoch '+str(count_epoch))
                break

        progress_bar.close()

        # Save model and likelihood
        self.model.eval()
        self.likelihood.eval()
        torch.save(self.model.state_dict(), self.path_output+f'model_{count_epoch+1}epoch.pth')
        torch.save(self.likelihood.state_dict(), self.path_output+f'likelihood_{count_epoch+1}epoch.pth')
        loss_log = pd.DataFrame(index=[f'epoch_{i}' for i in range(1, len(loss_records[:count_epoch+1])+1)],
                                data={'train_loss': loss_records[:count_epoch+1], 'val_loss': val_loss_records[:count_epoch+1]})
        loss_log.to_csv(self.path_output+'loss_log.csv')

        # Print inspection results
        lognorm_nll = LogNormalNLL()
        smooth_lognorm_nll = SmoothLogNormalNLL(beta=5)

        mu_list, sigma_list = [], []
        val_gau = torch.tensor(0., device=self.device, requires_grad=False)
        val_smooth_gau = torch.tensor(0., device=self.device, requires_grad=False)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for count_batch, (interaction_context, current_spacing) in enumerate(self.val_dataloader, start=1):
                f_dist = self.model(interaction_context.to(self.device))
                y_dist = self.likelihood(f_dist)
                mu, var = y_dist.mean, y_dist.variance # mu: [batch_size], var: [batch_size]
                log_var = torch.log(torch.clamp(var, min=small_eps))
                mu_list.append(mu.cpu().numpy())
                sigma_list.append(var.sqrt().cpu().numpy())
                val_gau += lognorm_nll((mu, log_var), torch.exp(current_spacing.to(self.device)))
                inducing_out = self.get_inducing_out(interaction_context.to(self.device))
                val_smooth_gau += smooth_lognorm_nll((mu, log_var), torch.exp(current_spacing.to(self.device)), inducing_out)

        mu_sigma = pd.DataFrame(data={'mu': np.concatenate(mu_list), 'sigma': np.concatenate(sigma_list)})
        mu_sigma['mode'] = np.exp(mu_sigma['mu']-mu_sigma['sigma']**2)
        print(mu_sigma.describe().to_string())
        print(f'LogNormal NLL: {val_gau.item()/count_batch}, Smooth LogNormal NLL: {val_smooth_gau.item()/count_batch}')
        
    def val_loop(self,):
        self.model.eval()
        self.likelihood.eval()
        val_loss = torch.tensor(0., device=self.device, requires_grad=False)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for count_batch, (interaction_context, current_spacing) in enumerate(self.val_dataloader, start=1):
                output = self.model(interaction_context.to(self.device))
                val_loss += -self.loss_func(output, current_spacing.to(self.device))
        self.model.train()
        self.likelihood.train()
        return val_loss.item()/count_batch


def define_model(num_inducing_points, device):
    # Create representative points for the input space
    # This is defined when training. Don't change it when applying the model.
    inducing_points = np.concatenate([np.random.uniform(0.5, 16.2,(num_inducing_points,1)), # length_ego
                                      np.random.uniform(0.5, 16.2,(num_inducing_points,1)), # length_sur
                                      np.random.uniform(0.,2.5,(num_inducing_points,1)), # combined_width
                                      np.random.uniform(0.,45.,(num_inducing_points,1)), # vy_ego
                                      np.random.uniform(-45.,45.,(num_inducing_points,1)), # vx_sur
                                      np.random.uniform(-45.,45.,(num_inducing_points,1)), # vy_sur
                                      np.random.uniform(0.,2000.,(num_inducing_points,1)), # v_ego2
                                      np.random.uniform(0.,2000.,(num_inducing_points,1)), # v_sur2
                                      np.random.uniform(0.,2000.,(num_inducing_points,1)), # delta_v2
                                      np.random.uniform(-45.,45.,(num_inducing_points,1)), # delta_v
                                      np.random.uniform(-np.pi,np.pi,(num_inducing_points,1)), # psi_sur
                                      np.random.uniform(-np.pi,np.pi,(num_inducing_points,1))], # rho
                                      axis=1)
    inducing_points = torch.from_numpy(inducing_points).float()

    # Define the model
    model = SVGP(inducing_points)
    # Define the likelihood of the model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    ## Load trained model
    model_path = 'PreparedData/PosteriorInference/SafeBaseline/ucd/'
    existing_files = os.listdir(model_path)
    model_file = [file for file in existing_files if 'model_' in file]
    likelihood_file = [file for file in existing_files if 'likelihood_' in file]
    model.load_state_dict(torch.load(f'{model_path}/{model_file[0]}', map_location=torch.device(device), weights_only=True))
    likelihood.load_state_dict(torch.load(f'{model_path}/{likelihood_file[0]}', map_location=torch.device(device), weights_only=True))
    model.eval()
    likelihood.eval()
    model = model.to(device)
    likelihood = likelihood.to(device)
    print('Model loaded successfully.')

    return model, likelihood


class custom_dataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self): 
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float()
    

def lognormal_cdf(x, mu, sigma):
    x = np.maximum(small_eps, x)
    return 1/2+1/2*erf((np.log(x)-mu)/sigma/np.sqrt(2))


def extreme_cdf(x, mu, sigma, n=10):
    return (1-lognormal_cdf(x,mu,sigma))**n


def UCD(states, model, likelihood, device):
    interaction_context, spacing_list = states
    data_loader = DataLoader(custom_dataset(interaction_context), batch_size=1024, shuffle=False)

    mu_list, sigma_list = [], []
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for int_ctxt in tqdm(data_loader, desc='Inferring', ascii=True, dynamic_ncols=False, miniters=100):
            f_dist = model(int_ctxt.to(device))
            y_dist = likelihood(f_dist)
            mu, sigma = y_dist.mean.cpu().numpy(), y_dist.variance.sqrt().cpu().numpy()
            mu_list.append(mu)
            sigma_list.append(sigma)
    mu = np.concatenate(mu_list)
    sigma = np.concatenate(sigma_list)

    # 0.5 means that the probability of conflict is larger than the probability of non-conflict
    one_minus_cdf = 1 - lognormal_cdf(spacing_list, mu, sigma)
    one_minus_cdf = np.minimum(1-small_eps, np.maximum(small_eps, one_minus_cdf))
    max_intensity = np.log(0.5)/np.log(one_minus_cdf) # around (0.050171666, 693146.834)

    return mu, sigma, max_intensity

