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
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src_data_preparation.represent_utils.coortrans import coortrans
coortrans = coortrans()


# Create dataloader
class DataOrganiser:
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
        cur_spac = self.current_spacing.loc[self.idx_list[idx]].values
        cur_spac = torch.from_numpy(cur_spac).float()
        return int_ctxt, cur_spac

    def read_data(self,):
        features = pd.read_hdf(self.path_input + 'current_features_highD_' + self.dataset + '.h5', key='features')
        self.idx_list = features['scene_id'].values
        features = features.set_index('scene_id')
        self.interaction_context = features.drop(columns=['s']).copy()
        # log-transform spacing, and the spacing must be larger than 0
        if np.any(features['s']<=1e-6):
            print('There are spacings smaller than or equal to 0.')
            features.loc[features['s']<=1e-6, 's'] = 1e-6
        self.current_spacing = np.log(features[['s']]).copy()
        features = []


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
        mixture_kernel = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=10, ard_num_dims=11)
        rbf_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel(ard_num_dims=11))
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
                                        'w_ego': np.random.uniform(0.5, 2.6,num_inducing_points),
                                        'l_sur': np.random.uniform(0.5, 16.2,num_inducing_points),
                                        'w_sur': np.random.uniform(0.5, 2.6,num_inducing_points),
                                        'delta_v2': np.random.uniform(0.,400.,num_inducing_points),
                                        'delta_v': np.random.uniform(-20.,20.,num_inducing_points),
                                        'psi_sur': np.random.uniform(-np.pi,np.pi,num_inducing_points),
                                        'acc_ego': np.random.uniform(-5.5,5.5,num_inducing_points),
                                        'v_ego2': np.random.uniform(0.,2000.,num_inducing_points),
                                        'v_sur2': np.random.uniform(0.,2000.,num_inducing_points),
                                        'rho': np.random.uniform(-np.pi,np.pi,num_inducing_points)})
        return inducing_points.values


    # Validation loop
    def val_loop(self,):
        self.model.eval()
        self.likelihood.eval()

        val_loss = 0
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for count_batch, (interaction_context, current_spacing) in enumerate(self.val_dataloader):
                with torch.amp.autocast(device_type="cuda"):  # Enables Mixed Precision
                    output = self.model(interaction_context.to(self.device))
                    loss = -self.loss_func(output, current_spacing.squeeze().to(self.device)).item()
                val_loss += loss

        self.model.train()
        self.likelihood.train()

        return val_loss/(count_batch+1)


    def train_model(self, num_epochs=100, initial_lr=0.1):
        self.initial_lr = initial_lr

        # Move model and likelihood to device
        self.model = self.model.to(self.device)
        self.likelihood = self.likelihood.to(self.device)
        self.loss_func = self.loss_func.to(self.device)

        # Training
        num_batches = len(self.train_dataloader)
        loss_records = np.zeros((num_epochs, num_batches))
        val_loss_records = [100., 99., 98., 97., 96.]

        self.model.train()
        self.likelihood.train()
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.likelihood.parameters()),
            lr=self.initial_lr, amsgrad=True)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.6, patience=4, verbose='deprecated',
            threshold=1e-3, threshold_mode='rel', cooldown=4, min_lr=self.initial_lr*0.6**15
        )

        progress_bar = tqdm(range(num_epochs), desc='Epoch', ascii=True, dynamic_ncols=False)
        scaler = torch.amp.GradScaler()  # Initialize gradient scaler for mixed precision training
        for count_epoch in progress_bar:
            for batch, (interaction_context, current_spacing) in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                with torch.amp.autocast(device_type="cuda"):  # Enables Mixed Precision
                    output = self.model(interaction_context.to(self.device))
                    loss = -self.loss_func(output, current_spacing.squeeze().to(self.device))
                loss_records[count_epoch, batch] = loss.item()

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

            val_loss = self.val_loop()
            self.scheduler.step(val_loss)
            val_loss_records.append(val_loss)

            progress_bar.set_postfix({'lr=': self.optimizer.param_groups[0]['lr'], 
                                      'loss=': loss.item(), 
                                      'val_loss=': val_loss}, refresh=False)
            progress_bar.update(1)

            stop_condition = np.all(abs(np.diff(val_loss_records)[-4:]/np.array(val_loss_records)[-4:])<1e-3)
            if stop_condition:
                # early stopping if validation loss converges
                print('Validation loss converges and training stops at Epoch '+str(count_epoch))
                break

        # Save loss records
        loss_records = pd.DataFrame(loss_records, columns=[f'batch_{i}' for i in range(num_batches)])
        loss_records.to_csv(self.path_output+'loss_records.csv', index=False)
        # Save model every epoch
        torch.save(self.model.state_dict(), self.path_output+f'model_{count_epoch+1}epoch.pth')
        torch.save(self.likelihood.state_dict(), self.path_output+f'likelihood_{count_epoch+1}epoch.pth')


def define_model(num_inducing_points, device):
    # Create representative points for the input space
    # This is defined when training. Don't change it when applying the model.
    inducing_points = np.concatenate([np.random.uniform(0.5, 16.2,(num_inducing_points,1)), # length_ego
                                      np.random.uniform(0.5, 2.6,(num_inducing_points,1)), # width_ego
                                      np.random.uniform(0.5, 16.2,(num_inducing_points,1)), # length_sur
                                      np.random.uniform(0.5, 2.6,(num_inducing_points,1)), # width_sur
                                      np.random.uniform(0.,400.,(num_inducing_points,1)), # delta_v2
                                      np.random.uniform(-20.,20.,(num_inducing_points,1)), # delta_v
                                      np.random.uniform(-np.pi,np.pi,(num_inducing_points,1)), # psi_sur
                                      np.random.uniform(-5.5,5.5,(num_inducing_points,1)), # acc_ego
                                      np.random.uniform(0.,2000.,(num_inducing_points,1)), # v_ego2
                                      np.random.uniform(0.,2000.,(num_inducing_points,1)), # v_sur2
                                      np.random.uniform(-np.pi,np.pi,(num_inducing_points,1))], # rho
                                      axis=1)
    inducing_points = torch.from_numpy(inducing_points).float()

    # Define the model
    model = SVGP(inducing_points)
    # Define the likelihood of the model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    ## Load trained model
    model_path = 'src_safety_evaluation/reuse_ucd'
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
    data['delta_v2'] = (data['vx_ego']-data['vx_sur'])**2 + (data['vy_ego']-data['vy_sur'])**2
    data['delta_v'] = np.sqrt(data['delta_v2']) * np.sign(data['v_ego']-data['v_sur'])
    data['v_ego2'] = data['v_ego']**2
    data['v_sur2'] = data['v_sur']**2
    data_view_ego = coortrans.transform_coor(data, view='i')
    heading_sur = data_view_ego[['target_id','time','hx_sur','hy_sur']]
    data_relative = coortrans.transform_coor(data, view='relative')
    rho = coortrans.angle(1, 0, data_relative['x_sur'], data_relative['y_sur']).reset_index().rename(columns={0:'rho'})
    rho[['target_id','time']] = data_relative[['target_id','time']]
    interaction_context = data.drop(columns=['hx_sur','hy_sur']).merge(heading_sur, on=['target_id','time']).merge(rho, on=['target_id','time'])
    features = ['length_ego','width_ego','length_sur','width_sur',
                'delta_v2','delta_v','psi_sur','acc_ego','v_ego2','v_sur2','rho']
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

    max_intensity = np.log(0.5)/np.log(1-lognormal_cdf(proximity, mu_list, sigma_list)+1e-6)
    max_intensity = np.maximum(1., max_intensity)

    results = interaction_context[['event_id','target_id','time']].copy()
    results['proximity'] = proximity
    results['mu'] = mu_list
    results['sigma'] = sigma_list
    results['intensity'] = max_intensity
    return results

