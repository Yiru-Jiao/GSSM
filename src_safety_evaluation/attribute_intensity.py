'''
This script uses gradient to estimate the influence of features on potential conflict intensity.
'''

import os
import sys
import shap
import torch
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import time as systime
from sklearn.cluster import MiniBatchKMeans
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src_posterior_inference.inference_utils.utils_general import fix_seed, init_dl_program
from src_posterior_inference.inference_utils.utils_train_eval_test import train_val_test
from src_safety_evaluation.validation_utils.utils_attribution import get_sample
manual_seed = 131


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='The gpu number to use for training and inference (defaults to 0 for CPU only, can be 1,2 for multi-gpu)')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--features', type=int, default=0, help='The random seed')
    parser.add_argument('--reproduction', type=int, default=1, help='Whether this run is for reproduction, if set to True, the random seed would be fixed (defaults to True)')
    args = parser.parse_args()
    args.reproduction = bool(args.reproduction)
    return args


def send_x_to_device(batch, device):
    if isinstance(batch, list):
        return tuple([i.to(device) for i in batch])
    else:
        return batch.to(device)


def main(args, manual_seed, path_prepared, path_result):
    initial_time = systime.time()
    print('Available cpus:', torch.get_num_threads(), 'available gpus:', torch.cuda.device_count())
    
    # Set the random seed
    if args.reproduction:
        args.seed = manual_seed # Fix the random seed for reproduction
    if args.seed is None:
        args.seed = random.randint(0, 1000)
    print(f'Random seed is set to {args.seed}')
    fix_seed(args.seed, deterministic=args.reproduction)
    
    # Initialize the deep learning program
    print(f'--- Cuda available: {torch.cuda.is_available()} ---')
    if torch.cuda.is_available():
        print(f'--- Cuda device count: {torch.cuda.device_count()}, Cuda device name: {torch.cuda.get_device_name()}, Cuda version: {torch.version.cuda}, Cudnn version: {torch.backends.cudnn.version()} ---')
    device = init_dl_program(args.gpu)
    print(f'--- Device: {device}, Pytorch version: {torch.__version__} ---')

    path_save = path_result + 'FeatureAttribution/'
    os.makedirs(path_save, exist_ok=True)

    if args.features == 0:
        encoder_selection = ['current', 'environment']
    elif args.features == 1:
        encoder_selection = ['current', 'environment', 'profiles']
    # Define the model to be used for attribution
    dataset = ['SafeBaseline']
    model_name = f"SafeBaseline_{'_'.join(encoder_selection)}"
    if os.path.exists(path_save + f'{model_name}.h5'):
        print(f'{model_name} has been attributed.')
        sys.exit(0)

    pipeline = train_val_test(device, path_prepared, dataset, encoder_selection, single_output='intensity')
    pipeline.load_model()
    encoder = pipeline._model.combi_encoder
    decoder = pipeline._model.AttentionDecoder
    if not pipeline._model.training:
        print('The model is correctly loaded under evaluation mode.')
    else:
        print('The model is not under evaluation mode, please check whether the model is correctly loaded.')
        sys.exit(0)

    # Define sample generator
    sampler = get_sample(encoder_selection, path_result)
    results = sampler.event_id_list.copy()

    # Cluster representative training samples
    pipeline.create_dataloader(batch_size=1024)
    if os.path.exists(path_save + f'{model_name}_ref_samples.npy'):
        print(f'{model_name}_ref_samples.npy already exists, loading ...')
        ref_samples = np.load(path_save + f'{model_name}_ref_samples.npy')
    else:
        print(f'{model_name}_ref_samples.npy has not been created, clustering ...')
        kmeans = MiniBatchKMeans(n_clusters=1024, random_state=manual_seed, batch_size=1024)
        for batch, spacing in tqdm(pipeline.train_dataloader, total=len(pipeline.train_dataloader), ascii=True, desc='Clustering', miniters=50):
            representations = encoder(send_x_to_device(batch, device)).cpu()
            spacing = torch.cat([spacing.unsqueeze(-1).unsqueeze(-1)]*64, dim=-1)
            inputs = torch.cat((representations, spacing), dim=1).detach().numpy()
            if len(inputs)==1024:
                kmeans = kmeans.partial_fit(inputs.reshape(inputs.shape[0], -1))
            else:
                print(f'Number of samples in the last batch: {len(inputs)}')
        ref_samples = kmeans.cluster_centers_
        np.save(path_save + f'{model_name}_ref_samples.npy', ref_samples)
        del kmeans

    # Define explainer, reference samples are the cluster centers
    ref_samples = torch.from_numpy(ref_samples.reshape(ref_samples.shape[0],-1,64)).float().to(device)
    explainer = shap.GradientExplainer(decoder, ref_samples, batch_size=1024)
    ref_intensity = decoder(ref_samples).squeeze().detach().cpu().numpy()
    np.save(path_save + f'{model_name}_ref_intensity.npy', ref_intensity)
    del ref_samples
    del ref_intensity

    # Read event_id and target_id pairs
    voted_targets = pd.read_csv(path_result + 'Conflicts/Voted_conflicting_targets.csv')
    voted_targets = voted_targets[voted_targets['target_id']>=0]

    # Compute and save gradients for each event
    if os.path.exists(path_save + f'{model_name}.h5'):
        print(f'{model_name}.h5 already exists, loading ...')
        saved_results = pd.read_hdf(path_save + f'{model_name}.h5', key='results')
        existing_ids = saved_results.dropna(subset=['intensity']).reset_index()[['event_id', 'target_id']].drop_duplicates()
        print(f'{len(existing_ids)} event_id and target_id pairs already attributed.')
        voted_targets = voted_targets[~((voted_targets['event_id'].isin(existing_ids['event_id']))&
                                        (voted_targets['target_id'].isin(existing_ids['target_id'])))]
        print(f'{len(voted_targets)} pairs left to compute.')
    else:
        existing_ids = []

    eg_columns = [f'eg_{var}' for var in sampler.variables]
    std_columns = [f'std_{var}' for var in sampler.variables]
    event_count = len(existing_ids)
    for event_id, target_id in tqdm(voted_targets[['event_id','target_id']].values, total=len(voted_targets), ascii=True, desc='Attribution', miniters=10):
        samples, proximity, time = sampler.get_item(event_id, target_id)
        proximity = torch.from_numpy(proximity).float()
        proximity = torch.cat([proximity.unsqueeze(-1).unsqueeze(-1)]*64, dim=-1)
        sample_representations = encoder(send_x_to_device(samples, device)).cpu()
        sample_inputs = torch.cat((sample_representations, proximity), dim=1).to(device)
        eg_matrix, var = explainer.shap_values(sample_inputs, nsamples=1024, return_variances=True, rseed=manual_seed)
        intensity = decoder(sample_inputs).squeeze().detach().cpu().numpy()
        eg_values = eg_matrix[:,:-1,:,0].sum(axis=2)
        std = np.sqrt(var[0][:,:-1,:].sum(axis=2))
        assert eg_values.shape == std.shape
        if np.any(np.isnan(eg_values)):
            Warning(f'There are NaN values in event {event_id} with target {target_id}.')
        results.loc[(event_id, target_id, time), eg_columns] = eg_values
        results.loc[(event_id, target_id, time), std_columns] = std
        results.loc[(event_id, target_id, time), 'intensity'] = intensity
        event_count += 1
        if event_count%100 == 99: # Save once per 100 events
            results.to_hdf(path_save + f'{model_name}.h5', key='results', mode='w')
    results.to_hdf(path_save + f'{model_name}.h5', key='results', mode='w')

    print('--- Total time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - initial_time)) + ' ---')
    sys.exit(0)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    path_prepared = 'PreparedData/'
    path_result = 'ResultData/'
    main(args, manual_seed, path_prepared, path_result)
