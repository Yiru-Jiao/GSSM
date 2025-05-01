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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='The gpu number to use for training and inference (defaults to 0 for CPU only, can be "1,2" for multi-gpu)')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--stage', type=int, default=None, help='The random seed')
    parser.add_argument('--reproduction', type=int, default=1, help='Whether this run is for reproduction, if set to True, the random seed would be fixed (defaults to True)')
    args = parser.parse_args()
    args.reproduction = bool(args.reproduction)
    return args


def main(args, manual_seed, path_prepared, path_result):
    initial_time = systime.time()
    print('Available cpus:', torch.get_num_threads(), 'available gpus:', torch.cuda.device_count())
    
    # Set the random seed
    if args.reproduction:
        args.seed = manual_seed # Fix the random seed for reproduction
    if args.seed is None:
        args.seed = random.randint(0, 1000)
    print(f"Random seed is set to {args.seed}")
    fix_seed(args.seed, deterministic=args.reproduction)
    
    # Initialize the deep learning program
    print(f'--- Cuda available: {torch.cuda.is_available()} ---')
    if torch.cuda.is_available():
        print(f'--- Cuda device count: {torch.cuda.device_count()}, Cuda device name: {torch.cuda.get_device_name()}, Cuda version: {torch.version.cuda}, Cudnn version: {torch.backends.cudnn.version()} ---')
    device = init_dl_program(args.gpu)
    print(f'--- Device: {device}, Pytorch version: {torch.__version__} ---')

    path_save = path_result + 'FeatureAttribution/'
    os.makedirs(path_save, exist_ok=True)

    encoder_selection = ['current', 'environment']
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
    event_id_list = sampler.event_id_list.reset_index()

    # Cluster representative training samples
    pipeline.create_dataloader(batch_size=1024)
    if os.path.exists(path_save + f'{model_name}_ref_samples.npy'):
        print(f'{model_name}_ref_samples.npy already exists, loading ...')
        ref_samples = np.load(path_save + f'{model_name}_ref_samples.npy')
    else:
        print(f'{model_name}_ref_samples.npy has not been created, clustering ...')
        kmeans = MiniBatchKMeans(n_clusters=1024, random_state=manual_seed, batch_size=1024)
        for batch, spacing in tqdm(pipeline.train_dataloader, total=len(pipeline.train_dataloader), ascii=True, desc='Clustering', miniters=50):
            representations = encoder(batch)
            spacing = torch.cat([spacing.unsqueeze(-1).unsqueeze(-1)]*64, dim=-1)
            inputs = torch.cat((representations, spacing), dim=1).detach().numpy()
            if len(inputs)==1024:
                kmeans = kmeans.partial_fit(inputs.reshape(inputs.shape[0], -1))
            else:
                print(f'Number of samples in the last batch: {len(inputs)}')
        ref_samples = kmeans.cluster_centers_
        np.save(path_save + f'{model_name}_ref_samples.npy', ref_samples)

    # Define explainer, reference samples are the cluster centers
    ref_samples = torch.from_numpy(ref_samples.reshape(ref_samples.shape[0],-1,64)).float()
    explainer = shap.GradientExplainer(decoder, ref_samples, batch_size=1024)
    ref_intensity = decoder(ref_samples).squeeze().detach().numpy()
    np.save(path_save + f'{model_name}_ref_intensity.npy', ref_intensity)

    # Read event_id and target_id pairs
    voted_targets = pd.read_csv(path_result + 'Conflicts/Voted_conflicting_targets.csv')
    voted_targets = voted_targets[voted_targets['target_id']>=0]

    # Compute and save gradients for each event
    existing_ids = []
    if os.path.exists(path_save + f'{model_name}_stage0.h5'):
        print(f'{model_name}_stage0.h5 already exists, loading ...')
        result = pd.read_hdf(path_save + f'{model_name}_stage0.h5', key='results')
        existing_ids.append(result[['event_id','target_id']].drop_duplicates())
        results = [result]
        print(f'{existing_ids[0].shape[0]} event_id and target_id pairs in stage0.')
    else:
        results = []
    if os.path.exists(path_save + f'{model_name}_stage1.h5'):
        print(f'{model_name}_stage1.h5 already exists, loading ...')
        result = pd.read_hdf(path_save + f'{model_name}_stage1.h5', key='results')
        existing_ids.append(result[['event_id','target_id']].drop_duplicates())
        results.append(result)
        print(f'{existing_ids[-1].shape[0]} event_id and target_id pairs in stage1.')
    if os.path.exists(path_save + f'{model_name}_stage2.h5'):
        print(f'{model_name}_stage2.h5 already exists, loading ...')
        result = pd.read_hdf(path_save + f'{model_name}_stage2.h5', key='results')
        existing_ids.append(result[['event_id','target_id']].drop_duplicates())
        results.append(result)
        print(f'{existing_ids[-1].shape[0]} event_id and target_id pairs in stage2.')
    if len(existing_ids)>0:
        existing_ids = pd.concat(existing_ids, axis=0).drop_duplicates()
        voted_targets = voted_targets[~((voted_targets['event_id'].isin(existing_ids['event_id']))&
                                        (voted_targets['target_id'].isin(existing_ids['target_id'])))]
        print(f'{len(existing_ids)} event_id and target_id pairs already exist, {len(voted_targets)} pairs left to compute.')

    feature_list = sampler.variables + ['Spacing']
    eg_columns = [f'eg_{var}' for var in feature_list]
    std_columns = [f'std_{var}' for var in feature_list]
    event_count = 0
    for event_id, target_id in tqdm(voted_targets[['event_id','target_id']].values, total=len(voted_targets), ascii=True, desc='Attribution', miniters=10):
        samples, proximity = sampler.get_item(event_id, target_id)
        proximity = torch.from_numpy(proximity).float()
        proximity = torch.cat([proximity.unsqueeze(-1).unsqueeze(-1)]*64, dim=-1)
        sample_representations = encoder(samples)
        sample_inputs = torch.cat((sample_representations, proximity), dim=1)
        eg_matrix, var = explainer.shap_values(sample_inputs, nsamples=1024, return_variances=True, rseed=manual_seed)
        intensity = decoder(sample_inputs).squeeze().detach().numpy()
        eg_values = eg_matrix[:,:,:,0].sum(axis=2)
        std = np.sqrt(var[0].sum(axis=2))
        assert eg_values.shape == std.shape
        if np.any(np.isnan(eg_values)):
            Warning(f'There are NaN values in event {event_id} with target {target_id}.')
        result = event_id_list[(event_id_list['event_id']==event_id)&(event_id_list['target_id']==target_id)]
        result = result.sort_values(by='idx', ascending=False)[['event_id','target_id','time']]
        result[eg_columns] = eg_values
        result[std_columns] = std
        result['intensity'] = intensity
        results.append(result)
        event_count += 1
        if event_count%100 == 99:
            results = pd.concat(results, axis=0)
            results = [results]
        if event_count%900 == 899:
            pd.concat(results, axis=0).to_hdf(path_save + f'{model_name}_stage{event_count//900}.h5', key='results', mode='w')

    results = pd.concat(results, axis=0)
    results.to_hdf(path_save + f'{model_name}.h5', key='results', mode='w')

    print('--- Total time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - initial_time)) + ' ---')
    sys.exit(0)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()

    manual_seed = 131
    path_prepared = 'PreparedData/'
    path_result = 'ResultData/'
    
    main(args, manual_seed, path_prepared, path_result)
