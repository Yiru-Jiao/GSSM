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
from src_encoder_pretraining.ssrl_utils.utils_general import fix_seed, init_dl_program
from src_posterior_inference.inference_utils.utils_train_eval_test import train_val_test
from src_safety_evaluation.validation_utils.utils_evaluation import optimize_threshold
from src_safety_evaluation.validation_utils.utils_attribution import get_sample


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='The gpu number to use for training and inference (defaults to 0 for CPU only, can be "1,2" for multi-gpu)')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
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

    encoder_selections = [['current'], ['current', 'environment'], ['current', 'environment', 'profiles']]
    for encoder_selection in encoder_selections:
        # Define the model to be used for attribution
        dataset = ['SafeBaseline']
        pretrained_encoder = False
        model_name = f"SafeBaseline_{'_'.join(encoder_selection)}_not_pretrained"

        pipeline = train_val_test(device, path_prepared, dataset, encoder_selection, pretrained_encoder, single_output='intensity')
        pipeline.load_model()
        tokenizer = pipeline.model.combi_encoder
        decoder = pipeline.model.AttentionDecoder
        if not pipeline.model.training:
            print('The model is correctly loaded under evaluation mode.')
        else:
            print('The model is not under evaluation mode, please check whether the model is correctly loaded.')
            sys.exit(0)

        # Define sample generator
        sampler = get_sample(encoder_selection, path_result)
        event_id_list = sampler.event_id_list.reset_index()

        # Cluster representative training samples
        pipeline.create_dataloader(batch_size=1024)
        kmeans = MiniBatchKMeans(n_clusters=1024, random_state=manual_seed, batch_size=1024)
        for batch, spacing in tqdm(pipeline.train_dataloader, total=len(pipeline.train_dataloader), ascii=True, desc='Clustering', miniters=50):
            tokens = tokenizer(batch)
            spacing = torch.cat([spacing.unsqueeze(-1).unsqueeze(-1)]*64, dim=-1)
            inputs = torch.cat((tokens, spacing), dim=1).detach().numpy()
            if len(inputs)==1024:
                kmeans = kmeans.partial_fit(inputs.reshape(inputs.shape[0], -1))
            else:
                print(f'Number of samples in the last batch: {len(inputs)}')

        # Define explainer, background samples are the cluster centers
        bkgd_samples = kmeans.cluster_centers_
        bkgd_samples = torch.from_numpy(bkgd_samples.reshape(bkgd_samples.shape[0],-1,64)).float()
        explainer = shap.GradientExplainer(decoder, bkgd_samples, batch_size=1024)

        # Read events
        warning = pd.read_hdf(path_result + f'Analyses/Warning_{model_name}.h5', key='results')
        optimal_threshold = optimize_threshold(warning, 'SSSE', 'ROC')
        warning = warning[(warning['threshold']==optimal_threshold)&warning['danger_recorded']]

        # Compute and save gradients for each event
        results = []
        ig_columns = [f'ig_{var}' for var in sampler.variables]
        std_columns = [f'std_{var}' for var in sampler.variables]
        for event_id, target_id in tqdm(warning[['event_id','target_id']].values, total=len(warning), ascii=True, desc='Attribution', miniters=10):
            samples, proximity = sampler.get_item(event_id, target_id)
            proximity = torch.from_numpy(proximity).float()
            proximity = torch.cat([proximity.unsqueeze(-1).unsqueeze(-1)]*64, dim=-1)
            sample_tokens = tokenizer(samples)
            sample_inputs = torch.cat((sample_tokens, proximity), dim=1)
            ig_matrix, var = explainer.shap_values(sample_inputs, return_variances=True, rseed=manual_seed)
            intensity = decoder(sample_inputs).squeeze().detach().numpy()
            ig_values = ig_matrix[:,:-1,:,0].sum(axis=2)
            std = np.sqrt(var[0][:,:-1,:].sum(axis=2))
            assert ig_values.shape == std.shape
            if np.any(np.isnan(ig_values)):
                Warning(f'There are NaN values in event {event_id} with target {target_id}.')
            result = event_id_list[(event_id_list['event_id']==event_id)&(event_id_list['target_id']==target_id)]
            result = result.sort_values(by='idx', ascending=False)[['event_id','target_id','time']]
            result[ig_columns] = ig_values
            result[std_columns] = std
            result['intensity'] = intensity
            results.append(result)

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
