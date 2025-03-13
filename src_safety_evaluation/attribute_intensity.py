'''
This script uses gradient to estimate the influence of features on potential conflict intensity.
'''

import os
import sys
import shap
import random
import time as systime
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.special import erf
import torch
import argparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import OneHotEncoder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src_encoder_pretraining.ssrl_utils.utils_general import fix_seed, init_dl_program
from src_data_preparation.represent_utils.coortrans import coortrans
coortrans = coortrans()
from src_posterior_inference.inference_utils.utils_train_eval_test import train_val_test
from src_safety_evaluation.validation_utils.utils_evaluation import read_events, set_veh_dimensions
from src_safety_evaluation.validation_utils.utils_attribution import get_sample


def main(events, manual_seed, path_prepared, path_result):
    initial_time = systime.time()
    print('Available cpus:', torch.get_num_threads(), 'available gpus:', torch.cuda.device_count())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path_save = path_result + 'EventEvaluation/'
    os.makedirs(path_save, exist_ok=True)

    # Define the model to be used for attribution
    dataset = ['SafeBaseline']
    encoder_selection = ['current','profiles']
    pretrained_encoder = False

    pipeline = train_val_test(device, path_prepared, dataset, encoder_selection, pretrained_encoder, single_output='intensity')
    pipeline.load_model()
    tokenizer = pipeline.model.combi_encoder
    decoder = pipeline.model.AttentionDecoder

    # Define sample generator
    sampler = get_sample(encoder_selection, path_result)

    # Cluster representative training samples
    pipeline.create_dataloader(batch_size=1024)
    kmeans = MiniBatchKMeans(n_clusters=1024, random_state=manual_seed, batch_size=1024)
    for batch, spacing in tqdm(pipeline.train_dataloader, total=len(pipeline.train_dataloader), ascii=True, desc='Clustering', miniters=10):
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
    explainer = shap.GradientExplainer(decoder, bkgd_samples)

    # Compute and save gradients
    test_idx = (2843842, 13479, None)
    samples, proximity = sampler.get_item(*test_idx)
    sample_tokens = tokenizer_mu(samples)
    mu_shap_values = explainer_mu(sample_tokens)
    logvar_shap_values = explainer_logvar(sample_tokens)
    mu, logvar = decoder_mu(sample_tokens).squeeze().detach().numpy(), decoder_logvar(sample_tokens).squeeze().detach().numpy()
    shap_values = get_gradient(proximity[idx], mu_shap_values[idx,:,:,0].values, logvar_shap_values[idx,:,:,0].values, mu[idx], logvar[idx])


    print('--- Total time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - initial_time)) + ' ---')
    sys.exit(0)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()

    manual_seed = 131
    path_prepared = 'PreparedData/'
    path_result = 'ResultData/'

    # Load event information to create one-hot encoder later
    events = pd.read_csv('RawData/SHRP2/HondaDataSupport/InsightTables_csv/Event_Table.csv').set_index('eventID')
    
    main(args, events, manual_seed, path_prepared, path_result)
