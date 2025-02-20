'''
This script segments data into scenes and organises the features for model training.
'''

import os
import sys
import multiprocessing
import time as systime
import numpy as np
import matplotlib.pyplot as plt
font = {'family' : 'Arial',
        'size'   : 9}
plt.rc('font', **font)
from matplotlib.ticker import ScalarFormatter
from represent_utils.utils_data_segmentation import ContextSegmenter, read_dataset


def main(path_prepared, path_processed):
    initial_time = systime.time()
    print(f'Available cores for parallel processing: {multiprocessing.cpu_count()}')

    for dataset in ['highD', 'INTERACTION', 'Argoverse', 'SafeBaseline']:
        path_save = f'{path_prepared}{dataset}/'
        os.makedirs(path_save, exist_ok=True)
        if os.path.exists(path_save + 'speed_distribution_' + dataset + '.pdf'):
            print(f'{dataset} set already segmented.')
            continue

        print('Loading data...')
        data_both = read_dataset(dataset, path_processed)
        # Separate all the events into train (80%) and val (20%) sets, the test set will be (near-)crashes in SHRP2
        '''
        highD: 355,570 train scenes (min. dist. 0.74 m) + 87,062 val scenes (min. dist. 2.95 m)
        SafeBaseline: 469,087 train scenes (min. dist. 1.64 m) + 115,898 val scenes (min. dist. 1.90 m)
        INTERACTION: train 456,849 scenes (min. dist. 2.54 m) + 115,332 val scenes (min. dist. 2.45 m)
        Argoverse: 1,037,541 train scenes (min. dist. 0.04 m) + 258,493 val scenes (min. dist. 0.22 m)
        '''
        event_ids = data_both['event_id'].unique()
        len_event_ids = len(event_ids)
        train_event_ids = np.random.RandomState(manual_seed).choice(event_ids, int(0.8*len_event_ids), replace=False)
        val_event_ids = np.setdiff1d(event_ids, train_event_ids)
        data_train = data_both[data_both['event_id'].isin(train_event_ids)]
        data_val = data_both[data_both['event_id'].isin(val_event_ids)]

        # Segment and save scenes
        initial_scene_id = 0
        fig, axes = plt.subplots(1, 2, figsize=(5., 2.), constrained_layout=True)
        if dataset=='Argoverse' or dataset=='INTERACTION':
            bins = np.linspace(0, 20, 21)
        else:
            bins = np.linspace(0, 40, 31)
        for data, suffix in zip([data_train, data_val], ['train', 'val']):
            print('Segmenting ' + dataset + ' ' + suffix + ' set...')
            segmenter = ContextSegmenter(data, initial_scene_id, dataset)
            segmenter.profiles_set.to_hdf(path_save + f'profiles_{dataset}_{suffix}.h5', key='profiles')
            segmenter.current_features_set.to_hdf(path_save + f'current_features_{dataset}_{suffix}.h5', key='features')
            if dataset=='SafeBaseline':
                segmenter.environment_features_set.to_hdf(path_save + f'environment_features_{dataset}_{suffix}.h5', key='features')
            initial_scene_id = segmenter.current_features_set['scene_id'].max() + 1
            print(f'Number of scenes: {initial_scene_id - segmenter.initial_scene_id}')
            print(f"Minimum dist.: {segmenter.current_features_set['s'].min():.2f}")
            print(f'Unique scene ids in current features set: {segmenter.current_features_set['scene_id'].nunique()}, should be the same as the profiles set: {segmenter.profiles_set['scene_id'].nunique()}')
            ## save a plot of speed distribution
            ax = axes[0] if suffix=='train' else axes[1]
            ax.hist(segmenter.profiles_set['v_ego'], bins=bins, alpha=0.5, label='Ego vehicle')
            ax.hist(segmenter.profiles_set['v_sur'], bins=bins, alpha=0.5, label='Surrounding vehicles')
            ax.set_xlabel('Speed (m/s)')
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            ax.set_ylabel('Frequency')
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.1))
        axes[0].set_title('Train set')
        axes[1].set_title('Val set')
        fig.savefig(path_save + f'speed_distribution_{dataset}.pdf', bbox_inches='tight', dpi=600)

    print('--- Total time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - initial_time)) + ' ---')
    sys.exit(0)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    manual_seed = 131
    np.random.seed(manual_seed)

    path_prepared = './PreparedData/Segments/'
    path_processed = './ProcessedData/'
    main(path_prepared, path_processed)
