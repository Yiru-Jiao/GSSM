'''
This script segments data into scenes and organises the features for model training.
'''

import os
import sys
import multiprocessing
import time as systime
import numpy as np
from represent_utils.utils_data_segmentation import ContextSegmenter, read_dataset


def main(path_prepared, path_processed, manual_seed):
    initial_time = systime.time()
    print(f'Available cores for parallel processing: {multiprocessing.cpu_count()}')

    for dataset in ['highD', 'SafeBaseline', 'ArgoverseHV']:
        path_save = f'{path_prepared}{dataset}/'
        os.makedirs(path_save, exist_ok=True)
        if os.path.exists(path_save + f'profiles_{dataset}_val.h5'):
            print(f'{dataset} set already segmented.')
            continue

        print('Loading data...')
        data_both = read_dataset(dataset, path_processed, manual_seed)
        # Separate all the events into train (80%) and val (20%) sets, the test set will be (near-)crashes in SHRP2
        '''
        highD: 420,077 train scenes (min. dist. 1.14 m) + 102,853 val scenes (min. dist. 2.84 m)
        SafeBaseline: 427,674 train scenes (min. dist. 1.78 m) + 105,671 val scenes (min. dist. 2.59 m)
        ArgoverseHV: 439,175 train scenes (min. dist. 0.04 m) + 109,462 val scenes (min. dist. 0.20 m)
        '''
        event_ids = data_both['event_id'].unique()
        len_event_ids = len(event_ids)
        train_event_ids = np.random.RandomState(manual_seed).choice(event_ids, int(0.8*len_event_ids), replace=False)
        val_event_ids = np.setdiff1d(event_ids, train_event_ids)
        data_train = data_both[data_both['event_id'].isin(train_event_ids)]
        data_val = data_both[data_both['event_id'].isin(val_event_ids)]

        # Segment and save scenes
        initial_scene_id = 0
        for data, suffix in zip([data_train, data_val], ['train', 'val']):
            print('Segmenting ' + dataset + ' ' + suffix + ' set...')
            segmenter = ContextSegmenter(data, initial_scene_id, dataset, manual_seed)
            print(f'{suffix} current examples:', segmenter.current_features_set.iloc[:5].to_string(), '\n', segmenter.current_features_set.describe().to_string())
            print(f'{suffix} profiles examples:', segmenter.profiles_set.iloc[:5].to_string(), '\n', segmenter.profiles_set.describe().to_string())
            segmenter.profiles_set.to_hdf(path_save + f'profiles_{dataset}_{suffix}.h5', key='profiles')
            segmenter.current_features_set.to_hdf(path_save + f'current_features_{dataset}_{suffix}.h5', key='features')
            if dataset=='SafeBaseline':
                print(f'{suffix} environment features examples:', segmenter.environment_features_set.iloc[:5].to_string(), '\n', segmenter.environment_features_set.describe().to_string())
                segmenter.environment_features_set.to_hdf(path_save + f'environment_features_{dataset}_{suffix}.h5', key='features')
            initial_scene_id = segmenter.current_features_set['scene_id'].max() + 1

            print('--------------------- ' + dataset + ' ' + suffix + ' set segmented ---------------------')
            print(f'Number of scenes: {initial_scene_id - segmenter.initial_scene_id}')
            print(f"Minimum dist.: {segmenter.current_features_set['s'].min():.2f}")
            print(f"Unique scene ids in current features set: {segmenter.current_features_set['scene_id'].nunique()}, should be the same as the profiles set: {segmenter.profiles_set['scene_id'].nunique()}")
            print('--------------------------------------------------------------------')

    print('--- Total time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - initial_time)) + ' ---')


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    manual_seed = 131
    np.random.seed(manual_seed)

    os.makedirs('./PreparedData/Segments/', exist_ok=True)
    path_prepared = './PreparedData/Segments/'
    path_processed = './ProcessedData/'
    main(path_prepared, path_processed, manual_seed)
    sys.exit(0)
