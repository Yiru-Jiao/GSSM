'''
'''

import sys
import time as systime
import numpy as np
import pandas as pd
import multiprocessing
from itertools import product
from sklearn.preprocessing import OneHotEncoder


def main(path_prepared):
    initial_time = systime.time()
    print(f'Available cores for parallel processing: {multiprocessing.cpu_count()}')

    # Generate "complete" environment features
    print('Generating downsampled environment features...')
    events = pd.read_csv('./RawData/HondaDataSupport/InsightTables_csv/Event_table.csv').set_index('eventID')
    environment_feature_names = ['lighting','weather','surfaceCondition','trafficDensity']
    categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    events.loc[events['surfaceCondition']=='Other','surfaceCondition'] = 'Unknown'
    data2fit = events[environment_feature_names].fillna('Unknown')
    data2fit = data2fit.loc[(data2fit!='Unknown').all(axis=1)]
    categorical_encoder.fit(data2fit.values)

    categories_list = categorical_encoder.categories_
    categories_list = [list(categories)+['Unknown'] for categories in categories_list]
    all_combinations = np.array(list(product(*categories_list)))
    all_combinations = categorical_encoder.transform(all_combinations)
    all_combinations = pd.DataFrame(all_combinations, columns=categorical_encoder.get_feature_names_out(environment_feature_names))
    all_combinations = all_combinations.astype(np.int32)

    train_data = all_combinations.sample(frac=0.8, random_state=manual_seed)
    val_data = all_combinations.drop(train_data.index)
    train_data.to_hdf(path_prepared + 'Segments/environment_features_train_AE.h5', key='features')
    val_data.to_hdf(path_prepared + 'Segments/environment_features_val_AE.h5', key='features')

    print('--- Total time elapsed: ' + systime.strftime('%H:%M:%S', systime.gmtime(systime.time() - initial_time)) + ' ---')
    sys.exit(0)


if __name__ == '__main__':
    sys.stdout.reconfigure(line_buffering=True)
    manual_seed = 131
    np.random.seed(manual_seed)

    main(path_prepared='./PreparedData/')
