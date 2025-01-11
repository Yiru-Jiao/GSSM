# Trajectory reconstruction of crashes, near-crashes, and baselines from SHRP2 NDS data

The data were collected in the US between 2012 and 2013, covering 1,836 crashes, 6,881 manually annotated near-crashes, and 32,581 safe driving baselines. I believe this is valuable for not only my research, but also for you and your PhDs. An overview of SHRP2 is available at https://www.transportationops.org/Bigdata/NDS

SHRP2 data are not publicly accessible and require a use license. I've been contacting VITTI (the data owner) and am now filling in a form to access two specific datasets as detailed in the below links. These two sets include all the event data and are particularly suited for data-driven safety research.

https://dataverse.vtti.vt.edu/dataset.xhtml?persistentId=doi:10.15787/VTT1/DEDACT

https://dataverse.vtti.vt.edu/dataset.xhtml?persistentId=doi:10.15787/VTT1/FQLUWZ

This repository reconstructs bird's eye view trajectories of vehicles involved in crashes and near-crashes from 100-Car Naturalistic Driving Study (NDS) radar data.

## Open access to SHRP2 Safety-Critical Trajectory Data
Collaborating with VITTI, we have made the processed anoynomous trajectory data publicly available. The data include 1,836 crashes, 6,881 near-crashes, and 32,581 baselines. The data are available at

## Directory of dynamic figures


## To repeat the experiments
This offers a workflow to repeat the experiments in the paper. More detailed instructions can be found at the beginning of each script.

### Dependencies
`pandas`, `pytables`, `tqdm`, `numpy`, `matplotlib`

### Bird's eye trajectory reconstruction
`/src_trajectory_reconstruction/transform_files.py`, `/src_trajectory_reconstruction/organise_metadata.py`, `/src_trajectory_reconstruction/search_ekf_parameter.py`, `/src_trajectory_reconstruction/reconstruct_birdseye.py`

### Training data preparation
`src_data_preparation/segment_datasets.py`, `src_data_preparation/complete_environment_samples.py`

### Encoder pretraining
`ae_train_eval.py`, `src_encoder_pretraining/clt_search_hyperparameter.py`, `src_encoder_pretraining/clt_train_eval.py`,

### Posterior inference
`src_posterior_inference/pi_search_bslr.py`, `src_posterior_inference/pi_train_eval.py`

### Conflict analysis
`src_conflict_Detection/conflict_evaluation.py` ...

## Copyright
Free of use

Citation
