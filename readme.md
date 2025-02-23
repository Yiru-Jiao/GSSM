# Proposal and validation of traffic interaction safety evaluation without manual labelling
## Trajectory reconstruction and conflicting target identification of (near-)crashes in SHRP2 NDS

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
`pandas`, `pytables`, `tqdm`, `numpy`, `matplotlib` ...

### Bird's eye trajectory reconstruction
`/src_trajectory_reconstruction/transform_files.py`, `/src_trajectory_reconstruction/organise_metadata.py`, `/src_trajectory_reconstruction/search_ekf_parameter.py`, `/src_trajectory_reconstruction/reconstruct_birdseye.py`

### Training data preparation
`src_data_preparation/prepare_highD.py`, `src_data_preparation/prepare_INT.ipynb`, `src_data_preparation/prepare_argoverse.py`
`src_data_preparation/segment_datasets.py`, `src_data_preparation/complete_environment_samples.py`

### Encoder pretraining
`ae_train_eval.py`, `src_encoder_pretraining/clt_search_hyperparameter.py`, `src_encoder_pretraining/clt_train_eval.py`,

### Posterior inference
`src_posterior_inference/pi_search_bslr.py`, `src_posterior_inference/pi_train_eval.py`

### Safety evaluation
`src_safety_evaluation/organise_events.py`, `src_safety_evaluation/evaluate_safety.py`, `src_safety_evaluation/analyse_events.py`, `src_safety_evaluation/reuse_ucd/reuse_gaussian.py`, 

## Copyright
### Citation
```latex
@article{
}
```

### Repo references
Thanks to GitHub for offering the open environment, from which this work reuses/learns/adapts the following repositories to different extents:

- Multidimensional Kalman-Filter
  - https://github.com/balzer82/Kalman/blob/master/Extended-Kalman-Filter-CTRA.ipynb
  - https://github.com/balzer82/Kalman/blob/master/Extended-Kalman-Filter-CHCV.ipynb
- Time series contrastive learning and structure preservation
  - SPCLT https://github.com/yiru-jiao/spclt
    - TS2Vec https://github.com/zhihanyue/ts2vec
    - SoftCLT https://github.com/seunghan96/softclt
    - TopoAE https://github.com/BorgwardtLab/topological-autoencoders
    - GGAE https://github.com/JungbinLim/GGAE-public
- Two-dimensional traffic safety evaluation
  - EmergencyIndex https://github.com/AutoChengh/EmergencyIndex
  - UnifiedConflictDetection https://github.com/Yiru-Jiao/UnifiedConflictDetection

We are grateful for the authors' contributions to open science and reproducible research.
