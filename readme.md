# Codespace for "Learning unsafety from safety: scalable, context-aware, and generalisable proactive risk quantification of traffic collisions"
This study is being submitted and under review. A preprint is provided at https://arxiv.org/abs/. Questions, suggestions, comments, and collaborations are welcome. Please feel free to reach out via email or GitHub Issues.

## Directory of dynamic figures
Dynamic visualisations in this paper are saved in the folder [`./ResultData/DynamicFigures/`](ResultData/DynamicFigures/). Below we present the example in Figure 7 of a conflict 

<p align="center">
  <img src="ResultData/DynamicFigures/Figure7/Figure7.gif" alt="animated" width="75%" height="75%"/>
</p>

## Open access to SHRP2 Safety-Critical Trajectory Data
Collaborating with VITTI, we have made the processed anoynomous trajectory data publicly available. The data include 1,836 crashes, 6,881 near-crashes, and 32,581 baselines. The data are available at

## Abstract


## In order to repeat the experiments
This offers a workflow to repeat the experiments in the paper. More detailed instructions can be found at the beginning of each script.

### Dependencies
`pandas`, `pytables`, `tqdm`, `numpy`, `matplotlib` ...

### Bird's eye trajectory reconstruction
- **Transform .xlsx to .csv:** `/src_trajectory_reconstruction/transform_files.py`
- **Meta data:** `/src_trajectory_reconstruction/organise_metadata.py`
- **EKF:** `/src_trajectory_reconstruction/search_ekf_parameter.py`
- **Reconstruct:** `/src_trajectory_reconstruction/reconstruct_birdseye.py`

### Training data preparation
- **:** `src_data_preparation/prepare_highD.py`
- **:** `src_data_preparation/prepare_argoverse.py`
- **:** `src_data_preparation/complete_environment_samples.py`
- **:** `src_data_preparation/segment_datasets.py`

### Posterior inference
- **:** `src_posterior_inference/pi_train_eval.py`

### Test data preparation and first-stage safety evaluation
- **:** `src_safety_evaluation/organise_events.py`
- **:** `src_safety_evaluation/evaluate_safety.py`
- **:** `src_safety_evaluation/analyse_events.py`

### Second-stage safety evaluation and result analysis
- **:** `src_safety_evaluation/vote_conflicting_target.py`
- **:** `src_safety_evaluation/risk_evaluation.py`
- **:** `src_safety_evaluation/attribute_intensity.py`

## Copyright
This work is licensed under
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
- Two-dimensional traffic safety evaluation
  - SSMsOnPlane https://github.com/Yiru-Jiao/SSMsOnPlane
  - EmergencyIndex https://github.com/AutoChengh/EmergencyIndex

We are grateful for the authors' contributions to open science and reproducible research.
