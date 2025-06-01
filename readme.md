# Codespace for "Learning Collision Risk from Naturalistic Driving with Generalised Surrogate Safety Measures"
This study is being submitted and under review. A preprint is provided at [arXiv](https://arxiv.org/abs/2505.13556). Questions, suggestions, comments, and collaborations are welcome. Please feel free to reach out via email or [GitHub Discussions](https://github.com/Yiru-Jiao/GSSM/discussions).

<!-- ## Directory of dynamic figures
Dynamic visualisations in this paper are saved in the folder [`./ResultData/DynamicFigures/`](ResultData/DynamicFigures/). Below we present the example in Figure 7 of a conflict 

<p align="center">
  <img src="ResultData/DynamicFigures/Figure7/Figure7.gif" alt="animated" width="75%" height="75%"/>
</p>

## Open access to SHRP2 Safety-Critical Trajectory Data
Collaborating with VITTI, we have made the processed anoynomous trajectory data publicly available. The data include 1,836 crashes, 6,881 near-crashes, and 32,581 baselines. The data are available at -->

## TL;DR for Abstract
- Introduces GSSM (Generalised Surrogate Safety Measure) – a neural-network approach that learns potential collisions from naturalistic data without any crash or near-crash labels.
- For any traffic context (motion, weather, lighting, etc.), GSSM flags interactions whose multi-directional spacing deviates toward unsafe extremes, assigning data-driven risk scores and according probability of potential collisions.
- Trained on several public datasets and tested on thousands of real crash/near-crash events, a basic GSSM (using only instantaneous kinematics) achieves AUPRC ≈ 0.90 and warns ≈ 2.6 s before impact; adding richer context boosts performance further.
- GSSM outperforms existing baselines across rear-end, merge, and crossing scenarios, with feature analysis highlighting spacing direction, road surface, and the past second of motion as top risk factors—offering a context-aware and scalable tool for ADAS, safety monitoring, and emergency response.

## In order to repeat the experiments
This offers a workflow to repeat the experiments in the paper. More detailed instructions can be found at the beginning of each script.

### Dependencies
`pandas`, `pytables`, `tqdm`, `numpy`, `matplotlib`, `torch`, `torchvision`, `scikit-learn`, `scipy`, see more detailed dependencies in [`requirementx.txt`](requirements.txt).

### Data
- **SHRP2 NDS:** https://github.com/Yiru-Jiao/BirdsEyeTrajectoryReconstructionSHRP2NDS
- **highD:** https://www.highd-dataset.com/
- **ArgoverseHV:** https://github.com/RomainLITUD/conflict_resolution_dataset

### Bird's eye trajectory reconstruction
- **Transform .xlsx to .csv:** `/src_trajectory_reconstruction/transform_files.py`
- **Summarise meta data:** `/src_trajectory_reconstruction/organise_metadata.py`
- **Search for EKF parameters:** `/src_trajectory_reconstruction/search_ekf_parameter.py`
- **Reconstruct trajectory:** `/src_trajectory_reconstruction/reconstruct_birdseye.py`

### Training data preparation
- **Lane-changes in highD:** `src_data_preparation/prepare_highD.py`
- **Crossing and turning in ArgoverseHV:** `src_data_preparation/prepare_argoverse.py`
- **Segment samples:** `src_data_preparation/segment_datasets.py`

### Posterior inference
- **GSSM training:** `src_posterior_inference/pi_train_eval.py`

### Test data preparation and first-stage safety evaluation
- **Prepare test data:** `src_safety_evaluation/organise_events.py`
- **Apply GSSMs:** `src_safety_evaluation/evaluate_safety.py`
- **Initially evaluate warning at different thresholds:** `src_safety_evaluation/analyse_events.py`

### Second-stage safety evaluation and result analysis
- **Vote for conflicting objects:** `src_safety_evaluation/vote_conflicting_target.py`
- **Re-evaluate warning at different thresholds:** `src_safety_evaluation/risk_evaluation.py`
- **Attribute risk to contextual representations:** `src_safety_evaluation/attribute_intensity.py`

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
- Two-dimensional traffic safety evaluation
  - SSMsOnPlane https://github.com/Yiru-Jiao/SSMsOnPlane
  - EmergencyIndex https://github.com/AutoChengh/EmergencyIndex

We are grateful for the authors' contributions to open science and reproducible research.
