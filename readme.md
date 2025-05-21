# Codespace for "Learning Collision Risk from Naturalistic Driving with Generalised Surrogate Safety Measures"
This study is being submitted and under review. A preprint is provided at [arXiv](https://arxiv.org/abs/2505.13556). Questions, suggestions, comments, and collaborations are welcome. Please feel free to reach out via email or GitHub Issues.

<!-- ## Directory of dynamic figures
Dynamic visualisations in this paper are saved in the folder [`./ResultData/DynamicFigures/`](ResultData/DynamicFigures/). Below we present the example in Figure 7 of a conflict 

<p align="center">
  <img src="ResultData/DynamicFigures/Figure7/Figure7.gif" alt="animated" width="75%" height="75%"/>
</p>

## Open access to SHRP2 Safety-Critical Trajectory Data
Collaborating with VITTI, we have made the processed anoynomous trajectory data publicly available. The data include 1,836 crashes, 6,881 near-crashes, and 32,581 baselines. The data are available at -->

## Abstract
Accurately and proactively alerting drivers or automated systems to unfolding collisions remains a challenge for road safety, particularly in highly interactive urban environments. Existing approaches either require labour-intensive annotation of sparse risk, struggle to consider varying interaction context, or are useful only in the scenarios they are designed for. To address these limits, this study introduces the generalised surrogate safety measure (GSSM), a new data-driven approach that learns exclusively from naturalistic driving data without crash or risk labels. GSSM captures the patterns of normal driving and estimates the extent to which a traffic interaction deviates from the norm towards unsafe extreme. Utilising neural networks, normal interactions are characterised by context-conditioned distributions of multi-directional spacing between road users. Under the same interaction context, a spacing closer than normal entails higher risk of potential collision. Then a context-adaptive risk score and its associated probability can be calculated according to the theory of extreme values. Any measurable factors, such as motion kinematics, weather, lighting, etc., can serve as part of the context, thus allowing for diverse coverage of safety-critical interactions. Multiple public driving datasets are used to train GSSMs, which are tested with 4,875 real-world crashes and near-crashes reconstructed from the SHRP2 Naturalistic Driving Study. A vanilla GSSM using only instantaneous motion information achieves an area under the precision–recall curve of 0.9 and secures a median time advance of 2.6 seconds to prevent potential collisions. Additional interaction data and contextual factors provide further performance gains. Across various interaction types such as rear-end, merging, and crossing, the accuracy and timeliness of GSSM consistently outperforms existing baselines. Furthermore, feature attribution analyses reveal the dominant impacts on risk increase of spacing direction, road-surface condition, and historical kinematics in the passed second. GSSM therefore establishes a scalable, context-aware, and generalisable foundation for proactively quantifying collision risk in traffic interactions. This can support and facilitate driver-assistance systems, traffic safety assessment, and road emergency management.

## In order to repeat the experiments
This offers a workflow to repeat the experiments in the paper. More detailed instructions can be found at the beginning of each script.

### Dependencies
`pandas`, `pytables`, `tqdm`, `numpy`, `matplotlib`, `torch`, `torchvision`, `scikit-learn`, `scipy`, see more detailed dependencies in [`requirementx.txt`](requirements.txt).

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
