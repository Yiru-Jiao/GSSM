# Learning Collision Risk from Naturalistic Driving with Generalised Surrogate Safety Measures
This study is being submitted and under review. A preprint is provided at [arXiv](https://arxiv.org/abs/2505.13556). Questions, suggestions, comments, and collaborations are welcome. Please feel free to reach out.

<!-- ## Directory of dynamic figures
Dynamic visualisations in this paper are saved in the folder [`./ResultData/DynamicFigures/`](ResultData/DynamicFigures/). Below we present the example in Figure 7 of a conflict 

<p align="center">
  <img src="ResultData/DynamicFigures/Figure7/Figure7.gif" alt="animated" width="75%" height="75%"/>
</p>
 -->

## Discussions open to everyone
We enable [GitHub Discussions](https://github.com/Yiru-Jiao/GSSM/discussions) for this repository, where you are welcome to ask questions, share insights, and discuss the content of the paper or future research. We encourage everyone to participate in the discussions, as it helps foster a collaborative environment for learning and improvement.

## Access to trajectory data of crashes and near-crashes
Collaborated with Virginia Tech Transportation Institute (VITTI), we have made the trajectory reconstruction dataset of naturalistic crashes and near-crashes in SHRP2 NDS accessible. You are welcome to refer to [BirdsEyeTrajectoryReconstructionSHRP2NDS](https://github.com/Yiru-Jiao/BirdsEyeTrajectoryReconstructionSHRP2NDS) for more information and guidelines to use.

## TL;DR for Abstract
- Introduces GSSM (Generalised Surrogate Safety Measure) – a neural network approach that learns potential collisions from naturalistic interactions without crash or near-crash annotations.
- For all traffic interactions characterised by motion, weather, lighting, etc., GSSM can flag interactions whose multi-directional spacing deviates toward unsafe extremes, assigning data-driven risk scores and according probability of potential collisions.
- Trained on various public datasets and tested on 2,591 real crash/near-crash events, a basic GSSM (using only instantaneous kinematics) achieves AUPRC ≈ 0.90 and warns ≈ 2.6 s before impact; adding richer context boosts performance further.
- GSSM outperforms existing baselines across rear-end, merging, and crossing scenarios, with feature attributions highlighting spacing direction, road surface, and the past second of motion as top risk factors

This work enables **context-aware**, **scalable**, and **generalisable** learning of collision risk from everyday interactions. I believe it is opening a window to foundational models for proactive risk quantification of potential collisions. This hopefully will facilitate research in safe autonomous driving and traffic safety analytics.

## In order to repeat the experiments
Below we offer a step-by-step workflow to repeat the experiments in the paper. On the top of each script, we provide a brief description of its purpose. These scripts are designed to be run in sequence, and each script outputs necessary files for the next one.

**Quick Navigation:**
- [1 Dependencies](#1-dependencies)
- [2 Data](#2-data)
- [3 Bird's eye trajectory reconstruction](#3-birds-eye-trajectory-reconstruction)
- [4 Training data preparation](#4-training-data-preparation)
- [5 Posterior inference](#5-posterior-inference)
- [6 Test data preparation and first-stage evaluation](#6-test-data-preparation-and-first-stage-evaluation)
- [7 Second-stage evaluation and result analysis](#7-second-stage-evaluation-and-result-analysis)

### 1 Dependencies
`pandas`, `pytables`, `tqdm`, `numpy`, `matplotlib`, `torch`, `torchvision`, `scikit-learn`, `scipy`, see more detailed dependencies in [`requirements.txt`](requirements.txt).

### 2 Data
Result data such as trained models, loss logs, evaluation results, will be shared later (as I'm still finalising my thesis and uploading research data as required by TU Delft). A link will be provided soon.

Three datasets are used in this study.
- **SHRP2 NDS:**  
  Two options are available depending on whether you are interested in trajectory reconstruction from the original event data.
  - If you would like to skip trajectory reconstruction:  
    Download the dataset following the guidlines at https://github.com/Yiru-Jiao/BirdsEyeTrajectoryReconstructionSHRP2NDS. Put the files in `ReconstructedTrajectories.zip` under the directory `./ProcessedData/SHRP2/`. Put the files in `SafetyCriticalTestSet.zip` under the directory `./ResultData/EventData/`.
  - If you would like to reconstruct the trajectories:  
    Apply for the original datasets 
    - E. Sears, M. A. Perez, K. Dan, T. Shimamiya, T. Hashimoto, M. Kimura, S. Yamada, and T. Seo. A Study on the Factors That Affect the Occurrence of Crashes and Near-Crashes. Version V2. 2019. URL: https://doi.org/10.15787/VTT1/FQLUWZ
    - C. K. Layman, M. A. Perez, T. Sugino, and J. Eggert. Research of Driver Assistant System. Version V3. 2019. URL: https://doi.org/10.15787/VTT1/DEDACT  
    and put them under the directory `./RawData/SHRP2/`.  

- **highD:**  
  Download the dataset from https://www.highd-dataset.com/ to the directory `./RawData/highD/`.
- **ArgoverseHV:**  
  Download the dataset following the guidlines at https://github.com/RomainLITUD/conflict_resolution_dataset. Put the files under the directory `./RawData/Argoverse2/`.

### 3 Bird's eye trajectory reconstruction
Run the following scripts in order to reconstruct the trajectories of the events in SHRP2 NDS. 
- **Transform .xlsx to .csv:** `./src_trajectory_reconstruction/transform_files.py`
- **Summarise meta data:** `./src_trajectory_reconstruction/organise_metadata.py`
- **Search for EKF parameters:** `./src_trajectory_reconstruction/search_ekf_parameter.py`
- **Reconstruct trajectories:** `./src_trajectory_reconstruction/reconstruct_birdseye.py`
- **Visualise reconstructed trajectories:** `./src_trajectory_reconstruction/event_visualiser.ipynb` (This is optional in case you would like to check the reconstructed events.)

### 4 Training data preparation
Run the following scripts in order to prepare the training data. The SHRP2 SafeBaseline data are processed in the previous step, here we prepare the highD and ArgoverseHV data. Then we segment the data into samples, and split each dataset into a train set (80%) and a val set (20%) for GSSM training.
- **Extract lane-changes in highD:** `./src_data_preparation/prepare_highD.py`
- **Filter HV-HV crossings and turnings:** `./src_data_preparation/prepare_argoverse.py`
- **Segment samples:** `./src_data_preparation/segment_datasets.py`

### 5 Posterior inference
Run the following script in order to train the GSSMs. The trained models and loss logs will be saved for later use. You can run this script for multiple times. It will skip trained models and continue training until all models are trained.
- **GSSM training:** `./src_posterior_inference/pi_train_eval.py`

### 6 Test data preparation and first-stage evaluation
Run the following scripts in order to prepare the test set and implement the first-stage evaluation.
- **Prepare test set:** `./src_safety_evaluation/organise_events.py` (This can be skipped if you have downloaded the reconstructed trajectories and put `SafetyCriticalTestSet.zip` under the directory `./ResultData/EventData/`.)
- **Apply GSSMs to 4,875 events in the test set:** `./src_safety_evaluation/evaluate_safety.py`
- **Initially evaluate warning at different thresholds:** `./src_safety_evaluation/analyse_events.py`

### 7 Second-stage evaluation and result analysis
Run the following scripts in order to implement the second-stage evaluation and analyse the results.
- **Vote for conflicting objects:** `./src_safety_evaluation/vote_conflicting_target.py`
- **Re-evaluate warnings for 2,591 events that have conflicting objects determined:** `./src_safety_evaluation/risk_evaluation.py`
- **Attribute risk to contextual representations:** `./src_safety_evaluation/attribute_intensity.py`

### 8 Visualise results
Use the following notebook to reproduce the figures and tables in the paper.
- **Visualise results:** `./src_visualisation/figure_table_to_reproduce.ipynb`


## Copyright

### Citation
```latex
@article{jiao2025gssm,
    title = {Learning Collision Risk from Naturalistic Driving with Generalised Surrogate Safety Measures},
    author = {Yiru Jiao and Simeon C. Calvert and Sander {van Cranenburgh} and Hans {van Lint}},
    year = {2025},
    journal = {arXiv preprint},
    pages = {arXiv:2505.13556}
}
```

### License
This repository is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute the code, but please retain the original copyright notice and license in any copies or substantial portions of the software.

### Repo references
Thanks to GitHub for offering the open environment, from which this work reuses/learns/adapts the following repositories to different extents:

- Multidimensional Kalman-Filter
  - https://github.com/balzer82/Kalman/blob/master/Extended-Kalman-Filter-CTRA.ipynb
  - https://github.com/balzer82/Kalman/blob/master/Extended-Kalman-Filter-CHCV.ipynb
- Two-dimensional traffic safety evaluation
  - SSMsOnPlane https://github.com/Yiru-Jiao/SSMsOnPlane
  - EmergencyIndex https://github.com/AutoChengh/EmergencyIndex

We are grateful for the authors' contributions to open science and reproducible research.
