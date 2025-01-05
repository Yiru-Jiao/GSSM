# Trajectory reconstruction of crashes, near-crashes, and baselines from SHRP2 NDS data

The data were collected in the US between 2012 and 2013, covering 1,836 crashes, 6,881 manually annotated near-crashes, and 32,581 safe driving baselines. I believe this is valuable for not only my research, but also for you and your PhDs. An overview of SHRP2 is available at https://www.transportationops.org/Bigdata/NDS

SHRP2 data are not publicly accessible and require a use license. I've been contacting VITTI (the data owner) and am now filling in a form to access two specific datasets as detailed in the below links. These two sets include all the event data and are particularly suited for data-driven safety research.

https://dataverse.vtti.vt.edu/dataset.xhtml?persistentId=doi:10.15787/VTT1/DEDACT

https://dataverse.vtti.vt.edu/dataset.xhtml?persistentId=doi:10.15787/VTT1/FQLUWZ

This repository reconstructs bird's eye view trajectories of vehicles involved in crashes and near-crashes from 100-Car Naturalistic Driving Study (NDS) radar data.


## To repeat/adjust the processing
### Python libarary requirements
`pandas`, `pytables`, `tqdm`, `numpy`, `matplotlib`

### Wrokflow
**Step 1.** Download the raw data from [^3] in the folder `RawData`. This include: `100CarVehicleInformation_v1_0.txt`, `100CarEventVideoReducedData_v1_5.txt`, `HundredCar_Crash_Public_Compiled.txt`, `HundredCar_NearCrash_Public_Compiled.txt`, `Researcher Dictionary for Vehicle Data v1_0.pdf`, `Researcher Dictionary for Video Reduction Data v1.3.pdf`, and `DataDictionary_TimeSeries_v1_2.pdf`. (Given that the license of raw data is now CC0 1.0, which means no limits, this repo has included needed data for your convenience.)

**Step 2.** Convert `100CarVehicleInformation_v1_0.txt` into `100CarVehicleInformation.csv` using microsoft excel or other data sheet tools, and rename the column names based on corresponding data dictionary; similarly, convert the `100CarVehicleInformation_v1_0.txt` into `100CarEventVideoReducedData.csv`, rename and remain the columns of `webfileid`, `vehicle webid`, `event start`, `event end`, `event severity`, `target type`, `event nature`, then remove "Conflict with " in the descriptions and rename the column name `event nature` by `target`. (This has also been done in this repo.)

**Step 3.** Run `preprocessing_100Car.py`

**Step 4.** Run `processing_100Car.py`

**Step 5.** Run `event_matching.py`, which can be adjusted for your own matching

**Step 6.** Use `visualiser.ipynb` to observe the reconstructed events

## To repeat the experiments

**Step 1.** `/src_trajectory_reconstruction/transform_files.py`, `/src_trajectory_reconstruction/organise_metadata.py`, `/src_trajectory_reconstruction/search_ekf_parameter.py`, `/src_trajectory_reconstruction/reconstruct_birdseye.py`

**Step 2.** `src_data_preparation/segment_datasets.py`, `src_data_preparation/downsample_profiles.py`,

**Step 3.** `src_encoder_pretraining/clt_search_hyperparameter.py`, `src_encoder_pretraining/clt_train_eval.py`,

**Step 4.** `src_gaussian_regression/train_gpr.py`

**Step 5.** `src_conflict_Detection/conflict_evaluation.py`


## Copyright
