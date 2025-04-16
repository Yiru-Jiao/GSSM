# Data dictionary
## Data structure
The structure of data in this folder is as follows:
```
----- Parent folder
  |-- readme_data_structure.md
  |-- ekf_parameters.csv
  |-- metadata_birdseye.csv
  |-- event_counts.csv
  |-- Event/
  |..
```
where `Event` can be `Crash`, `NearCrash`, etc., and are catogarised events. More specifically about the .csv files, `ekf_parameters.csv` saves the parameters used in the extended Kalman filter (EKF) for trajectory reconstruction, `metadata_birdseye.csv` contains the metadata of the recorded events, and `event_counts.csv` concludes the number of events in each category. 

The data structure inside each `Event` folder is as follows:
- `plots_ego_ekf.pdf` where the plots of trajectory reconstruction for all events in the category using EKF are stored,
- `Ego_birdseye.h5` where the reconstructed ego vehicles' trajectories are stored,
- `Surrounding_birdseye.h5` where the reconstructed surrounding vehicles' trajectories are stored.

An exception is the `./SafeBaseline/` directory where the data are separated into 5 chuncks due to its large amount. Each chunk is marked by a suffix like `_0`~`_4` for plots (e.g., `plots_ego_ekf_0.pdf`), ego vehicle trajectories (e.g., `Ego_birdseye_1.h5`), and surrounding vehicle trajectories (e.g., `Surrounding_birdseye_2.h5`).

## Ego vehicle trajectory
The data in `Ego_birdseye.h5` is organised as a pandas dataframe with the following columns:
| Data nature   | Column name      | Data type | Description|
|---------------|------------------|-----------|------------|
| Raw           | 'event_id'       | int       | Index of events, consistent with video index|
| Raw           | 'timestamp'      | int       | Time index in the raw data, 1,000 timestamps = 1 second|
| Processed     | 'time'           | float     | Time in second for the convenience of trajectory smoothing|
| Raw           | 'speed_comp'     | float     | Ego vehicle speed indicated on speedometer in the raw data, unit: (m/s)|
| Raw           | 'yaw_rate'       | float     | Ego vehicle angular velocity around the vertical axis, unit: (deg/sec)|
| Raw           | 'acc_lat'        | float     | Ego vehicle acceleration in the lateral direction, unit: (g)|
| Raw           | 'acc_lon'        | float     | Ego vehicle acceleration in the longitudinal direction, unit: (g)|
| Raw           | 'brake'          | float     | On or off press of brake pedal (0.0: off, 1.0: on, 2: invalid data, 3: data not available, nan: null)|
| Raw           | 'wheel_steering' | float     | Angular position and direction of the steering wheel from neutral position, unit: (deg)|
| Raw           | 'turn_signal'    | float     | State of illumination of turn signals (0.0: off, 1.0: left, 2.0: right, 3.0: both, 254.0: invalid data, 255.0: data not available, nan:null)|
| Reconstructed | 'x_ekf'          | float     | Filtered coordinates of ego vehicle centroid position along the x-axis in the reconstructed coordinate system, unit: (m)|
| Reconstructed | 'y_ekf'          | float     | Filtered coordinates of ego vehicle centroid position along the y-axis in the reconstructed coordinate system, unit: (m)|
| Reconstructed | 'psi_ekf'        | float     | Filtered angle between ego vehicle heading direction and the x-axis (0,1) in the reconstructed coordinate system, unit: (rad)|
| Reconstructed | 'v_ekf'          | float     | Filtered ego vehicle speed in the heading direction, unit (m/s)|
| Reconstructed | 'omega_ekf'      | float     | Filtered yaw rate, i.e., angular velocity around the vertical axis, unit: (rad/sec)|
| Reconstructed | 'acc_ekf'        | float     | Filtered acceleration rate in the heading direction, unit: (m/s^2)|
| Processed     | 'event'          | int       | Wether the current moment is in an event, i.e., crash or near-crash (0: False, 1: True)|


## Surrounding vehicle trajectories

Similarly yet differently, the data in `Surrounding_birdseye.h5` is organised as a pandas dataframe with the following columns:
| Data nature   | Column name  | Data type | Description|
|---------------|--------------|-----------|------------|
| Raw           | 'event_id'   | int       | Index of events, consistent with `Ego_birdseye.h5` and video index|
| Processed     | 'target_id'  | int       | Index assigned for each surrounding vehicle, unique across the whole dataset|
| Raw           | 'timestamp'  | int       | Time index in the raw data, 1,000 timestamps = 1 second|
| Processed     | 'time'       | float     | Time in second for the convenience of trajectory smoothing|
| Raw           | 'local_dx'   | float     | Radar-detected position of surrounding vehicle front bumper in the lateral axis (perpendicular to ego vehicle heading direction, from left to right) of the ego vehicle local coordinate system unit: (m)|
| Raw           | 'local_dy'   | float     | Radar-detected position of surrounding vehicle front bumper in the longitudinal axis (ego vehicle heading direction) of the ego vehicle local coordinate system, unit: (m)|
| Raw           | 'delta_vx'   | float     | Radar-detected surrounding vehicle relative velocity component in the lateral axis (perpendicular to ego vehicle heading direction, from left to right) of the ego vehicle local coordinate system, unit: (m/s)|
| Raw           | 'delta_vy'   | float     | Radar-detected surrounding vehicle relative velocity component in the longitudinal axis (ego vehicle heading direction) of the ego vehicle local coordinate system, unit: (m/s)|
| Processed     | 'x'          | float     | Transformed position of surrounding vehicle front bumper in the x-axis of the global coordinate system reconstructed in `Ego_virdseye.h5`, unit: (m)|
| Processed     | 'y'          | float     | Transformed position of surrounding vehicle front bumper in the y-axis of the global coordinate system reconstructed in `Ego_birdseye.h5`, unit: (m)|
| Processed     | 'speed_comp' | float     | Transformed surrounding vehicle speed in its heading direction, unit: (m/s)|
| Reconstructed | 'x_ekf'      | float     | Filtered coordinates of surrounding vehicle front bumper along the x-axis in the reconstructed coordinate system, unit: (m)|
| Reconstructed | 'y_ekf'      | float     | Filtered coordinates of surrounding vehicle front bumper along the y-axis in the reconstructed coordinate system, unit: (m)|
| Reconstructed | 'v_ekf'      | float     | Filtered surrounding vehicle speed in its heading direction, unit (m/s)|
| Reconstructed | 'psi_ekf'    | float     | Filtered angle between surrounding vehicle heading direction and the x-axis (0,1) in the reconstructed coordinate system, unit: (rad)|
