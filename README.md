## NASA C-MAPSS RUL prediction
----

## Problem Statement
In general, an unscheduled engine maintenance is expensive and disruptive when a failure isn't foreseen. This can trigger a chain reaction causing grounded aircraft, rushed part sourcing, and missed flight schedules.

In this project, with the help of engine sensor readings (temperatures, pressures, speeds) we can predict the Remaining Useful Life (RUL) i.e. the number of operational cycles left before an engine requires maintenance.
With the help of RUL, we can anticipate when a maintenance window is likely to occur and plan for that maintenance, like procuring parts and tools, making shop available to accept that engine.

## Dataset
- Source: NASA - https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
- Subset: FD001 - chosen for its single operating condition and single fault mode.
    - `train_FD001.txt` contains sensor data recorded until failure.
    - `test_FD001.txt` contains sensor data of 100 engines, truncated before failure.
    - `RUL_FD001.txt` contains the true remaining cycles at each test engine's cutoff point.
- Structure: 100 engines, 26 raw columns. Narrowed to 19 columns after preprocessing (14 sensor readings + `unit_number`, `time_cycle`, and `op_setting_1/2/3`), though the operational settings were ultimately excluded from model training, since FD001 runs under a single operating condition and they carry no useful variance.


## Methodology
**1. Target construction:**
In the training data, the RUL is derived by getting the max cycle of that specific engine unit and then subtracting the current cycle count from the max cycle. This gives us the RUL of that specific engine at each cycle.
`RUL = max_cycle - time_cycle`
Additionally, we have clipped the RUL values at 125 because without clipping, the model would be forced to distinguish between RUL values like 250 and 200 during a period where sensor readings show no meaningful difference and adds noise rather than learnable signal.


**2. EDA Findings:**
The raw data contains 21 sensor readings for which we have filtered the engine 1 data and plotted them with respect to cycles using matplotlib's subplot functions. By observing the plots, we narrowed it down to 14 sensors, dropping the 7 that showed a flat trend: `s1, s5, s6, s10, s16, s18, s19`.
When looking at sensor plot (e.g. s4) we can observe that initially the readings are stable then they show a trend with increasing variance as engine nears failure.

**3. Preprocessing:**
- Using the plots of sensors, I have dropped the 7 sensors that had a flat trend from the DataFrame.
- Since sensors vary widely in scale and units (e.g. temperatures vs. pressures vs. speeds), `StandardScaler` was applied to normalize them.
- Before scaling, the training data was split into train/validation sets at the engine level (not row level — more on this below) to prevent leakage. `StandardScaler` was fit **only** on the training sensors, then used to transform (not re-fit) the validation and test sensors.

**4. Train/Validation strategy:**
Splitting the data randomly by row would cause cycles from the same engine to leak across train and validation sets, since RUL for a given engine is highly correlated across adjacent cycles. Instead, we split at the engine level: the 100 unique `unit_numbers` were split 80/20 using `train_test_split` with `random_state = 42` for reproducibility, then the corresponding engines' rows were pulled into `train_df/val_df` using the `.isin()` function.

**5. Models tried:**
First I set a naive baseline by predicting the mean RUL for every row, as a floor any real model should beat. Linear Regression came next as the baseline model: it is simple, fast, and easy to interpret. I then tried Random Forest, expecting it to handle non-linear patterns such as the growing sensor noise near failure. 
I intentionally kept the hyperparameters to their default since this was a first end-to-end comparison, and tuning is a next step. Models were scored with RMSE, which is in the same units as RUL (cycles) and penalizes large errors more. I did not use the asymmetric scoring function from the original NASA paper.

**6. Why validate before touching the test set:**
I split the 100 training engines 80/20 by engine. The 20 held-out engines are complete run-to-failure trajectories the model never trained on, so I used them to compare models and catch problems. This paid off when my first validation run showed a naive baseline error (65.7) larger than the standard deviation of the clipped RUL target (about 41.7), which is impossible if the target is really clipped. Tracing it showed the clipping hadn’t reached the training targets. After fixing it, validation RMSE dropped (Linear Regression 38.1 → 20.4). I kept the official test set untouched until the end and used it once, so the final score is an honest estimate on unseen engines.


## Results
| Model | Validation RMSE | Test RMSE |
|-------|--------|--------|
| Naive Baseline | 41.7 | n/a |
| Linear Regression | 20.4 | 20.8 |
| Random Forest | 17.3 | 16.8 |

From the above results, Random Forest model performed better than Linear Regression and reduced validation RMSE by about 58.6% relative to the naive baseline. I can also see that the validation and test scores are close, which which suggests the models generalize well to unseen engines. These results show that the Random Forest was a good choice for predicting RUL, with about 15% lower validation RMSE than Linear Regression.

These results are limited by the use of a single sub-dataset (FD001), scoring with plain RMSE rather than NASA’s asymmetric scoring function, and a test score based on only 100 engines.”.


## Lessons Learned

1. Initially, I did the scaler transformation before splitting the data into Train/Val split. This computed the mean and std using all 100 engines which leaked information from the validation engines into training. I learned to scale only after splitting into train/val or train/test. Additionally, fit the scaler on the training data only, then use `.transform()` (not `.fit_transform()`) on validation and test.

2. During the first run of the full code, I didn't realize that I had left out the RUL clip from the training targets, so I trained my model on unclipped RUL. This made my results look plausible even though they were wrong. I noticed the problem when re-running the notebook changed my results. Comparing the naive baseline error (65.7) with the target's std (41.7) showed the training targets were unclipped. With this I learned to sanity-check a model against a naive baseline and the target's own spread, and to make sure the same convention applies to the training target, the validation target, and the test ground truth.


## How to Run
1. Clone the repo:
   ```bash
   git clone <repo-link>
   cd <repo-folder>
   ```
2. Dependencies: Test with Python 3.13 or newer, numpy, pandas, matplotlib, seaborn, scikit-learn v1.4 or newer and Jupyter notebook
   ```bash
   pip install numpy pandas matplotlib seaborn "scikit-learn>=1.4" jupyter
   ```
3. This repo doesn't contain the data. Download it from https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data and place the `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` in a `data/` folder in the repo root.
4. Open the `main.ipynb` file in the root directory in VS Code or Jupyter and use Restart and Run All.