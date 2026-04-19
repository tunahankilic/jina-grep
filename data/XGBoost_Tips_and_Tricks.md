---
title: "XGBoost Tips and Tricks"
source: "https://www.kaggle.com/writeups/cdeotte/xgboost-tips-and-tricks#3310814"
author:
published:
created: 2026-04-19
description: "Tips and Tricks from years of using XGBoost!"
---
##### XGBoost Tips and Tricks

Tips and Tricks from years of using XGBoost!

Having used XGBoost for many years now, I would like to share some tips and tricks with Kaggle. These techniques have helped me win competitions and build and deploy successful machine learning solutions.

Let's discuss Data Science Foundations, XGBoost Fundamentals, Building and Optimizing Models, Scaling XGBoost for Large Data, and Deployment and Inference:

## Data Science Foundations

Three common techniques in all data science projects are Fast Experimentation, Local Validation, and Exploratory Data Analysis (EDA). Let's discuss these before talking about XGBoost.

## Fast Experimentation

The secret to success in any machine learning endeavor is Fast Experimentation. We need to make our local preprocess, feature engineering, model training, inference and evaluation as fast as possible. Then each day we can try more ideas and discover more things to improve our model's performance.

Using GPU vs. CPU is the number one trick to accelerate speed. My favorite libraries are [NVIDIA cuDF](https://developer.nvidia.com/topics/ai/data-science/cuda-x-data-science-libraries/cudf) and [cuML](https://developer.nvidia.com/topics/ai/data-science/cuda-x-data-science-libraries/cuml) which accelerate dataframe operations and ML model train/infer respectively. And adding GPU to XGB is as easy as adding the parameter `"device":"cuda"`!  
![](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/Nov-2025/experiments.png)

## Local Validation

In order to evaluate experiments and determine what work best we need reliable local validation. The best validation is KFold. This uses all the train data to evaluate model performance. We need to design KFold to mimic the relationship between test data and train data.

So if test data contains unseen patients (for a medical prediction model), then we need to use GroupKFold on patients locally so that our validation dataset has unseen patients similar to test data. Another example is if test is timeseries occurring after train data, we need to design our validation and train data to mimic this.  
![](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/Nov-2025/kfold.png)

## Exploratory Data Analysis

The more we understand our data and how features relate to targets, the better we can engineer features and design our model architecture. Therefore analyzing all aspects about our data helps  
![](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/Nov-2025/eda.png)

## XGBoost Fundamentals

What is XGBoost? XGB is an ensemble of decision trees. Below, we will discuss this and why it is important. But first, let's take a look at some XGB code. When writing code there are two main APIs:

## XGB APIs

XGBoost is simple to use. With Scikit-Learn API, we just create the model then call train and predict. With Native Python API, first we package our data, then create our model. Then call train and predict. See the two examples below.

### Native XGBoost Python API

```
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'reg:squarederror',
    'max_depth': 3,
    'eta': 0.1
}
bst = xgb.train(params, dtrain, num_boost_round=100)
preds = bst.predict(dtest)
```

Pros:

- More features than scikit-learn API (e.g., custom learning rate per iteration).
- Fine-grained training control (incremental training, callbacks, etc.).

Cons:

- Less convenient for beginners compared to sklearn API.

### Scikit-Learn API

```
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    objective='reg:squarederror'
)
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

Pros:

- Fits naturally into scikit-learn workflows (e.g., GridSearchCV, Pipeline).
- No need to manually create DMatrix.

Cons:

- Not all advanced features from the native API are exposed directly.

## What is XGBoost?

XGBoost is an ensemble of decision trees. Each subsequent tree is fit on the error of previous trees. There are two important properties to keep in mind when working with XGB:

- Decision trees split numbers and therefore only care about ordering not distribution
- Decision trees do not extrapolate given input variables outside train input range.

Below is an example of a single decision tree:  
![](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/Nov-2025/tree.png)

## Building & Optimizing Models

## Baselines - Our First Model

One of the reasons that XGB is most data scientists' favorite is that we can create a model without performing any preprocessing! Wow! We can leave missing data as is, we can leave categorical features as is, and we can leave numerical features as is. Other models often require us to impute missing, encode categorical, and normalize numerics. Below is a diagram showing how easy it is to create our first model with XGBoost:  
![](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/Nov-2025/first.png)

## Baselines - Our First Code

Here is the code using Native Python API for the diagram above. This code performs KFold to train and validate an XGB model.

```
oof_preds = np.zeros(len(train))

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(train)):

    X_tr, X_val = train.loc[train_idx,FEATURES], train.loc[val_idx,FEATURES]
    y_tr, y_val = train.loc[train_idx,TARGET], train.loc[val_idx,TARGET]
    dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
    dval   = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=10_000,
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=100,
        verbose_eval=200
    )
    oof_preds[val_idx] = model.predict(dval, 
        iteration_range=(0, model.best_iteration + 1))

print( roc_auc_score( train[TARGET], oof_preds ) )
```

## XGBoost Hyperparameters

Having to determine which hyperparameters to use can be overwhelming. But actually, we don't need to worry too much. The default parameters are good and we only need to turn a few knobs to get the majority of performance from XGB. Here are the main parameters.

```
params = {
    # KEY HYPERPARMETERS
    "objective": "binary:logistic",  
    "eval_metric": "auc",           
    "learning_rate": 0.1,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "device": "cuda",

    # REGULARIZATION
    "min_child_weight": 1,  
    "gamma": 0,             
    "lambda": 1,            
    "alpha": 0,
```

## Key Hyperparameters

The key hyperparameters are objective, eval\_metric, learning rate, max\_depth, subsample, colsample\_bytree, and device. For medium to large datasets, we should use `"device":"cuda"` for speed. Then we set our problem type with `objective` and `eval_metric`. We start our experiments with `"learning_rate":0.1` and decrease later for more performance.

The main three knobs to turn are `max_depth`, `subsample` and `colsample_bytree`. Begin with 6, 0.8, 0.8 respectively. Then try `max_depth` 3 thru 12. Afterwards try `colsample_bytree` 0.3 thru 0.9. And that's it! Turning only these 2 knobs will achieve 95%+ performance with XGBoost.

## Hyperparameter Optimization

If we wish to squeeze a little more performance from XGBoost, we can tune the regularization and/or play with additional parameters below. We can either tune by hand or use an optimization library like Optuna.

```
# MORE PARAMETERS
"scale_pos_weight": 1, 
"grow_policy": "depthwise",
"max_leaves": 64, 
"tree_method": "hist",  
"max_bin": 256,
```

## Feature Engineering

Personally, I don't spend much time optimizing hyperparameters. After I find a good `max_depth` and `colsample_bytree`, I usually invest my time in feature engineering where I believe we can achieve the biggest gains.

We can write an entire book about Feature Engineering. The most powerful technique to keep in mind is creating lots of new categorical features and then encoding them. The best encoding technique is to groupby categorical features and aggregate a numeric column statistic. If we aggregate a statistic of the target, this is called Target Encoding and we need to use proper techniques to avoid leakage. See the diagram below for an illustration of groupby aggregation encoding.

- There are numeric columns and categorical columns
- We can convert numeric to categorical with binning
- We can combine columns to make new columns
- We can split columns to make new columns
- Groupby categorical column and aggregate statistics of numeric column. (See example diagram below).
- Process categorical columns with categorical encoding
	- one hot encoding
		- label encoding
		- target encoding
		- count encoding
- NVIDIA cuDF and GPUs are essential to accelerate searching 1000s of feature engineering ideas!  
	![](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/Nov-2025/fe.png)

## Single Model Feature Engineering Wins Kaggle Comps!

Using feature engineering alone can win Kaggle competitions! The following techniques lead to recent victories!

- Put Numerical Column(s) into Bins
- Combine Categorical Columns
- Groupby(COL1)\[COL2\].agg(STAT)
- Target Encoding is Most Powerful!
- Groupby(COL1)\['Price'\].agg(“mean”)
- Groupby(COL1)\['Price'\].agg(HISTOGRAM BINS)
- Groupby(COL1)\['Price'\].agg(QUANTILES)
- NVIDIA cuDF accelerates groupby up 50x!

![](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/Nov-2025/win.png)

## Scaling XGBoost For Large Data

The main techniques for scaling XGB for large data are: reducing data types, using QuantileDMatrix from XGB v2.0 or v3.0, and using multiple GPUs with DASK XGB

## Reducing Data Types

We save memory by reducing datatypes to sizes only as big as we need:  
![](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/Nov-2025/reduce.png)

## XGBoost v3.0

Using QuantileDMatrix allows us to train with larger datasets without increasing our CPU RAM nor GPU VRAM. And using ExtMemQuantileDMatrix allows to push it even further! In both cases it is achieved with better memory management.

```
# OPTION A
dtrain = xgb.DMatrix(X_tr, label=y_tr)
dval   = xgb.DMatrix(X_val, label=y_val)

# OPTION B
iterator = Iterator() # Custom Data Loader
dtrain = xgb.QuantileDMatrix(iterator, max_bin=256)
dval   = xgb.DMatrix(X_val, label=y_val)

# OPTION C
iterator = Iterator() # Custom Data Loader
dtrain = xgb.ExtMemQuantileDMatrix(iterator, max_bin=256)
dval   = xgb.DMatrix(X_val, label=y_val)

with xgboost.config_context(use_rmm=True):
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        evals=[(dtrain, "train"), 
               (dval, "valid")],
 )
```

## DASK XGBoost

If our system has multiple GPUs, we can put them all to use with DASK XGBoost.

```
from dask.distributed import Client, LocalCluster
cluster = LocalCluster(n_workers=4, threads_per_worker=1)
client = Client(cluster)

dtrain = xgb.dask.DaskDMatrix(client, X_train, y_train)
dval = xgb.dask.DaskDMatrix(client, X_valid, y_valid)

booster = xgb.dask.train(
    client,
    params=params,
    dtrain=dtrain,
    num_boost_round=10_000,
    evals=[(dtrain, "train"), (dval, "valid")],
    early_stopping_rounds=100,
    verbose_eval=200
)['booster']

val_preds = xgb.dask.predict(client, booster, dval)

client.close()
cluster.close()
```

## Multiple RecSys Wins via Fast Experimentation!

Using Data Reduction, QuantileDMatrix, and DASK XGB, NVIDIA achieved **250x Speedup w/ 4xGPU vs 1xCPU** and **25x Speedup w/ 4xGPU vs 20xCPU**. This speedup enabled faster experimentation and lead Team NVIDIA to win multiple RecSys competitions!  
![](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/Nov-2025/recsys.png)

## Deployment & Inference

Two additional inference tricks are using NVIDIA cuML FIL and Refit on Full Data.

## NVIDIA cuML Forest Inference Library (FIL)

NVIDIA FIL allows us to accelerate inference.

```
import xgboost as xgb
from cuml.fil import ForestInference

xgb_model = xgb.XGBClassifier(device="cuda")  
xgb_model.fit(X_train, y_train)

xgb_model.save_model("model.ubj")  # or "model.json"

fil = ForestInference.load("model.ubj")  # auto-detects 

# (Optional) auto-tune FIL for your typical batch size
fil.optimize(batch_size=100_000)

y_hat = fil.predict(X_val)            # class labels)
p_hat = fil.predict_proba(X_val)      # probabilities
```

## Refit on Full Data

After finding optimal hyperparameters with KFold, we can retrain a single model using 100% train data. This is a common trick on Kaggle to boost leaderboard score!

- Using 100% train improves model performance compared with (K-1)/K% data
- During inference we have 1 model instead of K models.
- Train with number of epochs equal to K/(K-1) times optimal KFold number of epochs.

![](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/Nov-2025/full.png)

## Summary and Highlights

- Data Science Foundations
- XGBoost Fundamentals
- Building and Optimizing Models
- Scaling XGBoost for Large Data
- Deployment & Inference

![](https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/Nov-2025/kdd.png)