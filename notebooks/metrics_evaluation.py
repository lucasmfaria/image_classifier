#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
from pathlib import Path
import pandas as pd
import numpy as np


# In[2]:


data_path = (Path(__file__).resolve().parent.parent / 'data' / 'experiments')
files = [open(file) for file in data_path.iterdir() if file.suffix == '.json']
jsons = [json.load(file) for file in files]


# In[3]:
n_std = 2

metric_cols = list(jsons[0]['metrics'].keys())
experiment_cols = list(jsons[0]['parameters'].keys())
cols = metric_cols + experiment_cols
df = pd.DataFrame(columns=cols)
for json_result in jsons:
    df = pd.concat([df, pd.DataFrame(dict(json_result['metrics'], **json_result['parameters']), columns=cols)], ignore_index=True)


# In[14]:

def mean_std(x):
    return str(x.mean().round(3)) + ' +- ' + str(n_std * x.std().round(3))

#class_names = ['normal_tissue', 'metastatic_tissue']
class_names = ['non_charizard', 'charizard']
def mean_std_agg_dict(x):
    return {class_names[1]: str(x.map(lambda x: x[class_names[1]]).mean().round(3)) + ' +- ' + str(n_std * x.map(lambda x: x[class_names[1]]).std().round(3)), class_names[0]: str(x.map(lambda x: x[class_names[0]]).mean().round(3)) + ' +- ' + str(n_std * x.map(lambda x: x[class_names[0]]).std().round(3))}

def agg_func_roc_auc(x):
    return {class_names[1]: x.map(lambda x: x[class_names[1]]).mean(), class_names[0]: x.map(lambda x: x[class_names[0]]).mean()}

def agg_func_pr_auc(x):
    return {class_names[1]: x.map(lambda x: x[class_names[1]]).mean(), class_names[0]: x.map(lambda x: x[class_names[0]]).mean()}

agg_dict = {
    #'fit_time': np.mean,
    #'test_time': np.mean,
    #'accuracy': np.mean,
    #'roc_auc': agg_func_roc_auc,
    #'pr_auc': agg_func_pr_auc,
    'fit_time': mean_std,
    'test_time': mean_std,
    'accuracy': mean_std,
    'roc_auc': mean_std_agg_dict,
    'pr_auc': mean_std_agg_dict,
}

metrics = df.groupby(experiment_cols).agg(agg_dict)


# In[33]:


metrics.reset_index(inplace=True)
metrics = metrics.set_index(metrics.SAMPLE_DATASET.str.split('_').map(lambda x: x[-1]).rename('MODEL')).drop(experiment_cols, axis=1)


# In[37]:

for col in metrics.columns:
    if isinstance(metrics[col].iloc[0], dict):
        for class_name in metrics[col].iloc[0].keys():
            print(metrics[col].map(lambda x: x[class_name]).rename(col + '_' + class_name))
    else:
        print(metrics[col])
