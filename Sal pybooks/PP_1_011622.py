import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime as dt
from datetime import timedelta
from scipy.signal import lfilter
import pickle as pickle
import sys

# load required files
dates = pd.read_table('dates.txt',header=None,index_col=0)
old_date = dates.loc['old_date'].values[0]
new_date = dates.loc['new_date'].values[0]

with open(f"../Input_files/jh_daily_ir_{new_date}.pkl",'rb') as file: #Contains only country name for each date
    jh_daily_ir = pickle.load(file)
with open(f"../Input_files/jh_data_{new_date}.pkl",'rb') as file:
    jh_data=pickle.load(file)
with open(f"../Input_files/seq_country_counts_{new_date}.pkl",'rb') as file:
    sequence_country_counts = pickle.load(file)
with open(f"../Input_files/basis_vector_IR_FR_PP_{new_date}.pkl",'rb') as file:
    PP = pickle.load(file)
with open(f"../Input_files/aggregated_mutations_window00_{dt.strftime(dt.strptime(old_date,'%m_%d_%y') + timedelta(days=1),'%m_%d_%y')}_{new_date}.pkl",'rb') as file:
    aggregated_mutations_window = pickle.load(file)




#Looping over single days
list_dates=jh_daily_ir.columns[9::]

# For this step, create a few parallel scripts, each cycling through ~15 days. For example, if your update covers 30 days, you would create 2 scripts for each IR, PP and lagged_FR.
# Get the indexes corresponding to days by looking up list_dates as before.

# CHANGE INDEXES
for date in tqdm(list_dates[684:700]):
#Looping over single variants
    for var in aggregated_mutations_window[date].keys():
        #For each variant count countries,
        #totalCounts=len(aggregated_mutations_window[date][var][0])
        for counted in Counter(aggregated_mutations_window[date][var]).items(): #counted is country and counts
            if sequence_country_counts[date][counted[0]]==0:
                PP[date][var].append(jh_data.loc[jh_data['Country_Region']==counted[0],'Population'].sum())
            else:
                PP[date][var].append(jh_data.loc[jh_data['Country_Region']==counted[0],'Population'].sum())

with open(f"../Input_files/PP1_{new_date.replace('_','')}.pkl",'wb') as file:
    pickle.dump(PP,file,protocol=pickle.HIGHEST_PROTOCOL)
