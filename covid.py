#%% Import modules and set directories
import time
start_time = time.time()

import numpy as np
import scipy as sp
from PIL import Image as im
from PIL import ImageDraw
import sympy as smp
import sys
import pandas as pd
pd.set_option('display.max_rows',100)
pd.set_option('display.min_rows',100)
pd.set_option('display.max_columns',100)
import seaborn as sns

import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib import figure
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
plt.rcParams['figure.figsize'] = [25,25];#default plot size
plt.rcParams['font.size']=48;
plt.rcParams['lines.linewidth'] = 8
# np.set_printoptions(threshold=sys.maxsize)
# from matplotlib import rc
# rc('text', usetex=True)

# np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(threshold=100000)

import itertools as iter
import pyttsx3
engine = pyttsx3.init()
from datetime import datetime,date
import os
import re
import csv

os.chdir(r'C:\Users\bcalverley\OneDrive - Scripps Research\Documents\0Balch lab')
import to_precision

#%% Import data
cell_time = time.time()

fileName = '12_15_21'
# proteinLength = 29903
os.chdir(r'C:\Users\bcalverley\OneDrive - Scripps Research\Documents\0Balch lab\0COVID')
covid = pd.read_csv('BCC collated covid data.csv')

# Only needed if importing from scratch:
# proteins_data = pd.read_excel('unipCov2Chain_UCSC_Wuh1_edited_CNBC_3.xlsx')
# proteins_data.index = proteins_data.name
# proteins_data.drop('ORF10',inplace=True)
# covid_raw = pd.read_csv(fileName+'.csv')
# covid = covid_raw.copy()
# # covid = covid[covid['Variant Type']=='SNP']
# # covid['VarSeqP'] = covid.Start/proteinLength
# covid['IR'] = (covid.infection_rate - np.min(covid.infection_rate))/(np.max(covid.infection_rate) - np.min(covid.infection_rate))
# covid['FR'] = (covid.fatality_rate - np.min(covid.fatality_rate))/(np.max(covid.fatality_rate) - np.min(covid.fatality_rate))
# covid = covid[(covid.countries>=3)&(covid.counts>=3)].copy()
# covid = covid[covid.AA.notna()].copy()
# covid['ProtPos'] = covid[covid.AA.notna()].AA.str.extract('(\d+)')[0].astype('int64')
#
# prot_SEL = proteins_data.position.copy()
# prot_SEL = prot_SEL.str.extract('amino acids (\d+)-(\d+)')
# prot_SEL = prot_SEL.fillna(0).astype('int64')
# prot_SEL.rename(columns={0:'protStart',1:'protEnd'},inplace=True)
# prot_SEL['length'] = prot_SEL['protEnd'] - prot_SEL['protStart'] + 1
# # proteins_data['length'] = prot_SEL[1] - prot_SEL[0] + 1
#
# covid['Protein'] = '-'
# for cc in covid.index:
#     for prot in proteins_data.index:
#         if covid.loc[cc,'Start'] in range(proteins_data.loc[prot,'chromStart'],proteins_data.loc[prot,'chromEnd']):
#             covid.loc[cc,'Protein'] = prot
# covid = covid[covid.Protein != '-'].copy()
# covid['VarSeqP'] = [(covid.loc[idx,'ProtPos'] - prot_SEL.loc[covid.loc[idx,'Protein'],'protStart'] + 1)/(prot_SEL.loc[covid.loc[idx,'Protein'],'length']) for idx in covid.index]
#
# covid.to_csv('BCC collated covid data.csv')
# proteins_data.length.to_csv('protein lengths.csv')

print('---Cell run time:',round(np.floor((time.time() - cell_time)/60)),'minutes,',round((time.time() - cell_time)%60),'seconds---')
print("Cell completed:",datetime.now())

#%%
cell_time = time.time()

def kriging_output(innie,note,X,Y,Z,norm=False,save=False,scatter=True,scatterSave=False,yzscatter=True,yzscatterSave=False,yzLogPlot=False,scatLog=False,yzLog=False):
    return_dir = os.getcwd()
    export_name = note+'-'+X+'-'+Y+'-'+Z
    data_directory = 'C:\\Users\\bcalverley\\OneDrive - Scripps Research\\Documents\\0Balch lab\\0COVID\\Kriging\\'+export_name
    if not os.path.exists(data_directory):
        os.mkdir(data_directory)
    os.chdir(data_directory)
    df = innie.copy()
    df = df[[X,Y,Z]]
    df.columns=['x','y','z']
    df = df.dropna()
    if yzLog:
        df.y = np.log(df.y)
        df.z = np.log(df.z)
    if norm:
        # df.x = df.x/proteinLength
        df.y = (df.y - np.min(df.y))/(np.max(df.y)-np.min(df.y))
        df.z = (df.z - np.min(df.z))/(np.max(df.z)-np.min(df.z))
    if yzscatter:
        fig,ax = plt.subplots()
        ax.scatter(df.z,df.y,s=50)
        ax.set_xlabel(Z,fontweight='bold')
        ax.set_ylabel(Y,fontweight='bold')
        ax.grid(True)
        if yzLogPlot:
            ax.set_xscale('log')
            ax.set_yscale('log')
        else:
            ax.set_xlim(-.01,1.01)
            ax.set_ylim(-.01,1.01)
        ax.set_aspect(1./ax.get_data_ratio())
        fig.tight_layout()
        plt.show()
        if yzscatterSave:
            fig.savefig(export_name+'-yzscatter.png')
    if save:
        df.groupby(['x','y']).mean().to_csv(export_name+'.csv')
    if scatter:
        fig,ax = plt.subplots()
        ax.scatter(df.x,df.y,c=df.z,cmap='plasma',s=50,zorder=5)
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cm.get_cmap('plasma'), norm=mpl.colors.Normalize(vmin=0, vmax=1)))
        ax.set_xlabel(X,fontweight='bold')
        if yzLog:
            ax.set_ylabel('log '+Y,fontweight='bold')
            cb.set_label('log '+Z,fontweight='bold')
        else:
            ax.set_ylabel(Y,fontweight='bold')
            cb.set_label(Z,fontweight='bold')
        ax.grid(True)
        if scatLog:
            ax.set_yscale('log')
        else:
            ax.set_xlim(-.01,1.01)
            ax.set_ylim(-.01,1.01)
        ax.set_aspect(1./ax.get_data_ratio())
        fig.tight_layout()
        plt.show()
        if scatterSave:
            fig.savefig(export_name+'-scatter.png',bbox_inches='tight')
    os.chdir(return_dir)
    return df


print('---Cell run time:',round(np.floor((time.time() - cell_time)/60)),'minutes,',round((time.time() - cell_time)%60),'seconds---')
print("Cell completed:",datetime.now())

#%% Export for kriging by protein
cell_time = time.time()

# spans = {'nsp1':[1,180],'nsp2':[181,818],'nsp3':[819,2763],'nsp4':[2764,3263],'3CL-PRO':[3264,3569],'nsp6':[3570,3859],'nsp7':[3860,3942],'nsp8':[3943,4140],'nsp9':[4141,4253],
#     'nsp10':[4254,4392],'nsp11':[4393,4405],'nsp12':[4393,5324],'nsp13':[5325,5925],'nsp14':[5926,6452],'nsp15':[6453,6798],'nsp16':[6799,7096],'spike':[21563,25384],
#     'orf3a':[25393,26220],'orf4':[26245,26472],'orf5':[26523,27191],'orf6':[27202,27387],'orf7a':[27394,27759],'orf7b':[27756,27887],'orf8':[27894,28259],'orf9':[28274,29533],'orf10':[29558,29674]}
# # lengths = {}
# # for prot in spans:
# #     lengths[prot] = spans[prot][1] - spans[prot][0] + 1

for prot in np.unique(covid.Protein):
    kriging_output(covid[covid.Protein==prot],fileName+'-'+prot+'-all_mutations','VarSeqP','IR','FR',yzLogPlot=True,scatLog=True,save=True,scatterSave=True,yzscatterSave=True)

for prot in np.unique(covid.Protein):
    kriging_output(covid[covid.Protein==prot],fileName+'-'+prot+'-all_mutations-IR_FR_indinorm','VarSeqP','IR','FR',norm=True,yzLogPlot=True,scatLog=True,save=True,scatterSave=True,yzscatterSave=True)

for prot in np.unique(covid.Protein):
    kriging_output(covid[covid.Protein==prot],fileName+'-'+prot+'-all_mutations-IR_FR_indinorm_and_log','VarSeqP','IR','FR',norm=True,yzLog=True,yzLogPlot=True,scatLog=False,save=True,scatterSave=True,yzscatterSave=True)

print('---Cell run time:',round(np.floor((time.time() - cell_time)/60)),'minutes,',round((time.time() - cell_time)%60),'seconds---')
print("Cell completed:",datetime.now())
