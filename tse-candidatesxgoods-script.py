#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
from matplotlib import rcParams
import glob

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('pylab', 'inline')


# In[151]:


dfConsulta = pd.read_csv('./data/candidatos-consulta/consulta_cand_2018_BRASIL.csv', sep=";", encoding = 'latin1', error_bad_lines=False)
dfBens = pd.read_csv('./data/candidatos-bens/bem_candidato_2018_BRASIL.csv', sep=";", encoding = 'latin1', error_bad_lines=False)

dfMerged = dfConsulta.merge(dfBens, on='SQ_CANDIDATO')

dfMerged.head()


# In[114]:


dfMergedFilteredBens = dfMerged.filter(items=['NM_URNA_CANDIDATO','NR_PARTIDO','SG_PARTIDO','SQ_CANDIDATO','VR_BEM_CANDIDATO','CD_COR_RACA','DS_COR_RACA','CD_GRAU_INSTRUCAO','DS_GRAU_INSTRUCAO','CD_GENERO','DS_GENERO','NR_IDADE_DATA_POSSE','CD_CARGO','DS_CARGO','SG_UE_y'])


# In[115]:


dfMergedFilteredBens["VR_BEM_CANDIDATO"] = dfMergedFilteredBens["VR_BEM_CANDIDATO"].str.replace(',','.').astype(float)


# In[116]:


dfMergedFilteredBensSum = dfMergedFilteredBens.groupby(['NM_URNA_CANDIDATO','NR_PARTIDO','SG_PARTIDO','SQ_CANDIDATO','CD_COR_RACA','DS_COR_RACA','CD_GRAU_INSTRUCAO','DS_GRAU_INSTRUCAO','CD_GENERO','DS_GENERO','NR_IDADE_DATA_POSSE','CD_CARGO','DS_CARGO','SG_UE_y'])['VR_BEM_CANDIDATO'].sum().reset_index()


# In[117]:


dfMergedFilteredBensSum.to_csv('/home/angeloreale/candidatura-redebahia/teste-python/dados/dfMergedFilteredBensSum.csv', encoding = 'latin1')


# In[118]:


dfMergedFilteredBensSum.describe()


# In[119]:


fig = plt.figure(figsize=(12, 6))
age = fig.add_subplot(121)
goods = fig.add_subplot(122)

age.hist(dfMergedFilteredBensSum.NR_IDADE_DATA_POSSE, bins=80)
age.set_xlabel('Age')
age.set_title("Histogram of Age")

goods.hist(dfMergedFilteredBensSum.VR_BEM_CANDIDATO, bins=20)
goods.set_xlabel('Goods')
goods.set_title("Histogram of Candidate's Goods")

plt.show()


# In[120]:


import statsmodels.api as sm
from statsmodels.formula.api import ols

m = ols('VR_BEM_CANDIDATO ~ NR_IDADE_DATA_POSSE + CD_COR_RACA + CD_GENERO + CD_GRAU_INSTRUCAO + CD_GENERO + CD_CARGO', dfMergedFilteredBensSum).fit()
print (m.summary())


# In[122]:


sns.jointplot(x="NR_IDADE_DATA_POSSE", y="VR_BEM_CANDIDATO", data=dfMergedFilteredBensSum, kind = 'reg',fit_reg= True, size = 7)
plt.show()


# In[124]:


sns.jointplot(x="CD_GENERO", y="NR_IDADE_DATA_POSSE", data=dfMergedFilteredBensSum, kind = 'reg',fit_reg= True, size = 7)
plt.show()


# In[125]:


sns.jointplot(x="CD_GENERO", y="VR_BEM_CANDIDATO", data=dfMergedFilteredBensSum, kind = 'reg',fit_reg= True, size = 7)
plt.show()


# In[127]:


sns.jointplot(x="CD_COR_RACA", y="VR_BEM_CANDIDATO", data=dfMergedFilteredBensSum, kind = 'reg',fit_reg= True, size = 7)
plt.show()


# In[128]:


sns.jointplot(x="CD_GRAU_INSTRUCAO", y="VR_BEM_CANDIDATO", data=dfMergedFilteredBensSum, kind = 'reg',fit_reg= True, size = 7)
plt.show()


# In[129]:


sns.jointplot(x="CD_CARGO", y="VR_BEM_CANDIDATO", data=dfMergedFilteredBensSum, kind = 'reg',fit_reg= True, size = 7)
plt.show()


# In[147]:


with sns.axes_style(style='ticks'):
    g = sns.factorplot("CD_COR_RACA", "VR_BEM_CANDIDATO", "DS_GENERO", data=dfMergedFilteredBensSum, size=10, kind="box")
    g.set_axis_labels("Etnia", "Bens");
    g.savefig('factorplot01.png')


# In[ ]:




