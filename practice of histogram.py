#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
data=np.random.randn(1000)
plt.hist(data);


# In[2]:


plt.hist(data,bins=30,normed=True,alpha=0.5,histtype='stepfilled',color='steelblue',
        edgecolor='none');


# In[ ]:





# In[3]:


plt.hist(data,bins=30,density=True,alpha=0.5,histtype='stepfilled',color='steelblue',
        edgecolor='none');


# In[4]:


x1=np.random.normal(0,0.8,1000)
x2=np.random.normal(-2,1,1000)
x3=np.random.normal(3,2,1000)
kwargs=dict(histtype='stepfilled',alpha=0.3,density=True,bins=40)
plt.hist(x1,**kwargs)
plt.hist(x2,**kwargs)
plt.hist(x3,**kwargs);


# In[5]:


counts,bin_edges=np.histogram(data,bins=5)
print(counts)


# In[6]:


d={'student1':pd.Series([85.,72.],index=['maths','science']),
  'student2':pd.Series([62.,70.,55.],index=['maths','science','english']),
  'student3':pd.Series([45.,48.,70.],index=['maths','science','english'])}
df=pd.DataFrame(d)
print(df.head())


# In[7]:


plt.hist(df);


# In[11]:


plt.hist(df)
plt.legend()
plt.title("SCORECARD");


# In[12]:


mean=[0,0]
cov=[[1,1],[1,2]]
x,y=np.random.multivariate_normal(mean,cov,10000).T
plt.hist2d(x,y,bins=30,cmap='Blues')
cb=plt.colorbar()
cb.set_label('counts in bin') # Two dimensional histogram using plt.hist2d


# In[13]:


plt.hexbin(x,y,gridsize=30,cmap='Blues')
cb=plt.colorbar(label='count in bin') # Two dimensional histogram using plt.hexbin


# In[23]:


from scipy.stats import gaussian_kde
data=np.vstack([x,y])
kde=gaussian_kde(data)
xgrid=np.linspace(-3.5,3.5,40)
ygrid=np.linspace(-6,6,40)
xgrid,ygrid=np.meshgrid(xgrid,ygrid)
z=kde.evaluate(np.vstack([xgrid.ravel(),ygrid.ravel()]))
plt.imshow(z.reshape(xgrid.shape),origin='lower',aspect='auto',extent=[-3.5,3.5,-6,6],cmap='Blues')
cb=plt.colorbar()
cb.set_label("density")


# In[24]:


plt.style.use('classic')
x=np.linspace(0,10,1000)
fig,ax=plt.subplots()
ax.plot(x,np.sin(x),'-b',label='Sine')
ax.plot(x,np.cos(x),'--r',label='Cosine')
ax.axis('equal')
leg=ax.legend();


# In[26]:


ax.legend(fancybox=True,framealpha=1,shadow=True,borderpad=1,loc='lower center',ncol=2)
fig


# In[31]:


x=np.linspace(0,10,1000)
y=np.linspace(0,25,1000)
lines=plt.plot(x,y)
plt.legend(lines[:2],['first','second']);


# In[33]:


cities=pd.read_csv('https://raw.githubusercontent.com/jakevdp/data-USstates/master/state-population.csv')
print(cities.head())


# In[38]:


df=pd.read_csv('https://raw.githubusercontent.com/allisonhorst/palmerpenguins/1a19e36ba583887a4630b1f821e3a53d5a4ffb76/data-raw/penguins_raw.csv')
print(df.head())


# In[41]:


import pymysql
con=pymysql.connect(host='localhost',user='test',password='',db='palmerpenguin')
df=read_sql(f'''SELECT*FROM penguins''',con)
print(df.head())


# In[44]:


df=pd.read_csv('https://raw.githubusercontent.com/allisonhorst/palmerpenguins/1a19e36ba583887a4630b1f821e3a53d5a4ffb76/data-raw/penguins_raw.csv',
           chunksize=100)
print(df)


# In[45]:


df_list=[]
for df in df:
    df_list.append(df)
    df=pd.concat(df_list,sort=False)
    print(df)


# In[47]:


if len(df)>0:
    print(f'Length of df{len(df)},number of columns{len(df.columns)},dimensions{df.shape},number of elements{df.size}')
else:
          print(f'problem loading df,df is empty')


# In[48]:


df.info()


# In[49]:


df.info(memory_usage='deep')


# In[ ]:




