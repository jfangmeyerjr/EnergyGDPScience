import pandas as pd
import numpy as np 

gdp = pd.DataFrame.from_csv('course1_downloads/world_bank.csv')
gdp


# In[126]:

gdp  = gdp[4:] #cut header
gdp 


# In[127]:

gdp.reset_index(inplace = True) #make country names editable by taking them out of index
gdp


# In[128]:

#"Korea, Rep.": "South Korea", 
#"Iran, Islamic Rep.": "Iran",
#"Hong Kong SAR, China": "Hong Kong"
gdp.replace(to_replace = ['Korea, Rep.', 'Iran, Islamic Rep.','Hong Kong SAR, China'], 
               value = ['South Korea','Iran','Hong Kong'], inplace = True)
gdp


# In[129]:

gdp.set_index(keys = 'Data Source', inplace = True)
print(gdp)


# In[130]:

GDP = gdp
