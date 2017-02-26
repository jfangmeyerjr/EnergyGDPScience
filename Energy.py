# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.5** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource._
# 
# ---

# # Assignment 3 - More Pandas
# This assignment requires more individual learning then the last one did - you are encouraged to check out the [pandas documentation](http://pandas.pydata.org/pandas-docs/stable/) to find functions or methods you might not have used yet, or ask questions on [Stack Overflow](http://stackoverflow.com/) and tag them as pandas and python related. And of course, the discussion forums are open for interaction with your peers and the course staff.

# ### Question 1 (20%)
# Load the energy data from the file `Energy Indicators.xls`, which is a list of indicators of [energy supply and renewable electricity production](Energy%20Indicators.xls) from the [United Nations](http://unstats.un.org/unsd/environment/excel_file_tables/2013/Energy%20Indicators.xls) for the year 2013, and should be put into a DataFrame with the variable name of **energy**.
# 
# Keep in mind that this is an Excel file, and not a comma separated values file. Also, make sure to exclude the footer and header information from the datafile. The first two columns are unneccessary, so you should get rid of them, and you should change the column labels so that the columns are:
# 
# `['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']`
# 
# Convert `Energy Supply` to gigajoules (there are 1,000,000 gigajoules in a petajoule). For all countries which have missing data (e.g. data with "...") make sure this is reflected as `np.NaN` values.
# 
# Rename the following list of countries (for use in later questions):
# 
# ```"Republic of Korea": "South Korea",
# "United States of America": "United States",
# "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
# "China, Hong Kong Special Administrative Region": "Hong Kong"```
# 
# There are also several countries with numbers and/or parenthesis in their name. Be sure to remove these, 
# 
# e.g. 
# 
# `'Bolivia (Plurinational State of)'` should be `'Bolivia'`, 
# 
# `'Switzerland17'` should be `'Switzerland'`.
# 
# <br>
# 
# Next, load the GDP data from the file `world_bank.csv`, which is a csv containing countries' GDP from 1960 to 2015 from [World Bank](http://data.worldbank.org/indicator/NY.GDP.MKTP.CD). Call this DataFrame **GDP**. 
# 
# Make sure to skip the header, and rename the following list of countries:
# 
# ```"Korea, Rep.": "South Korea", 
# "Iran, Islamic Rep.": "Iran",
# "Hong Kong SAR, China": "Hong Kong"```
# 
# <br>
# 
# Finally, load the [Sciamgo Journal and Country Rank data for Energy Engineering and Power Technology](http://www.scimagojr.com/countryrank.php?category=2102) from the file `scimagojr-3.xlsx`, which ranks countries based on their journal contributions in the aforementioned area. Call this DataFrame **ScimEn**.
# 
# Join the three datasets: GDP, Energy, and ScimEn into a new dataset (using the intersection of country names). Use only the last 10 years (2006-2015) of GDP data and only the top 15 countries by Scimagojr 'Rank' (Rank 1 through 15). 
# 
# The index of this DataFrame should be the name of the country, and the columns should be ['Rank', 'Documents', 'Citable documents', 'Citations', 'Self-citations',
#        'Citations per document', 'H index', 'Energy Supply',
#        'Energy Supply per Capita', '% Renewable', '2006', '2007', '2008',
#        '2009', '2010', '2011', '2012', '2013', '2014', '2015'].
# 
# *This function should return a DataFrame with 20 columns and 15 entries.*

# In[87]:

import pandas as pd
#help(pd.read_excel)


# In[88]:

import pandas as pd
import numpy as np
energy = pd.read_excel('Energy Indicators.xls') 
energy = energy[16:] #cut out header


# In[89]:

del energy['Unnamed: 0'] #delete empty column


# In[90]:


energy = energy.reset_index() #reset index


# In[91]:

list(energy) #see column names


# In[92]:


energy.head()


# In[93]:

del energy['index'] #delete old index column


# In[94]:

energy.head()


# In[95]:

energy = energy[:227] # drop off footer


# In[96]:

#rename columns
#df.columns = ['a', 'b']
energy.columns = ['Index', 'Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']
energy.head()


# In[97]:

# make country names index
#DataFrame.set_index(keys, drop=True, append=False, inplace=False, verify_integrity=False)
energy.set_index('Index', inplace = True)
energy.head()


# In[98]:

#For all countries which have missing data (e.g. data with "...") make sure this is reflected as np.NaN values.
#DataFrame.fillna(value=None, method=None, axis=None, inplace=False, limit=None, downcast=None, **kwargs)
#d = d.applymap(lambda x: np.nan if isinstance(x, basestring) and x.isspace() else x)
energy = energy.replace('...', np.nan)
energy


# In[99]:

#Convert Energy Supply to gigajoules (there are 1,000,000 gigajoules in a petajoule)
energy['Energy Supply'] = energy['Energy Supply']*1000000
energy.head()
# No puedo ver población, pero lo puedo calcular de Energy Supply per Capita, también en gigajoules
# Albania vaya!


# In[100]:

#energy.loc['Republic of Korea']


# In[101]:

energy.loc['Republic of Korea']


# In[102]:

#"Republic of Korea": "South Korea",
#"United States of America": "United States",
#"United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
#"China, Hong Kong Special Administrative Region": "Hong Kong"
#DataFrame.replace(to_replace=None, value=None, inplace=False, limit=None, regex=False, method='pad', axis=None)
energy.replace(to_replace='Republic of Korea', value = 'South Korea', inplace = True, regex = True)


# In[103]:

energy.loc['Republic of Korea']


# In[104]:

energy.replace(to_replace = ('United States of America', 'United Kingdom of Great Britain and Northern Ireland',
                             'China, Hong Kong Special Administrative Region'), 
               value = ('United States','United Kingdom','Hong Kong'),
               inplace = True,
               regex = True)


# In[105]:

print(energy.loc['United States of America'])
print(energy.loc['United Kingdom of Great Britain and Northern Ireland'])
print(energy.loc['China, Hong Kong Special Administrative Region'])


# In[106]:

energy.replace(to_replace = 'United States20', value = 'United States', inplace = True, regex = True)
energy.loc['United States of America']


# In[107]:

energy.replace(to_replace = 'United Kingdom19', value = 'United Kingdom', inplace = True, regex = True)
energy.replace(to_replace = 'Hong Kong3', value = 'Hong Kong', inplace = True, regex = True)


# In[108]:

energy.loc['United Kingdom of Great Britain and Northern Ireland']


# In[109]:

energy.loc['China, Hong Kong Special Administrative Region']


# In[110]:

#df['col1'].str.contains('^')
#DataFrame.where(cond, other=nan, inplace=False, axis=None, level=None, try_cast=False, raise_on_error=True)
paren = energy.where(energy['Country'].str.contains('\('))



# In[111]:

paren.dropna()


# In[112]:

paren.head()


# In[113]:

names_with_paren = ('Bolivia (Plurinational State of)','Falkland Islands (Malvinas)',
                    'Iran (Islamic Republic of)',
                    'Micronesia (Federated States of)',
                   'Sint Maarten (Dutch part)',
                   'Venezuela (Bolivarian Republic of)')
names_with_paren


# In[114]:

new_names = ('Bolivia', 'Falkland Islands', 'Iran', 'Mironesia',
            'Sint Maarten', 'Venezuela')
new_names


# In[115]:

energy.replace(to_replace = names_with_paren, value = new_names, inplace = True) #con madre wey


# In[116]:

energy.loc['Switzerland']


# In[117]:

# find names with numbers
numbers = energy.where(energy['Country'].str.contains('1') | energy['Country'].str.contains('2')
            | energy['Country'].str.contains('3')
            | energy['Country'].str.contains('4')
            | energy['Country'].str.contains('5')
            | energy['Country'].str.contains('6')
            | energy['Country'].str.contains('7')
            | energy['Country'].str.contains('8')
            | energy['Country'].str.contains('9')
            | energy['Country'].str.contains('0'))
numbers


# In[118]:

numbers.dropna() # see only names with numbers


# In[119]:

old_names_list = numbers['Country'].dropna().tolist()
old_names_list


# In[120]:

smallnamesdf = numbers.dropna().reset_index()
smallnamesdf


# In[121]:

smallnamesdf.replace(to_replace = 'China, Macao Special Administrative Region', 
               value = 'Macao', inplace = True) #con madre wey
smallnamesdf


# In[122]:

old_names_list = smallnamesdf['Country'].tolist() #get bad names
new_names_list = smallnamesdf['Index'].tolist() # set good names
print(old_names_list)
print(new_names_list)


# In[123]:

energy.replace(to_replace = old_names_list, 
               value = new_names_list, inplace = True) #con madre wey
print(energy)

