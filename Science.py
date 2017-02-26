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


#Antecedente
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

# Empieza aqui
ScimEn = pd.read_excel('course1_downloads/scimagojr-3.xlsx')
ScimEn


# In[133]:

GDP_columns = list(range(1960, 2016))
GDP_columns


# In[134]:

#df = pd.DataFrame({'A':[1,2,3],
#                   'B':[4,5,6],
 #                  'C':[7,8,9],
#                  'D':[1,3,5],
#                   'E':[5,3,6],
#                   'F':[7,4,3]})
#df
#cols = list(range(1,7))
#cols
#df.columns = cols
#df


# In[135]:

#df.columns = df.columns[:2].tolist() + namesList
GDP.columns = GDP.columns[:3].tolist() + GDP_columns #rename columns by years
GDP


# In[136]:

#DataFrame.merge(right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, 
#                sort=False, suffixes=('_x', '_y'), copy=True, indicator=False)


# In[137]:

energy_backup = energy


# In[138]:

energy_gdp = energy.merge(GDP, how='inner', left_on = 'Country', right_index = True)
energy_gdp


# In[139]:

energy_gdp_sci = energy_gdp.merge(ScimEn, left_on = 'Country', right_on = 'Country')
energy_gdp_sci


# In[140]:

print(len(energy), len(GDP), len(ScimEn), len(energy_gdp), len(energy_gdp_sci))


# In[141]:

years_to_delete = list(range(1960,2006))
years_to_delete


# In[142]:

goal_cols = ['Country','Rank', 'Documents', 'Citable documents', 'Citations', 'Self-citations', 'Citations per document', 'H index', 'Energy Supply', 'Energy Supply per Capita', '% Renewable', 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015]
energy_gdp_sci = energy_gdp_sci[goal_cols]
energy_gdp_sci


# In[143]:

energy_gdp_sci_columns = energy_gdp_sci #save work


# In[144]:

energy_gdp_sci = energy_gdp_sci_columns #recover position
print(list(energy_gdp_sci))
print(len(energy))


# In[145]:

energy_gdp_sci = energy_gdp_sci[energy_gdp_sci['Rank']<16]


# In[146]:

energy_gdp_sci


# In[147]:

energy_gdp_sci.sort(columns='Rank', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')


# In[148]:

#agarrar a Mexico
Mexicoframe = energy_gdp_sci_columns[energy_gdp_sci_columns['Rank']==24]
Mexicoframe


# In[149]:

energy_gdp_sci_mx = pd.concat([Mexicoframe, energy_gdp_sci])
energy_gdp_sci_mx.sort(columns='Rank', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')


# In[150]:

list(energy_gdp_sci_mx)


# In[151]:

colnames = (['Country',
 'Rank by Docs',
 'Documents 1996-2015',
 'Citable documents 1996-2015',
 'Citations 1996-2015',
 'Self-citations 1996-2015',
 'Citations per document 1996-2015',
 'H index 1996-2015',
 'Energy Supply 2013',
 'Energy Supply per Capita 2013',
 '% Renewable 2013',
 '2006 GDP (in 2010 USD)',
 '2007 GDP (in 2010 USD)',
 '2008 GDP (in 2010 USD)',
 '2009 GDP (in 2010 USD)',
 '2010 GDP (in 2010 USD)',
 '2011 GDP (in 2010 USD)',
 '2012 GDP (in 2010 USD)',
 '2013 GDP (in 2010 USD)',
 '2014 GDP (in 2010 USD)',
 '2015 GDP (in 2010 USD)'])


# In[152]:

energy_gdp_sci_mx.columns = colnames


# In[153]:

energy_gdp_sci_mx


# In[154]:


def answer_one():
    
    return "ANSWER"


# In[155]:

df = energy_gdp_sci_mx
print(df)