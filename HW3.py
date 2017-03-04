
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

#import pandas as pd
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
energy


# In[124]:

#Next, load the GDP data from the file world_bank.csv, which is a csv containing countries' GDP from 1960 to 2015 
#from World Bank. Call this DataFrame GDP.
#Make sure to skip the header, and rename the following list of countries:
#"Korea, Rep.": "South Korea", 
#"Iran, Islamic Rep.": "Iran",
#"Hong Kong SAR, China": "Hong Kong"


# In[125]:

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
gdp


# In[130]:

GDP = gdp


# In[131]:

###
#Finally, load the Sciamgo Journal and Country Rank data for Energy Engineering and Power Technology from the 
#file scimagojr-3.xlsx, which ranks countries based on their journal contributions in the aforementioned area. 
#Call this DataFrame ScimEn.
#Join the three datasets: GDP, Energy, and ScimEn into a new dataset (using the intersection of country names). 
 #   Use only the last 10 years (2006-2015) of GDP data and only the top 15 countries by Scimagojr 'Rank' 
#  (Rank 1 through 15).
#The index of this DataFrame should be the name of the country, and the columns should be 
#['Rank', 'Documents', 'Citable documents', 'Citations', 'Self-citations', 'Citations per document', 
# 'H index', 'Energy Supply', 'Energy Supply per Capita', '% Renewable', '2006', '2007', '2008', '2009', '2010', 
# '2011', '2012', '2013', '2014', '2015'].
###


# In[132]:

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


#def answer_one():
    
 #   return "ANSWER"


# In[155]:

df = energy_gdp_sci_mx


# ### Question 2 (6.6%)
# The previous question joined three datasets then reduced this to just the top 15 entries. When you joined the datasets, but before you reduced this to the top 15 items, how many entries did you lose?
# 
# *This function should return a single number.*

# In[156]:

#get_ipython().run_cell_magic('HTML', '', '<svg width="800" height="300">\n  <circle cx="150" cy="180" r="80" fill-opacity="0.2" stroke="black" stroke-width="2" fill="blue" />\n  <circle cx="200" cy="100" r="80" fill-opacity="0.2" stroke="black" stroke-width="2" fill="red" />\n  <circle cx="100" cy="100" r="80" fill-opacity="0.2" stroke="black" stroke-width="2" fill="green" />\n  <line x1="150" y1="125" x2="300" y2="150" stroke="black" stroke-width="2" fill="black" stroke-dasharray="5,3"/>\n  <text  x="300" y="165" font-family="Verdana" font-size="20">Everything but this!</text>\n</svg>')


# In[157]:

#def answer_two():
#    return "ANSWER"


# <br>
# 
# Answer the following questions in the context of only the top 15 countries by Scimagojr Rank (aka the DataFrame returned by `answer_one()`)

# ### Question 3 (6.6%)
# What is the average GDP over the last 10 years for each country? (exclude missing values from this calculation.)
# 
# *This function should return a Series named `avgGDP` with 15 countries and their average GDP sorted in descending order.*

# In[158]:

df['avgGDP'] = df[['2006 GDP (in 2010 USD)',
 '2007 GDP (in 2010 USD)',
 '2008 GDP (in 2010 USD)',
 '2009 GDP (in 2010 USD)',
 '2010 GDP (in 2010 USD)',
 '2011 GDP (in 2010 USD)',
 '2012 GDP (in 2010 USD)',
 '2013 GDP (in 2010 USD)',
 '2014 GDP (in 2010 USD)',
 '2015 GDP (in 2010 USD)']].mean(axis=1)
df


# In[159]:

list(df)


# In[160]:

#energy_gdp_sci.sort(columns='Rank', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
df.sort(columns='avgGDP', axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last')


# In[ ]:




# In[ ]:




# In[ ]:




# In[161]:

#def answer_three():
#    Top15 = answer_one()
#    return "ANSWER"


# ### Question 4 (6.6%)
# By how much had the GDP changed over the 10 year span for the country with the 6th largest average GDP?
# 
# *This function should return a single number.*

# In[162]:

df[5:6]


# In[163]:

#df['Minimum'] = df.loc[:, ['B0', 'B1', 'B2']].min(axis=1)
df['GDPrange'] = (df.loc[:, ['2006 GDP (in 2010 USD)',
 '2007 GDP (in 2010 USD)',
 '2008 GDP (in 2010 USD)',
 '2009 GDP (in 2010 USD)',
 '2010 GDP (in 2010 USD)',
 '2011 GDP (in 2010 USD)',
 '2012 GDP (in 2010 USD)',
 '2013 GDP (in 2010 USD)',
 '2014 GDP (in 2010 USD)',
 '2015 GDP (in 2010 USD)']].max(axis=1) - 
                df.loc[:, ['2006 GDP (in 2010 USD)',
 '2007 GDP (in 2010 USD)',
 '2008 GDP (in 2010 USD)',
 '2009 GDP (in 2010 USD)',
 '2010 GDP (in 2010 USD)',
 '2011 GDP (in 2010 USD)',
 '2012 GDP (in 2010 USD)',
 '2013 GDP (in 2010 USD)',
 '2014 GDP (in 2010 USD)',
 '2015 GDP (in 2010 USD)']].min(axis=1))
df


# In[164]:

df[5:6]['GDPrange']


# In[165]:

#def answer_four():
#    Top15 = answer_one()
#    return "ANSWER"


# ### Question 5 (6.6%)
# What is the mean `Energy Supply per Capita`?
# 
# *This function should return a single number.*

# In[166]:

#print('The enemey gets hit for {} hitpoints'.format(damage))
print('The mean Energy Supply per Capita is {} gigajoules.'.format(df['Energy Supply per Capita 2013'].mean()))


# In[167]:

#def answer_five():
 #   Top15 = answer_one()
  #  return "ANSWER"


# ### Question 6 (6.6%)
# What country has the maximum % Renewable and what is the percentage?
# 
# *This function should return a tuple with the name of the country and the percentage.*

# In[168]:

list(df)


# In[169]:

#You just need the argmax() (now called idxmax) function. It's straightforward:
df['% Renewable 2013'].idxmax() # me regresa indice
# quiero pasar indice a otro comanda
most_renewable_country = [df.loc[df['% Renewable 2013'].idxmax()]['Country'],df['% Renewable 2013'].max(axis=0)]
print(most_renewable_country)
#def answer():
 # tuple = [df.loc[df['% Renewable 2013'].idxmax()]['Country'],df['% Renewable 2013'].max(axis=0)]
  #return tuple


# In[170]:

#def answer_six():
 #   Top15 = answer_one()
#  return "ANSWER"


# In[171]:
country = df.loc[df['% Renewable 2013'].idxmax()]



# In[172]:

print(("The country with the highest proportion of renewable energy is {} with {} percent renewable energy.".format(df.loc[df['% Renewable 2013'].idxmax()]['Country'],df['% Renewable 2013'].max(axis=0))))

# ### Question 7 (6.6%)
# Create a new column that is the ratio of Self-Citations to Total Citations. 
# What is the maximum value for this new column, and what country has the highest ratio?
# 
# *This function should return a tuple with the name of the country and the ratio.*

# In[173]:

df['Self-Citation Ratio'] = df['Self-citations 1996-2015']/df['Citations 1996-2015']
df


# In[174]:

# fast column renaming procedure
#df.columns = ['log(gdp)' if x=='gdp' else x for x in df.columns]
df.columns = ['Self-citation Ratio' if x=='Self-Citation Ratio' else x for x in df.columns]
df


# In[175]:

df.sort(columns='Self-citation Ratio', axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last')
df


# In[176]:

def answer2():
    country2 = df.loc[df['Self-citation Ratio'].idxmax()]['Country']
    data2 = df['Self-citation Ratio'].max(axis=0)
    print('The country with the highest self-citation ratio is {} and its ratio is {}.'
                 .format(country2, data2))
    return
answer2()


# In[177]:

#def answer_seven():
 #   Top15 = answer_one()
  #  return "ANSWER"


# ### Question 8 (6.6%)
# 
# Create a column that estimates the population using Energy Supply and Energy Supply per capita. 
# What is the third most populous country according to this estimate?
# 
# *This function should return a single string value.*

# In[178]:

df['Population'] = df['Energy Supply 2013']/df['Energy Supply per Capita 2013']
df.sort(columns = 'Population', axis = 0, ascending = False, inplace = True, kind = 'quicksort')
df[2:3]['Country']


# In[179]:

df[2:3]


# In[180]:

#def answer_eight():
 #   Top15 = answer_one()
  #  return "ANSWER"


# ### Question 9 (6.6%)
# Create a column that estimates the number of citable documents per person. 
# What is the correlation between the number of citable documents per capita and the energy supply per capita? Use the `.corr()` method, (Pearson's correlation).
# 
# *This function should return a single number.*
# 
# *(Optional: Use the built-in function `plot9()` to visualize the relationship between Energy Supply per Capita vs. Citable docs per Capita)*

# In[181]:

df['Docs per person'] = df['Citable documents 1996-2015']/df['Population']
df


# In[182]:

print('The correlation between energy supply per person and energy publications per person is {}.'.format(df.corr().loc['Energy Supply per Capita 2013']['Docs per person']))
df.corr()


# In[184]:



df.columns = ['Population estimate 2013' if x=='Population' 
  else 'Energy documents per person estimate (Docs 1996-2015, Population 2013)' 
  if x=='Docs per person'
  else x for x in df.columns]
df


# In[185]:

#def answer_nine():
 #   Top15 = answer_one()
  #  return "ANSWER"


# In[186]:

#def plot9():
#    import matplotlib as plt
#    %matplotlib inline
#    
#    Top15 = answer_one()
#    Top15['PopEst'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
#    Top15['Citable docs per Capita'] = Top15['Citable documents'] / Top15['PopEst']
#    Top15.plot(x='Citable docs per Capita', y='Energy Supply per Capita', kind='scatter', xlim=[0, 0.0006])


# In[187]:

print ("code executed until this line! 839")
print ("df value is: ")
print (df)

import matplotlib as plt


# # In[188]:

#for i in range(0,16):
 #   print('plt.pyplot.text(docs[{}],energy[{}],labels[{}])'.format(i,i,i))


# # In[ ]:




# # In[ ]:

# #get_ipython().magic('matplotlib inline')

df.plot(x='Energy documents per person estimate (Docs 1996-2015, Population 2013)',
  y='Energy Supply per Capita 2013',
  kind='scatter',
  xlim=[0, 0.0006])
plt.pyplot.suptitle('Energy publications and Energy supply per Capita')
plt.pyplot.xlabel('Energy Publications per Capita \n(Total documents 1996-2015/Population 2013)')
plt.pyplot.ylabel('Energy Supply per Capita \n(Gigajoules 2013)')
docs = df['Energy documents per person estimate (Docs 1996-2015, Population 2013)'].tolist()
energy = df['Energy Supply per Capita 2013'].tolist()
labels = df['Country'].tolist()
plt.pyplot.text(docs[0],energy[0],labels[0])
plt.pyplot.text(docs[1],energy[1],labels[1])
plt.pyplot.text(docs[2],energy[2],labels[2])
plt.pyplot.text(docs[3],energy[3]-10,labels[3])
plt.pyplot.text(docs[4],energy[4],labels[4])
plt.pyplot.text(docs[5],energy[5],labels[5])
plt.pyplot.text(docs[6],energy[6],labels[6])
plt.pyplot.text(docs[7],energy[7],labels[7])
plt.pyplot.text(docs[8],energy[8],labels[8])
plt.pyplot.text(docs[9],energy[9],labels[9])
plt.pyplot.text(docs[10]-20,energy[10],labels[10])
plt.pyplot.text(docs[11],energy[11],labels[11])
plt.pyplot.text(docs[12],energy[12]+10,labels[12])
plt.pyplot.text(docs[13],energy[13]-10,labels[13])
plt.pyplot.text(docs[14],energy[14],labels[14])
plt.pyplot.text(docs[15],energy[15],labels[15])
plt.pyplot.show()

#for i in range(0,16):
#    plt.pyplot.text(docs[i],energy[i],labels[i])
#plt.pyplot.text(docs[1],energy[1],labels[1])



#y=[2.56422, 3.77284,3.52623,3.51468,3.02199]
#z=[0.15, 0.3, 0.45, 0.6, 0.75]
#n=[58,651,393,203,123]

#fig, ax = plt.subplots()
#ax.scatter(z, y)

#for i, txt in enumerate(n):
#    ax.annotate(txt, (z[i],y[i]))

#docs = df['Energy documents per person estimate (Docs 1996-2015, Population 2013)']
#energy = df['Energy Supply per Capita 2013']
#labels = df['Country']
#for i in len(labels):
#    plt.pyplot.text(docs[i],energy[i],labels[i])


# In[ ]:

# print(type(labels))
# labels


# # In[ ]:

# import matplotlib.pyplot as plt2
# docs = df['Energy documents per person estimate (Docs 1996-2015, Population 2013)']
# energy = df['Energy Supply per Capita 2013']
# labels = df['Country']

# fig, ax = plt2.subplots()
# ax.scatter(docs, energy)

# for i, text in enumerate(labels):
#     ax.annotate(text, (docs[i],energy[i]))


# In[ ]:

#plot9() # Be sure to comment out plot9() before submitting the assignment!


# ### Question 10 (6.6%)
# Create a new column with a 1 if the country's % Renewable value is at or above the median for all countries in the top 15, and a 0 if the country's % Renewable value is below the median.
# 
# *This function should return a series named `HighRenew` whose index is the country name sorted in ascending order of rank.*

# In[ ]:

#def answer_ten():
 #   Top15 = answer_one()
  #  return "ANSWER"


# ### Question 11 (6.6%)
# Use the following dictionary to group the Countries by Continent, then create a dateframe that displays the sample size (the number of countries in each continent bin), and the sum, mean, and std deviation for the estimated population of each country.
# 
# ```python
# ContinentDict  = {'China':'Asia', 
#                   'United States':'North America', 
#                   'Japan':'Asia', 
#                   'United Kingdom':'Europe', 
#                   'Russian Federation':'Europe', 
#                   'Canada':'North America', 
#                   'Germany':'Europe', 
#                   'India':'Asia',
#                   'France':'Europe', 
#                   'South Korea':'Asia', 
#                   'Italy':'Europe', 
#                   'Spain':'Europe', 
#                   'Iran':'Asia',
#                   'Australia':'Australia', 
#                   'Brazil':'South America'}
# ```
# 
# *This function should return a DataFrame with index named Continent `['Asia', 'Australia', 'Europe', 'North America', 'South America']` and columns `['size', 'sum', 'mean', 'std']`*

# In[ ]:

#def answer_eleven():
 #   Top15 = answer_one()
  #  return "ANSWER"


# ### Question 12 (6.6%)
# Cut % Renewable into 5 bins. Group Top15 by the Continent, as well as these new % Renewable bins. How many countries are in each of these groups?
# 
# *This function should return a __Series__ with a MultiIndex of `Continent`, then the bins for `% Renewable`. Do not include groups with no countries.*

# In[ ]:

#def answer_twelve():
 #   Top15 = answer_one()
  #  return "ANSWER"


# ### Question 13 (6.6%)
# Convert the Population Estimate series to a string with thousands separator (using commas). Do not round the results.
# 
# e.g. 317615384.61538464 -> 317,615,384.61538464
# 
# *This function should return a Series `PopEst` whose index is the country name and whose values are the population estimate string.*

# In[ ]:

#def answer_thirteen():
 #   Top15 = answer_one()
  #  return "ANSWER"


# ### Optional
# 
# Use the built in function `plot_optional()` to see an example visualization.

# In[ ]:

#def plot_optional():
 #   import matplotlib as plt
    #get_ipython().magic('matplotlib inline')
  #  Top15 = answer_one()
   # ax = Top15.plot(x='Rank', y='% Renewable', kind='scatter', 
    #                c=['#e41a1c','#377eb8','#e41a1c','#4daf4a','#4daf4a','#377eb8','#4daf4a','#e41a1c',
      #                 '#4daf4a','#e41a1c','#4daf4a','#4daf4a','#e41a1c','#dede00','#ff7f00'], 
     #              xticks=range(1,16), s=6*Top15['2014']/10**10, alpha=.75, figsize=[16,6]);

    #for i, txt in enumerate(Top15.index):
     #   ax.annotate(txt, [Top15['Rank'][i], Top15['% Renewable'][i]], ha='center')

#print("This is an example of a visualization that can be created to help understand the data. This is a bubble chart showing % Renewable vs. Rank. The size of the bubble corresponds to the countries' 2014 GDP, and the color corresponds to the continent.")
print("Thank you, Homero.")

# In[ ]:

#plot_optional() # Be sure to comment out plot_optional() before submitting the assignment!


# In[ ]:






