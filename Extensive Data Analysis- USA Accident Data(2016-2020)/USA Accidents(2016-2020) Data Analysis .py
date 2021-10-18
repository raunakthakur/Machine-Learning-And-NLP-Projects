#!/usr/bin/env python
# coding: utf-8

# # USA Accident Data Analysis
# 
# <br>Description<br>
# This is a countrywide car accident dataset, which covers 49 states of the USA. The accident data are collected from February 2016 to Dec 2020, using multiple APIs that provide streaming traffic incident (or event) data. These APIs broadcast traffic data captured by a variety of entities, such as the US and state departments of transportation, law enforcement agencies, traffic cameras, and traffic sensors within the road-networks. Currently, there are about 1.5 million accident records in this dataset. Check here to learn more about this dataset.<br> 
# 
# 
# [USA Accident Dataset Link](https://www.kaggle.com/sobhanmoosavi/us-accidents)

# ## 1. The problem statement 

# US-Accidents can be used for numerous applications such as real-time car accident prediction, studying car accidents hotspot locations, casualty analysis and extracting cause and effect rules to predict car accidents, and studying the impact of precipitation or other environmental stimuli on accident occurrence. The most recent release of the dataset can also be useful to study the impact of COVID-19 on traffic behavior and accidents.

# ###  Data Preparation 

# ### 1. Import libraries 
# 
# The first step in building the model is to import the necessary libraries.  

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# ### 2. Import dataset 
# 
# The next step is to import the dataset.

# In[3]:


data='US_Accidents_Dec20_Updated.csv'
df = pd.read_csv(data)


# ### 3. Exploratory data analysis 
# 
# - We have imported the data. 
# 
# - Now, its time to explore the data to gain insights about it. 

# **View dimensions of dataset** 

# In[4]:


df.shape


# The USA Accident data is huge having around ```29 million``` rows and ```47``` columns that are our Features ,which we will analyse in a while .   

# In[5]:


df.head()


# **Preview the dataset** 

# In[6]:


df.head(5)


# **Checking Feature Names**

# In[7]:


df.columns


# We have 47 Features , a look on features is a good kick start  to  brain storming  , on  how features may be interacting.
# Since there are large number of feature sometimes they are not visible in Tabular dataframe its good pracrice to check columns before hand.

# **View statistical properties of dataset** 

# In[8]:


df.describe()


# #### Important points to note
# - The above command `df.describe()` helps us to view the statistical properties of numerical variables. It excludes character variables.
# - If we want to view the statistical properties of character variables, we should run the following command -
#       df.describe(include=['object'])
# - If we want to view the statistical properties of all the variables, we should run the following command -
#       df.describe(include='all')

# **View summary of dataset**

# In[9]:


df.info()


# here we can see  what are the data types of our feature ,in this dataset there are :<br>
# Numeric Columns (14)=```float64(13)```+ ```int64(1)```,<br>
# categorical Columns(20) =```object(20)```,<br>
# Bolean Columns(13)=```boolean(13)```

# #### Important points to note
# 
# 
# - The above command `df.describe()` helps us to view the statistical properties of numerical variables. It excludes character variables.
# 
# - If we want to view the statistical properties of character variables, we should run the following command -
# 
#        df.describe(include=['object'])
# 
# - If we want to view the statistical properties of all the variables, we should run the following command -
# 
#       df.describe(include='all')

# **Suggestion** 
# *[Although i am not using it here]*
# 
# There is  klib a Python library for importing, cleaning, analyzing and preprocessing data.
# I have found it very handy for data cleaning ,preprocessing  and  analyzing.
# Check it here [Klib Library](https://klib.readthedocs.io/en/latest/).

# ## Lets Dive Deeper into the Data Analysis & Visualization 

# **Missing Values by percentage**

# In[10]:


missing_values=df.isna().sum().sort_values(ascending=False)
missing_perc=df.isna().sum().sort_values(ascending=False)/len(df)*100
missing_perc=missing_perc[missing_perc!=0]
missing_values=missing_values[missing_values!=0]


# In[11]:


df_missing_data=pd.DataFrame(list(zip(missing_values,missing_perc)),columns=['No of missing values','% of missing values'])
df_missing_data.style.background_gradient('Greens')


# **Visualizing  missing Percentage of Values in Features**

# In[12]:


plt.figure(figsize=(11.7,8.27))
missing_perc[missing_perc!=0].plot(kind='barh',color="green")


# **Here we can see ```The Missing Values``` in our Dataset:** 
# 
# - Number(Street number) - we can say for that 65% of the cases of accident the Street number were not recoded .This may be Due     to various reason maybe  administrative reason or due the the data collection for various region may have begun on later       dates.
#   we can Map the reason  for Low number of records by finding the longitude and latitude of state and get the density  if more   number missing is region specific or time specific.
# 
# - Precipitation(in) and other weather data -```Precipitation(in)-44.7%``` ,```Wind_Chill(F)-40.77%``` ,```Wind_Speed(mph)-10.56``` Data Collection may  started recoding on later dates. 
# 
# - Location of Accidents-```End_Lat& End_Lng-9.73 %``` Medical emergency can be one of the cause for unavailability of data .  
#                    

# #### Dropping  columns as many missing values are present

# In[13]:


df.drop(columns=['Number','Precipitation(in)','Wind_Chill(F)'],inplace=True)


# ### We will now Analyse some of the Feature  in our  dataset :
# - city 
# - start time 
# - start lat,long
# 
# 
# ```Note```:This is not exhaustive analysis ,there is more we can add like  ..
# - temperature 
# - weather condition

# ### Information About Accident Based on Cities: 

# **[A] The number of Cities**

# In[14]:


cities=df.City.unique()
len(cities)


# In our dataset ,There are  ```11790``` cities where accidents were recorded.

# **[B] Top 10 cities with highest accidents**

# In[15]:


cities_by_acc=df.City.value_counts()
cities_by_acc[:10]


# In[53]:


plt.figure(figsize=(10,5))
cities_by_acc[:10].plot.bar(x='cities',y='Count',title='Top 10 Cities By Accident',legend=True,color="green") 


# In[51]:


"New York " in df.State  


# In[54]:


'Phoenix' in df.State


# **Top 10 cities with highest accidents:**<br> 
# 
# ```Observation:```   **This dataset does not contain data for some cities like  New York,Pheonix which are some of the  largest cities by population of USA **.
# 
# ![USA%20Population%20Data%28Wiki%29.png](attachment:USA%20Population%20Data%28Wiki%29.png)
# 
# - Among the Top 10 Cities By Accidents :<br>
# ```Los Angeles```,```Houston```,```Charlotte```, ```Dallas```,```Austin``` comes  Top 20 cities by US population.<br>         
# ```Miami```,```Orlando```,```Atlanta```,```Sacramento``` comes  Top 50 cities by US population .

# #### [C] Cities in Terms of Percentage Accounting High Accidents(> 1000)

# In[56]:


high_cities_by_acc=cities_by_acc[cities_by_acc>=1000]
high_per_total=len(high_cities_by_acc)/len(cities)*100
round(high_per_total, 3)


# **Plot for Cities Accounting High Accidents(> 1000)**

# In[80]:


pl=sns.displot(high_cities_by_acc,color='g', height=8.27, aspect=11.7/8.27)
pl.set(xlabel='Number Of Accidents in Cities ', ylabel='Count of Cities',title='Cities Accounting High Accidents(> 1000)')
plt.show()


# - Only ```4.35 % ``` of cities account for more than 1000 during 2016-2020 ,which is possibly due to high population density and Traffic in the concerned cities.The city street layout ,loops,roundabout,Diversion ,UrbanLandscape can be analysed s well to get the reason for ,why bunch of cities contribute to more accidents.<br>
# 
# - This is evident by the plot being sparsely scattered in a wide range showing  ```Accidents Greater than 1000``` over a span of 2016-2020 .

# #### [D] Cities in Terms of Percentage Accounting Low  Accidents(< 1000)

# In[20]:


low_cities_by_acc=cities_by_acc[cities_by_acc<1000]
low_per_total=len(low_cities_by_acc)/len(cities)
round(low_per_total, 3)


# **Plot for Cities Accounting Low Accidents(< 1000)**

# In[81]:


plt.figure(figsize=(16, 6))
pl=sns.histplot(low_cities_by_acc,log_scale=True,color='g')
pl.set(xlabel='Number Of Accidents in Cities ', ylabel='Count of Cities',title='Cities Accounting Low Accidents(< 1000)')
plt.show()


# **Cities Accounting Low Accidents(< 1000):**
# 
# - We can see from our calculation that ```95.6%``` of the cities accounted for less than 1000 case of Accidents.
# <br>
# - the plot displays less than about ```Accidents less than 1000``` over a span of 2016-2020 ,it will be good to see Number Cities having ```Fewer Accidents (<10 )```.

# #### [E] Number Of  Cities having Fewer Accidents (<10 )

# In[73]:


less_10=cities_by_acc[cities_by_acc<=10].value_counts()
less_10


# **Plot for Number of  Cities having Fewer Accidents (<10)**

# In[69]:


plt.figure(figsize=(16, 6))
pl=sns.histplot(less_10,color='g')
pl.set(xlabel='Number Of  Cities', ylabel='Number Of Accidents',title='Number of Cities having Fewer Accidents (<10)')
plt.show()


# ## Datetime Columns  Analysis  -
# 
# ###  Start_time column 
# 

# In[74]:


df.Start_Time=pd.to_datetime(df.Start_Time)


# ### [A] What Time of the Day the accident are high ??

# In[84]:


pl=sns.displot(df.Start_Time.dt.hour,bins=24,kde=False,color="g",height=8.27, aspect=11.7/8.27)
pl.set(xlabel='Hours(0-24)', ylabel='Number of Accidents',title='Accidents by Time of The Day')
plt.show()


# - Most of the accidents take place at between `7-9 AM``` and 16-18PM` ,The traffic might increase due to Commutation for Jobs,School,Start of market activity in general.
# - Overall from the graph we can Say that from `5 AM to 20 PM` the instance of Accident are quite High.
# - traffic data is not available .Number of loop in road has to be checked .

# ### [B] What Day of Week the Accident are High ?

# In[83]:


pl=sns.displot(df.Start_Time.dt.dayofweek,bins=7,kde=False,color='g',height=8.27, aspect=11.7/8.27)
pl.set(xlabel='Week Days-(Monday-0 To Sunday-6) ', ylabel='Number of Accidents',title='Accidents by Week Of The Day')
plt.show()


# - It can be seen that` weekends (Saturday,Sunday)` has significantly less events of Accidents `Saturday ~20,000 and Sunday ~18000` than weekdays having  `around 50000`  on an average. 

# ### [C]   Is The Distribution By Hour , Same On Weekends As on Weekdays

# **Weekend(Sunday)**

# In[28]:


sunday=df.Start_Time[df.Start_Time.dt.dayofweek==6]


# In[79]:


pl=sns.displot(sunday.dt.hour,bins=24,kde=False,color='g',height=8.27, aspect=11.7/8.27)
pl.set(xlabel='Hours(24)',ylabel='Number of Accidents',title='Accidents By Hours of Sunday')
plt.show()


# **The Distribution By Hour , Same On Weekends As on Weekdays -**<br>
# 
# **On Weekend(Sunday)**
# - Its afternoon in weekend when  most of the accidents occur `Between 10AM and 14PM` .
# - But still there are more case after evenings than morning  as well might  be due to travel to workplace  .

# **Weekday(Monday)**

# In[85]:


monday=df.Start_Time[df.Start_Time.dt.dayofweek==0]
monday


# In[86]:


pl=sns.displot(monday.dt.hour,bins=24,kde=False,color='g',height=8.27, aspect=11.7/8.27)
pl.set(xlabel='Hours(24)',ylabel='Number of Accidents',title='Accidents By Hours of Monday')
plt.show()


# **The Distribution By Hour , Same On Weekends As on Weekdays -**<br>
# **Weekday(Monday)**
# 
# - its on weekdays the maximum accidents occur at office,School commutation hours morning `Between 6AM and 8AM` and evening `Between 15PM and 17PM`.
# 

# ### [D] What Month of Year The Accident are High ??

# In[89]:


pl=sns.displot(df.Start_Time.dt.month,bins=12,kde=False,color='g',height=8.27, aspect=11.7/8.27)
pl.set(xlabel='Months(12)',ylabel='Number of Accidents',title='Accidents By Months')
plt.show()


# **Month of Year The Accident are High-**<br>
# - In winter the number of accident increases possibly due to Christmas Holidays,weather condition like Snowfall(reduce Visibility),Sleet formation on road(makes vehicle tyre slip).
# 
# - 7th month i.e JULY is Summer Season across USA weather remains well ,possibly resulting in Lesser Accident Counts. 
# 
# - correaltion with temp ,season ,fog, snow ,has to be checked for better interpretation.

# ### [E] In Which  Year The Accident are High ??

# In[90]:


df.Start_Time.dt.year.nunique()


# #### There are 5 years starting from 2016 To 2020 :
# lets plot for number of Accident cases per year.

# **Year 2020**

# In[108]:


df_2020=df[df.Start_Time.dt.year==2020]
plt.figure(figsize=(16, 6))
pl=sns.distplot(df_2020.Start_Time.dt.month,bins=12,kde=False,color='g')
pl.set(xlabel='Months(12)',ylabel='Number of Accidents',title='Year 2020')
plt.show()


# **Year 2019**

# In[102]:


df_2019=df[df.Start_Time.dt.year==2019]
plt.figure(figsize=(16, 6))
pl=sns.distplot(df_2019.Start_Time.dt.month,bins=12,kde=False,color='g')
pl.set(xlabel='Months(12)',ylabel='Number of Accidents',title='Year 2019')
plt.show()


# **Year 2018**

# In[103]:


df_2018=df[df.Start_Time.dt.year==2018]
plt.figure(figsize=(16, 6))
pl=sns.distplot(df_2018.Start_Time.dt.month,bins=12,kde=False,color='g')
pl.set(xlabel='Months(12)',ylabel='Number of Accidents',title='Year 2018')
plt.show()


# **Year 2017**

# In[104]:


df_2017=df[df.Start_Time.dt.year==2017]
plt.figure(figsize=(16, 6))
pl=sns.distplot(df_2017.Start_Time.dt.month,bins=12,kde=False,color='g')
pl.set(xlabel='Months(12)',ylabel='Number of Accidents',title='Year 2017')
plt.show()


# **Year 2016**

# In[105]:


df_2016=df[df.Start_Time.dt.year==2016]
plt.figure(figsize=(16, 6))
pl=sns.distplot(df_2016.Start_Time.dt.month,bins=12,kde=False,color='g')
pl.set(xlabel='Months(12)',ylabel='Number of Accidents',title='Year 2016')
plt.show()


# #### Occurance of Accidents by Year :
# 
# - Data has been missing for year 2016  for July month. 
# 
# - Year 2019 & 2020 has Recorded at an Average  Higher cases of Accident starting from September 2019 ,Exception being `July & August 2020 `where cases are  Visibily less,this maybe due to data collection might have increased year on year.
# 
# - Since 2020 is impacted by `COVID-19 lockdown and restriction` and same were relaxed Around `September 2020`  there is spike after September in 2020,more study is required to estblish this claim.

# In[ ]:





# ## 3. latitude and longitude

# **Mapping the Geographical Location (latitude and longitude ) is a great way to analyse the  Spatial Data,it is even more useful when we are looking at the scenario of Accident.**
# 
# -To gain better understanding we will use `**FOLIUM Library**`.<br>
# **briefly about folium**-*folium makes it easy to visualize data that’s been manipulated in Python on an interactive leaflet map. It enables both the binding of data to a map.*
# 
# **`Note:Since the data is huge 29 Million Rows ,i am using Sample Data From Dataset due to Limited machine's Computing Compulsion.`** 

# In[118]:


plt.figure(figsize=(16,10))
sns.scatterplot(x=df.Start_Lng,y=df.Start_Lat,color='g',size=0.001)


# **sample DataSet around 2900 Datapoints**

# In[116]:


sample_df=df.sample(int((0.001)*len(df)))
len(sample_df)


# In[42]:


import folium


# **Adding  marker with folium to point Geographical Coordinates** 

# In[119]:


lat_lon_pairs=list(zip(sample_df.Start_Lat,sample_df.Start_Lng))


#  Converting Sample of lat_lon_pairs to dataframe For Mapping 

# In[45]:


sample_df_lat_lon=pd.DataFrame(lat_lon_pairs,columns=['Latitide','Longitude'])


# **Plotting on Folium**

# In[46]:


map=folium.Map(width=850,height=550,location=[37.0902,-95.7129],zoom_start=4,min_zoom=4,max_zoom=14)
for lat,lon in sample_df_lat_lon.itertuples(index=False) :
  marker=folium.Marker(location=[lat,lon])
  marker.add_to(map)   
folium.TileLayer('Stamen Terrain').add_to(map)
folium.TileLayer('Stamen Toner').add_to(map)
folium.TileLayer('Stamen Water Color').add_to(map)
folium.TileLayer('cartodbpositron').add_to(map)
folium.LayerControl().add_to(map)

map


# **`NOTE`**:On the Map Window On **`TOP RIGHT`** We can see **`TILE FILTER`** to see Map with different view like StreetMap,Terrain Map,Water Color,Do access it.`

# **`Since there Around 2900 datapoint I am just randomly exploring few of them :`**<br>
# Specifically searching for Road Diversion ,roundabouts,crossing,Square and like wise for South-East Coast of USA(Florida):
# - There Are More Accident Around  Loops,Crossing ,Diversion and Roundabout in the Countryside  Area  ,may be Due to High speeds,
# - in Cities -Square ,Merging lanes, Roundabout are the easily visible sites where accident occurs.
# 
# Correlation between Weather ,Terrain ,Traffic Data ,speed limit data need to be studied to get indept interpretation . 

# ## 4. Summary Of USA Accident DATA(2016-2020)  Analysis :
# 
# **`Missing Data :`**   
# - No Data for New York, Phoenix  . 
# - No Data for year July,2016 .
# 
# **`Facts :`**
# - The number of accidents per city decreases exponentially.
# - Less than `5%`of cities have `more than 1000`  accidents across years(2016-2020).
# - More than `95%`of cities have `Less than 1000`  accidents across years(2016-2020).
# - Over `1300` cities have reported just one accident(need more investigation) in given period between `2016-2020`.
# 
# **`Observations :`**
# - Its on weekdays the maximum accidents occur at office,School commutation hours morning `Between 6AM and 8AM` and evening `Between 15PM and 17PM`.
# - In USA in winter the number of accident increases possibly due to `Christmas Holidays,weather condition like Snowfall(reduce     Visibility),Sleet formation on road(makes vehicle tyre slip)`.
# - 7th month i.e JULY is Summer Season across USA weather remains well ,possibly resulting in Lesser Accident Counts due to   Better Weather Conditions . 
# - In general There Are More Accident Around Loops,Crossing ,Diversion and Roundabout in the Countryside Area ,may be Due to High speeds,
# - In Cities -Square ,Merging lanes, Roundabout are the easily visible sites where accident occurs.
# - After Covid lockdown relaxtion After september 2020 sudden peak in accidents were recorded,more Traffic data needs to evualuated.

# ## More To Do :
# 
# - Getting The Data for Big Cities like New York will help use make better analysis as we have seen strong correlation between population,population density and accidents.
# - Collecting data of precipitation (65% data is missing) Checking the correlation on Weather condition like precipitation ,wind speed,wind chills etc will give as better insight.
# - Severity of accidents by region.
# - Understanding Accidents pattern During Covid-19 time in 2020. 

# ## Anaylsis Can Be Put To Use To Prevent Accidents:
# 
# - For Traffic management like managing City Traffic Congestion at peak hours, displaying Traffic Lights,making alternate routes.
# - For providing better Healthcare serive at  sites where Frequent accidents occurs ,specially creating first Aid centres by Exploring the map through markers and scatterplot.
# - Public Awareness-Radio alerts During winter season .
# - Civil Departments can use the street map to Do away with dangerous lanes diversion,Creating road Straight roads whereever feasible instead of loops  etc.
# - Police Department can look at analysis for futher study to see if accidents have strong correlation with Speeding ,drunk driving etc.

# ## Sources and Helpful Contents :
# [kaggle-](https://www.kaggle.com/sobhanmoosavi/us-accidents) <br> 
# [The 200 Largest Cities in the United States by Population -](https://worldpopulationreview.com/us-cities) <br> 
# [folium Documentation-](https://python-visualization.github.io/folium/)

# ## Acknowledgements:<br> 
# I had cited  the following papers For this dataset:<br> 
# ​
# Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, and Rajiv Ramnath. “A Countrywide Traffic Accident Dataset.”, 2019.<br> 
# ​
# Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, Radu Teodorescu, and Rajiv Ramnath. "Accident Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights." In proceedings of the 27th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems, ACM, 2019.<br> 

# In[ ]:




