#!/usr/bin/env python
# coding: utf-8

# #               Rain in Australia-Predict next-day rain in Australia
# 
# 
# 
# ## Table Of Content:
# 
# ### Introduction 
# 
# ### Problem Statement
# 
# ### Flow of Project
# 
# ### Data Exploration and Analysis
#     - Loading The dataset
#     - Exploring Dataset through Visualisation 
# 
# ### Preparing the Data For Training
#     - Create train/ test/ val split
#     - Identify input and target columns
#     - Identify numeric and catergorical columns
#     - Impute(fill) missing numeric values
#     - MinMaxScaler-Scale numeric values to the (0,1) range
#     - Encode categorical columns to one hot encoder
# ### Training a Random Forest Model
#     - Visualization-Forest Tree 
#     - Feature Importance
# ### Hyperparameter Tuning and Overfitting
#     - max_depth
#     - n_estimators
#     - max_ features
#     - class_weight
#     - bootstraps and max_samples
# ### Summarizing it together
#     - Making a Model with tuning
# ### Making predictions on a New Inputs
#     - predicting through test case

# ### Introduction 
# 
# 
# Weather phenomenon has always been Hard To predict.As till date there are research going on to understand how Atmospheric condition influence Weather pattern .The reasons are enormous and vary from Geographical location to locations.<br>
# Weather Reports/Channels/apps  till date on reports  with possibilty and not surety about the weather condition specially when Forecasting Rain .<br>
# 
# **Briefly about dataset:**<br>
# 
# This dataset contains about 10 years of daily weather observations from many locations across Australia.<br>
# RainTomorrow is the target variable to predict. It means -- did it rain the next day, Yes or No? This column is Yes if the rain for that day was 1mm or more.<br>
# 
# This data is taken from kaggle :To download and know more about data description [LINK HERE](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package)
# 

# ### Problem Statement
# 
# This is a Classification based problem .We are Dealing with imbalanced dataset ,here we are going to predict RainTomorrow based on  : <br>
# No:no rainfall<br>
# Yes:Rainfall<br>
# We will Attempt to predict the Rain Forecast by applying it on Austrailia Weather dataset using Random Forest ML algorithm.<br>
# in this project we see what is accuracy of the model we have created,and try out some hypermeter tuning .
# 

# ### Flow of Project
# The flow of project for ‘sentiment classification’ is as follows:<br>
# 
# - Loading The dataset<br>
# - Exploring Dataset<br>
# - Preparing the Australia Rainfall  Data For Training<br>
# - Training a Random Forest Model<br>
# - Hyperparameter Tuning and Overfitting<br>
# - Making a Model with tuning<br>
# - Make the prediction of Rain tomorrow on test case<br>
# - Finding model Accuracy<br>

# #### Downloading Useful Libraries

# In[1]:


# Libraries for Data Analysis and manuplation
import pandas as pd
import numpy as np
# libraries for Visualization 
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('darkgrid')
matplotlib.rcParams['font.size']= 14
matplotlib.rcParams['figure.figsize']=(10,6)
matplotlib.rcParams['figure.facecolor']='#00000000'
# Scikit library for machine learning
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix


# #### Load data

# In[2]:


df=pd.read_csv("C:\\Users\\91942\\Desktop\\KAGGLE DATA SET\\logistic regression data set\\rain in austraila\\weatherAUS.csv")


# **View dimensions of dataset** 

# In[3]:


df.shape


# The Rain in Australia data is having around ```1.45 lakh``` rows and ```23``` columns that are our Features ,which we will use in a while .   

# **Preview the dataset** 

# In[4]:


df.head()


# **Checking Feature Names**

# In[5]:


df.columns


# We have 23 Features , a look on features is a good kick start to brain storming , on how features may be interacting. Since there are large number of feature sometimes they are not visible in Tabular dataframe its good pracrice to check columns before hand.<br>
# Here  `'RainTomorrow'` the last columns is target column.

# **View summary of dataset**

# In[6]:


df.info()


# here we can see what are the data types of our feature ,in this dataset there are :<br>
# Numeric Columns (16)=float64(16),<br>
# categorical Columns(7) =object(7),<br>
# here Bolean Columns are not present.<br>

# **View statistical properties of dataset** 

# In[7]:


df.describe()


# We will Check if any and  how many missing values in our Target column i.e RainTomorrow.

# In[8]:


missing_target=sum(df.RainTomorrow.isna())
missing_target


# There,are 3627 values are missing for RainTomorrow ,Which can not be used for Prediction .
# <br> 
# We will Drop the isna() values from the dataset.

# In[9]:


df.dropna(subset=['RainToday','RainTomorrow'],inplace=True)
df.info()


# ## Lets Begin Data Exploration and analysis

# #### Location And RainToday

# In[10]:


df['Location'].nunique()


# There are 49 location for which we have weather data available

# In[11]:


px.histogram(df,x='Location',title='Location vs. Rainy Days',color='RainToday')


# Nhil, Kathrine and Uluru may have started collection data little late. 

# #### Rain Tomorrow vs. Rain Today

# In[12]:


px.histogram(df,x='RainTomorrow',title='Rain Tomorrow vs. Rain Today',color='RainToday')


# From the histogram we can see that there is imbalance in YES and NO RainTomorrow available,This is due to frequency of rain is generally low in Austrailia and have regional variations.<br>
# The North Australia  rainfall -Monsoon Type.<br>
# In Australian Coast recieves -Westerlies dependent in temperate region .<br>
# The Interior are Rain deprived due to distance from WaterBodies.<br>

# #### Min Temp vs Max Temp

# In[13]:


px.scatter(df.sample(2000),title='Min Temp vs Max Temp',x='MinTemp',y='MaxTemp',color='RainToday')


# Broadly We can see from Min Temp vs Max Temp ,there is more rainfall when max temp is less than 30  .

# #### Temperature at 3 Pm vs Rain Tomorrow

# In[14]:


px.histogram(df,x='Temp3pm',title='temperature at 3 Pm vs Rain Tomorrow',color='RainTomorrow')


# Temperature at 3 Pm vs Rain Tomorrow ,is following Guassian Distribution, we can see that 3 pm Temperature of around 20-30 degree has considerable dependence on rainfall. 

# #### Temperature(3pm) vs Humidity(3 pm)

# In[15]:


px.strip(df.sample(2000),x='Temp3pm',y='Humidity3pm',title="temperature(3pm) vs Humidity(3 pm)",color='RainTomorrow')


# Looking at the  scatterplot ,it can be interpreted temperature as the temperature decreases and humidity remains high ,there are higher incidences of RainTomorrow.<br>
# The reason is ,With lower Temperature Due point is achieved fast resulting in rainfall.Droplets starts to form when due point of 70% is achieved.

# ## Preparing the Data For Training
# 
# **1.create train/ test/ val split<br>
# 2.identify input and target columns<br>
# 3.identify numeric and catergorical columns<br>
# 4.impute(fill) missing numeric values<br>
# 5.scale numeric values to the (0,1) range<br>
# 6.Encode categorical columns to one hot encoder**

# ### 1.create train/ test/ val split

# In[16]:


plt.title('No of rows per year')
sns.countplot(x=pd.to_datetime(df.Date).dt.year)


# **When we are dealing with time series data its often a good idea to take last fraction as test sets to predict a better prediction out come.**

# In[17]:


year=pd.to_datetime(df.Date).dt.year
train_df=df[year<2015]
val_df=df[year==2015]
test_df=df[year>2015]


# In[18]:


print('train_df:',train_df.shape)
print('val_df:',val_df.shape)
print('test_df:',test_df.shape)


# ### Identifying Inputs and Target Columns
# Often, not all the columns it the dataset are useful for trianing a model,it increases computational cost and increases time and may impact accuracy adversely .<br> 
# We can ignore the date column,since we only want weather condition to make a prediction about whether it will rain the next day.

# In[19]:


input_cols=list(train_df.columns)[1:-1]
target_col='RainTomorrow'


# In[20]:


print(input_cols)
print(target_col)


# In[21]:


train_inputs=train_df[input_cols].copy()
train_target=train_df[target_col].copy()


# In[22]:


val_inputs=val_df[input_cols].copy()
val_target=val_df[target_col].copy()


# In[23]:


test_inputs=test_df[input_cols].copy()
test_target=test_df[target_col].copy()


# #### lets now also identify the numeric and categoircal columns

# In[24]:


numeric_cols=train_inputs.select_dtypes(include=np.number).columns.tolist()
categorical_cols=train_inputs.select_dtypes('object').columns.tolist()


# In[25]:


numeric_cols


# In[26]:


categorical_cols


# ### Imputing Missing Numeric Data,.
#  - Machine learning model can't work woth missing numerical data.<br>
#  - we are use imputation technique to minimize the data loss and filling the Nan value with mean.<br>
# There is a good article on limitation of  mean imputation.[Missing Data: Two Big Problems with Mean Imputation](https://www.theanalysisfactor.com/mean-imputation/)

# In[27]:


#from sklearn.impute import SimpleImputer


# In[28]:


imputer=SimpleImputer(strategy='mean')


# Before performing imputation ,lets check the no of missing values in each numeric column.

# In[29]:


train_inputs[numeric_cols].isna().sum().sort_values(ascending=False)


# In[30]:


imputer.fit(df[numeric_cols])


# In[31]:


list(imputer.statistics_)


# **The missing values in the training ,validation ,test sets can be now be filled with transform method of the imputer.**

# In[32]:


train_inputs[numeric_cols]=imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols]=imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols]=imputer.transform(test_inputs[numeric_cols])


# In[33]:


train_inputs[numeric_cols].isna().sum()


# ### Scaling Numeric Features 
# - Another good practice is to scale the numeric features to a small range of values eg (0,1),(-1,1). scaling numeric features ensures that no particular features has a disproprtionate impact on the model's loss. optimzation algorithms also works better in small numbers.<br>
# - The numeric columns in our dataset have varying ranges.

# lets use MinMaxScaler from sklearn.preprocessing to scale to te range of (0,1)range

# In[34]:


#from sklearn.preprocessing import MinMaxScaler


# In[35]:


scaler=MinMaxScaler().fit(df[numeric_cols])


# In[36]:


train_inputs[numeric_cols]=scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols]=scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols]=scaler.transform(test_inputs[numeric_cols])


# In[37]:


train_inputs[numeric_cols].describe().loc[['min','max']]


# ### Encoding Categorical Data 
# since machine learning model can only be trained with numeric data ,we need to convert categorical data to numbers, A common technique is to sue one hot encoding for categorical columns.

# In[38]:


#from sklearn.preprocessing import OneHotEncoder


# In[39]:


encoder=OneHotEncoder(sparse=False,handle_unknown='ignore')


# In[40]:


df[categorical_cols].isna().sum()


# In[41]:


df2=df[categorical_cols].fillna('Unknown')


# In[42]:


encoder.fit(df2[categorical_cols])


# In[43]:


df2[categorical_cols].dtypes


# In[44]:


encoder.categories_


# **we can generate column names for each individual categories using get_feature_names.**

# In[45]:


encoded_cols=list(encoder.get_feature_names(categorical_cols))
print(encoded_cols)


# **Filling the missing Categorical values in input with "Unknown"  value.**

# In[46]:


train_inputs[encoded_cols]=encoder.transform(train_inputs[categorical_cols].fillna('Unknown'))
val_inputs[encoded_cols]=encoder.transform(val_inputs[categorical_cols].fillna('Unknown'))
test_inputs[encoded_cols]=encoder.transform(test_inputs[categorical_cols].fillna('Unknown'))


# **Creating Traning,Validation,Test Set for training and testing RF model .**

# In[47]:


X_train=train_inputs[numeric_cols+encoded_cols]
X_val=val_inputs[numeric_cols+encoded_cols]
X_test=test_inputs[numeric_cols+encoded_cols]


# ### Training a Random Forest Model
# 
# While tuning the hypermeter using *single decision tree* may lead to some improvements,a much more effective strategy is to combine the results of several decision tree trained with slightly different parameters.This is called a random forest model. 
# 
# The key idea is that each decision tree in the forest will make different kind of errors,and upon averaging ,many of their error will cancel out.<br>This Genreal Technique of combining the reults of many models is called as"ensembling", 
# **This Is Why is called Bagging Ensembling Technique.**<br>
# ![download%20%281%29.jpg](attachment:download%20%281%29.jpg)
# 

# ### Training
# 

# In[48]:


#from sklearn.ensemble import RandomForestClassifier


# In[49]:


model=RandomForestClassifier(n_jobs=-1,random_state=42)


# In[50]:


model.fit(X_train,train_target)


# ### Evaluation
# Now we will  evaluate the Random Forest using the accuracy score and Probability

# In[51]:


from sklearn.metrics import accuracy_score,confusion_matrix


# In[52]:


train_preds=model.predict(X_train)
train_preds


# In[53]:


pd.value_counts(train_preds)


# **The probability for each tree prediction of Random Forest are:**

# In[54]:


trian_probs=model.predict_proba(X_train)
trian_probs


#  **The accuracy score of Random Forest**

# In[55]:


accuracy_score(train_preds,train_target)


# The model is looks quite confident ,the training set accuracy is 100% .but we cant rely on the training set for accuracy.we must evaluate the model on validation set too.We make prediction and compute accuracy of Validation set

# In[56]:


model.score(X_val,val_target)


# ##### **Although the training accuracy is 100% ,the accuracy of val is about 85%,there is difference of 15 % in predicting the Rain tomorrow.
# This Genreal Technique of combining the reults of many models is called as"ensembling",This Genreal Technique of combining the reults of many models is called as"ensembling"
# ![download%20%281%29.jpg](attachment:download%20%281%29.jpg)
# 

# we can access individual decision trees using model.estimators_

# In[57]:


len(model.estimators_)


# **it appears that the model has learned the training example perfectly, and doesnt generalize well to previously unseen examples.This phenomenon is called "overfitting ", and reducing overfitting is one of the most important parts of any machine learning projects**

# ### Visualization

# we can visualize the decision tree learned fro m the training data

# In[58]:


from sklearn.tree import plot_tree,export_text


# In[59]:


plt.figure(figsize=(80,20))
plot_tree(model.estimators_[0],feature_names=X_train.columns,max_depth=2,filled=True,rounded=True)


# In[60]:


plt.figure(figsize=(80,20))
plot_tree(model.estimators_[15],feature_names=X_train.columns,max_depth=2,filled=True,rounded=True)


# ### Feature Importance

# Random forest  assign an "importance to each feature,by combining the importance values from individual trees.<br>
# 
# Creating a dataframe and visualize the important feature.

# In[61]:


#X_train.columns


# In[62]:


model.feature_importances_


# In[63]:


importance_df=pd.DataFrame({
    "Feature":X_train.columns,"Importance":model.feature_importances_
}).sort_values('Importance',ascending=False)


# In[64]:


importance_df.head(10)


# In[65]:


plt.title("Feature importance")
sns.barplot(data=importance_df.head(10),x="Importance",y="Feature")


# ### Hyperparameter Tuning and Overfitting

# As we saw the RandomForest classifier memorized all training examples,leading to a 100 % training accuracy, while the validation accuracy was only marginally better than the dumb based model.This phenomenon of overfitting  and in this section we will look at some stratigies for reducing overfitting .The process of reducing overfitting is known as *regularization* .
# 
# The RandomForest classifier accepts several arguments ,some of which can be modified to reduce overfitting.
# Underfitting and Overfitting:<br>
# ![download.png](attachment:download.png)

# These arguments are called as hyperparameters because they must be confugired manually as opposed to a parameters within the model which are learned from the data.Well explore a couple of hyperparameters
# - max_depth
# - n_estimators
# - max_features

# **max_depth**<br>
# By reducing number of max depth of a RandomForest ,we can prevent the tree from memorizing all traing examples,which may lead to better generalization.<br>
# By,default no max depth is specified,which is why each tree has a training accuracy of 100% .<br>
# you can specify a max-depth to reduce overfittting .<br>

# In[66]:


def max_depth_error(md):
    model=RandomForestClassifier(max_depth=md,random_state=42)
    model.fit(X_train,train_target)
    train_error=1-model.score(X_train,train_target)
    val_error=1-model.score(X_val,val_target)
    return{'max_depth':md,'Training error':train_error,'Validation Error':val_error}


# In[67]:


error_df=pd.DataFrame([max_depth_error(md) for md in range(1,21)])


# In[68]:


plt.figure()
plt.plot(error_df['max_depth'],error_df['Training error'])
plt.plot(error_df['max_depth'],error_df['Validation Error'])
plt.title("training vs Validation Error")
plt.xticks(range(0,21,2))
plt.xlabel('max depth')
plt.ylabel('prediction error(1-accuracy)')
plt.legend(['Training','Validation'])        


# **Creating A base Model To compare with Hyperparamete tuned Model** 

# In[69]:


base_model=RandomForestClassifier(n_jobs=-1,random_state=42).fit(X_train,train_target)


# In[70]:


base_train_acc=base_model.score(X_train,train_target)
base_val_acc=base_model.score(X_val,val_target)


# In[71]:


base_acc=base_train_acc,base_val_acc
base_acc


# ### n_estimators
# 
# This argumet controls the number of decision trees in the Random forest.The default value of 100.For the larger datasets,it helps to have a greater number of estimators .as a general rule,try to have a few estiamtors as needed.
# 

# **10_estimators**

# In[72]:


model=RandomForestClassifier(random_state=42,n_jobs=-1,n_estimators=10)
model.fit(X_train,train_target)


# In[73]:


model.score(X_train,train_target),model.score(X_val,val_target)


# In[74]:


base_acc


# **500 estimators**

# In[75]:


get_ipython().run_cell_magic('time', '', 'model=RandomForestClassifier(random_state=42,n_jobs=-1,n_estimators=500)\nmodel.fit(X_train,train_target)')


# In[76]:


model.score(X_train,train_target),model.score(X_val,val_target)


# In[77]:


base_acc


#  Helper function test-params to make it easy to test hyperparameters.

# In[78]:


def test_params(**params):
    model=RandomForestClassifier(random_state=42,n_jobs=-1,**params).fit(X_train,train_target)
    return model.score(X_train,train_target),model.score(X_val,val_target)


# ** params we cane pass multiple arguments<br>
# ** = kwarg function<br>
# 
# test_params(max_depth=5,max_leaf_nodes=1024,n_estimators=1000)

# **Max_ features**

# Instead of picking all feature columns for every split ,we can specify that only a fraction of features be chosen randomly to figure out a split.
# max_features{“auto”, “sqrt”, “log2”}, int or float, default=”auto”
# The number of features to consider when looking for the best split:
# 
# If int, then consider max_features features at each split.
# 
# If float, then max_features is a fraction and round(max_features * n_features) features are considered at each split.
# 
# If “auto”, then max_features=sqrt(n_features).
# 
# If “sqrt”, then max_features=sqrt(n_features) (same as “auto”).
# 
# If “log2”, then max_features=log2(n_features).
# 
# If None, then max_features=n_features.
# 
# Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than max_features features.

# In[79]:


test_params(max_features='log2')


# In[80]:


test_params(max_features=3)


# In[81]:


test_params(max_features=6)


# In[82]:


base_acc


# ### bootstraps and max_samples
# 
# Defination(Wikipedia) -Bootstrap aggregating, also called bagging (from bootstrap aggregating), is a machine learning ensemble meta-algorithm designed to improve the stability and accuracy of machine learning algorithms used in statistical classification and regression. It also reduces variance and helps to avoid overfitting. Although it is usually applied to decision tree methods, it can be used with any type of method. Bagging is a special case of the model averaging approach.<br>
# ![660px-Ensemble_Bagging.svg.png](attachment:660px-Ensemble_Bagging.svg.png)<br>
# By default, a random forest doesnt use the entire dataset for training each decison tree.Instead it applies a technique called bootstrapping .For each tree, rows from the datset are picked one by one randomly ,with replacement ie some rows may not show up at all,while some rows may show up multiple times. <br>
# 
# bootstrap bool, default=True
# Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.
# bootstrappinf helps the random forest generalize better, because each decision tree only see a fractiob of the training set,snd some rows randomly get higher weightage than others.<br>
# 
# 

# In[83]:


test_params(bootstrap=False)


# In[84]:


test_params(bootstrap=True)


# when bootstrapping is enabled ,we can also control the number of fraction of rows to be considered for each bootstrap using max-samples.this can further generalize the model.

# In[85]:


test_params(max_samples=.8)  ### high overfitting high complexity ,when max-samples=1


# In[86]:


base_acc


# In[87]:


train_target.value_counts()/len(train_target)


# In[88]:


model.classes_


# #### Class Weight
# 
# When we have imbalanced data we can assign class weight to our target ,in order to overcome imbalnce to certain limit.  
# 

# In[89]:


test_params(class_weight="balanced")


# In[90]:


test_params(class_weight={"No":1,"Yes":2})


# ### Summarizing it together
# 
# Now we will  Make a Random Forest with customized hyperparameter .

# In[91]:


model=RandomForestClassifier(n_jobs=-1,n_estimators=500,max_features=20,max_depth=8,class_weight={"No":1,"Yes":1.5},random_state=42)


# In[92]:


model.fit(X_train,train_target)


# In[93]:


model.score(X_train,train_target),model.score(X_val,val_target)


# In[94]:


base_acc


# #### Confusion matrix

# In[95]:


test_pred=model.predict(X_test)


# In[96]:


cf_matrix = confusion_matrix(test_target,test_pred)
print(cf_matrix)


# In[97]:


plt.figure(figsize = (10,7))
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Greens')


# ### Summary :
# 
# We've increased the accuracy from 84.7% with a Base Model to 85.7% with a well tuned random forest .Depending on the dataset and kind of problem ,we may or may not see a significant improvement with hyperparameter tuning .
# This could be due to any of the following reasons:
#   - we may not have found the right mix of hyperparameter to regularize (reduce fitting) the model properly,and we should keep trying to impeove the model.
#   - we may have reached the limits of the modelling techniques we're currently using (random firest), we should try another modelling technique e.g. gradient boosting .
#   - we may have reached the limits of what we can predict using the given amoiunt of data,we may need more data to improv the model.
#   - we may have reached the limits of how well we can predict wheather it will rain tomorrow using the given weather measurements,and we may need more features (columns) to further improve the model. in many cases,we can also generate new feature using existing features (this is called feature engineering)
#   - Wheather it will rain tomorrow may be inherently random and chaotic phenomenon which simply cannot be predicted beyond a certin accuracy any amount od data for any number of wheather measuements with any modelling technique. 
# 
# Remember that ultimately all model are wrong,but some are useful.If we can rely on the model we've created today to make a travel decision for tomorrow,then the model is useful,even tohough it may  be sometimes be wrong.
#   

# finally lets compute the accuracy of the model on the test dataset
# 

# In[98]:


model.score(X_test,test_target)


# Notice That the test accuracy is lower maybe we have over optimized the validation set

# ### Making predictions on a New Inputs

# In[99]:


def predict_input(model,single_input):
    input_df=pd.DataFrame([single_input])
    input_df[numeric_cols]=imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols]=scaler.transform(input_df[numeric_cols])
    input_df[encoded_cols]=encoder.transform(input_df[categorical_cols])
    X_input=input_df[numeric_cols+encoded_cols]
    pred=model.predict(X_input)[0]
    prob=model.predict_proba(X_input)[0][list(model.classes_).index(pred)]
    return pred,prob


# In[100]:


new_input={'Date':'2021-06-20', 
           'Location':'uluru',
           'MinTemp':23.2,
           'MaxTemp':33.2, 
           'Rainfall':10.2, 
           'Evaporation':4.2,
           'Sunshine':np.nan,
           'WindGustDir':'NNW',
           'WindGustSpeed':60.0,
           'WindDir9am':'NW',
           'WindDir3pm':'NNE',
           'WindSpeed9am':13.0,
           'WindSpeed3pm':20.0, 
           'Humidity9am':89.0, 
           'Humidity3pm':50.0,
           'Pressure9am':1002.8, 
           'Pressure3pm':1001.5, 
           'Cloud9am':8.0, 
           'Cloud3pm':5.0, 
           'Temp9am':25.7,
           'Temp3pm':33.0,
           'RainToday':'Yes'}


# In[101]:


predict_input(model,new_input)


# The prediction shows there is 57.98 % probability that it will rain tomorrow ,based on provided input.

# ### Useful links to articles 
# Articles link here -
# - [understanding-random-forest](https://towardsdatascience.com/understanding-random-forest-58381e0602d2) <br>
# - [random-forest-algorithm]( https://builtin.com/data-science/random-forest-algorithm)<br>
# - [how-ensemble-learning-works](https://machinelearningmastery.com/how-ensemble-learning-works/)<br>
# - [from-a-single-decision-tree-to-a-random-forest](https://www.dataversity.net/from-a-single-decision-tree-to-a-random-forest/)<br>
# - [improving-random-forest-in-python](https://towardsdatascience.com/improving-random-forest-in-python-part-1-893916666cd)<br>
# - [understanding-random-forest](https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/)
