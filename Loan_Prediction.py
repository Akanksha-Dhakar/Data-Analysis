#!/usr/bin/env python
# coding: utf-8

# # Loan Prediction

# In[329]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder


# In[330]:


data = pd.read_csv("C:/Users/akanksha/Desktop/myproject/loan-Approval-Prediction-master/train.csv")
test = pd.read_csv("C:/Users/akanksha/Desktop/myproject/loan-Approval-Prediction-master/test.csv")


# In[331]:


data.head() # top 5 row dataset


# In[332]:


data.tail() # top last 5 row dataset


# In[334]:


data.shape # 614 rows and 13 columns is there 


# # Store total number of observation in training dataset
# # Store total number of columns in testing dataset

# In[335]:


data_length= len(data)
test_col = len(test.columns)


# # Understanding the various featurs(columns) of the dataset

# In[336]:


data.describe() # summary of numerical variables for training dataset 


# # Get the unique values and their frequency of variable property_Area

# In[337]:


data["Property_Area"].value_counts() #Get the unique values and their frequency of variable Property_Area


# In[338]:


## box plot for understanding the distribution and to observe the outliers (%matplotlin inline)
# Histogram of variable Application


# In[339]:


data['ApplicantIncome'].hist()


# In[340]:


# Box Plot for variable ApplicantIncome of training data set
data.boxplot(column='ApplicantIncome')


# In[341]:


# Box Plot for variable ApplicantIncome by variable Education of training data set
data.boxplot(column='ApplicantIncome', by = 'Education')


# In[342]:


# Histogram of variable LoanAmount
data['LoanAmount'].hist(bins=50)


# In[343]:


# Box Plot for variable LoanAmount of training data set
data.boxplot(column='LoanAmount')


# In[344]:


# Box Plot for variable LoanAmount by variable Gender of training data set
data.boxplot(column='LoanAmount', by = 'Gender')


# In[345]:


import seaborn as sns
sns.countplot(x='Education',hue='Loan_Status',data=data)


# In[346]:


#Dependent column values
data['Dependents'].value_counts()


# # Understanding Distribution of Categorical Variables# 

# In[347]:


# Loan approval rates in absolute numbers
loan_approval = data['Loan_Status'].value_counts()['Y']
print(loan_approval)


# In[348]:


# Credit History and Loan Status
pd.crosstab(data ['Credit_History'], data ['Loan_Status'], margins=True)


# In[349]:


# Replace missing value of Self_Employed with more frequent category

data['Self_Employed'].fillna('No',inplace=True)


# # Outliers of LoanAmount and applicant Income

# In[350]:


# Add both ApplicantIncome and CoapplicantIncome to TotalIncome
data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome']
# Looking at the distribtion of TotalIncome
data['LoanAmount'].hist(bins=20)


# In[351]:


# Perform log transformation of TotalIncome to make it closer to normal
data['LoanAmount_log'] = np.log(data['LoanAmount'])
# Looking at the distribtion of TotalIncome_log
data['LoanAmount_log'].hist(bins=20)


# # Data preparation for model Building

# In[352]:


# Impute missing values for Gender
# Impute missing values for Married
# Impute missing values for Dependents
# Impute missing values for Credit_History
# Convert all non-numeric values to number


# In[353]:


data['Gender'].fillna(data['Gender'].mode()[0],inplace=True)
data['Married'].fillna(data['Married'].mode()[0],inplace=True)
data['Dependents'].fillna(data['Dependents'].mode()[0],inplace=True)
data['Credit_History'].fillna(data['Credit_History'].mode()[0],inplace=True)
cat=['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area']
for var in cat:
    le = preprocessing.LabelEncoder()
    data[var]=le.fit_transform(data[var].astype('str'))
data.dtypes


# # Classification Function

# In[354]:


#Generic function for making a classification model and accessing performance
#Fit the model:
#Make predictions on training set:
#Print accuracy
#Perform k-fold cross-validation with 5 folds
# Filter training dat
# The target we're using to train the algorithm.
# Training the algorithm using the predictors and target
#Record error from each cross-validation run
#Fit the model again so that it can be refered outside the function


# In[355]:


#Import models from scikit learn module:
from sklearn import metrics
def classification_model(model, data, predictors, outcome):
   model.fit(data[predictors],data[outcome])
   predictions = model.predict(data[predictors])
   accuracy = metrics.accuracy_score(predictions,data[outcome])
   print ("Accuracy : %s" % "{0:.3%}".format(accuracy))
   kf = KFold(data.shape[0], n_folds=5)
   error = []
   for train, test in kf:
       train_predictors = (data[predictors].iloc[train,:])
       train_target = data[outcome].iloc[train]
       model.fit(train_predictors, train_target)
       error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test])) 
   print ("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))
   model.fit(data[predictors],data[outcome])       


# # Model Building

# In[356]:


#Combining both train and test dataset
#Create a flag for Train and Test Data set


# In[357]:


data['Type']='Train'
test['Type']='Test'
fullData = pd.concat([data,test],axis=0, sort=True)
fullData.isnull().sum()


# In[358]:


#Identify categorical and continuous variables


# In[359]:


ID_col = ['Loan_ID']
target_col = ["Loan_Status"]
cat_cols = ['Credit_History','Dependents','Gender','Married','Education','Property_Area','Self_Employed']


# In[360]:


#Imputing Missing values with mean for continuous variable


# In[361]:


fullData['LoanAmount'].fillna(fullData['LoanAmount'].mean(), inplace=True)
fullData['LoanAmount_log'].fillna(fullData['LoanAmount_log'].mean(), inplace=True)
fullData['Loan_Amount_Term'].fillna(fullData['Loan_Amount_Term'].mean(), inplace=True)
fullData['ApplicantIncome'].fillna(fullData['ApplicantIncome'].mean(), inplace=True)
fullData['CoapplicantIncome'].fillna(fullData['CoapplicantIncome'].mean(), inplace=True)


# In[362]:


#Imputing Missing values with mode for categorical variablesmmmmmmmmm


# In[363]:


fullData['Gender'].fillna(fullData['Gender'].mode()[0], inplace=True)
fullData['Married'].fillna(fullData['Married'].mode()[0], inplace=True)
fullData['Dependents'].fillna(fullData['Dependents'].mode()[0], inplace=True)
fullData['Loan_Amount_Term'].fillna(fullData['Loan_Amount_Term'].mode()[0], inplace=True)
fullData['Credit_History'].fillna(fullData['Credit_History'].mode()[0], inplace=True)


# In[364]:


#Create a new column as Total Income
#Histogram for Total Income

fullData['TotalIncome']=fullData['ApplicantIncome'] + fullData['CoapplicantIncome']
fullData['TotalIncome_log'] = np.log(fullData['TotalIncome'])
fullData['TotalIncome_log'].hist(bins=20)


# In[365]:


#create label encoders for categorical features
for var in cat_cols:
    number = LabelEncoder()
    fullData[var] = number.fit_transform(fullData[var].astype('str'))
train_modified=fullData[fullData['Type']=='Train']
test_modified=fullData[fullData['Type']=='Test']
train_modified["Loan_Status"] = number.fit_transform(train_modified["Loan_Status"].astype('str'))


# # LogisticRegression

# In[366]:


from sklearn.linear_model import LogisticRegression


# In[367]:


predictors_Random=['Credit_History','Education','Gender']
x_train = train_modified[list(predictors_Logistic)].values
y_train = train_modified["Loan_Status"].values
x_test=test_modified[list(predictors_Logistic)].values


# In[368]:


model = LogisticRegression()


# In[369]:


model.fit(x_train, y_train)


# In[370]:


predicted= model.predict(x_test)


# In[371]:


predicted = number.inverse_transform(predicted)


# In[372]:


test_modified['Loan_Status']=predicted
outcome_var = 'Loan_Status'
classification_model(model, data,predictors_Logistic,outcome_var)


# In[ ]:





# In[ ]:




