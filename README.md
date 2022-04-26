# Predicting-Probability-and-the-Level-of-Dementia-Using-Machine-Learning-Models
Dementia is a general term for a decline in mental ability severe enough to interfere with daily life. Clinical Dementia Rating Scale (CDR) is a global rating scale for staging patients diagnosed with dementia. In her project, she is using Cross-Sectional and longitudinal OASIS MRI structural and demographic data to train machine learning models to predict if and at what level an individual has dementia. This problem is formulated as a binary classification problem (CDR = 0 and CDR > 0) and a multiclass problem (CDR = 0, CDR = 0.5, CDR = 1).  

# Background
### Dementia

Dementia is not a specific disease. It is an overall term for impaired ability to remember, think, or make decisions that interferes with doing everyday activities. Though dementia mostly affects older adults, it is not a part of normal aging.
### Clinical Dementia Rating(CDR):

CDR is a global rating scale for staging patients diagnosed with dementia. It evaluates cognitive, behavioral, and functional aspects of Alzheimer disease and other dementias. In this project  Clinical Dementia Rating (CDR) values provided in the data set will be used as "targets" for training the classification models. 

# Dataset

Combined Cross-Sectional and Longitudinal data  from OASIS  brain project (http://www.oasis-brains.org/) to train machine learning models

Cross-Sectional dataset consists of a cross-sectional collection for 436 persons including male and female.For each person, 3 to 4 T1-weighted MRI scans that were obtained in single scan sessions are included.
Longitudinal dataset  consists of a longitudinal collection of 373 subjects aged 60 to 96. Each subject was scanned on two or more visits, here we only use their first visit result. 

After concating , there are 809 subjects in total.

## Inputs(Features) and Outputs(Targets)

I am using 8 features as input and CDR as the output to train these models, the descriptions are summaried in the table below.

![image](https://user-images.githubusercontent.com/89502586/165397913-37a0b439-f19c-40f4-a337-a960c35ddec8.png)



## Data Pre-process

### Import Libraries 

```
import pandas as pd
import seaborn as sns
from pylab import rcParams
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute  import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
```
### Load 2 Datasets & concat them
```
cross = pd.read_csv("oasis_cross-sectional.csv")
long =  pd.read_csv("oasis_longitudinal.csv")
cross_cp = cross.copy()
long_cp = long.copy()
```

Remove unnecessary columns 
```
cross_cp.drop(columns=['ID','Delay'],inplace=True)
long_cp = long.rename(columns={'EDUC':'Educ'})
long_cp.drop(columns=['Subject ID','MRI ID','Group','Visit','MR Delay'],inplace=True)
data = pd.concat([cross_cp,long_cp])
```

### Data Clean

- Unnecessary data removal:
  - Rows missing CDR value (608 subjects remained)
  - ASF column : Because it is almost correlated with factor eTIV (99%)	
  ![image](https://user-images.githubusercontent.com/89502586/165398227-378edec8-aa57-49a4-ab78-e1d4080ef23e.png)

  - Hand column: All right-handed, not representative
- Missing values imputer
  - SES 
  ![image](https://user-images.githubusercontent.com/89502586/165398389-62587e41-1e1e-4f41-ba7b-c6b9871b2b33.png)
  Because SES is discrete data, so use "most_frequent as strategy , and MMSE is not normal distribution， so use "median" as strategy
  - MMSE
  ![image](https://user-images.githubusercontent.com/89502586/165398437-b79cda66-2532-4dcb-b5b0-650067707739.png)


```
data_cp = data.copy()
# remove ASF column because it is almost correlated with eTIV 
# remove hand column
data_cp.drop(['ASF'],axis = 1,inplace = True)
data_cp.drop(['Hand'],axis = 1,inplace = True)
imputer = SimpleImputer ( missing_values = np.nan,strategy='most_frequent')
imputer.fit(data_cp[['SES']])
data_cp[['SES']] = imputer.fit_transform(data_cp[['SES']])

# missing values imputer 
imputer = SimpleImputer ( missing_values = np.nan,strategy='median')
imputer.fit(data_cp[['MMSE']])
data_cp[['MMSE']] = imputer.fit_transform(data_cp[['MMSE']])
```
Change  "Gender" from categorical data to numerica data

```
gender_map = {'M':0, 'F':1}
data_cp['Gender'] = data_cp['M/F'].map(gender_map)
data_cp.drop(['M/F'],axis=1,inplace= True)
```


# Binary Classification

To predict whether this individual has dementia by Binary classifier, if the outout of target is 0 means "CDR = 0" and this individual is "Non-Dementia" , if the outout of target is 1 means "CDR > 0" and this individual is "Dementia" 

```
data_cp['CDR'] = data_cp['CDR'].astype(str)
data_cp['CDR'] = data_cp['CDR'].str.replace('2','1')
data_bi = data_cp.copy()
ClassDict = {'0.0':0,'0.5':1,'1.0':1}
data_bi.loc[:,'CDR'] = data_bi.loc[:, 'CDR'].apply(lambda x: ClassDict[x])
```
## Data Explore

#### Age Group & Dementia

![image](https://user-images.githubusercontent.com/89502586/165405580-073aa06d-74ed-4027-ae21-8e92332bb484.png)

Majority of cases of Dementia are in the age group of 70-80 years (around 45%) while second most highest cases are in 80-90 years of age.

#### Gender & Dementia

![image](https://user-images.githubusercontent.com/89502586/165406003-65f9fd84-37c1-434c-af10-6dad0a47a990.png)

For Male, most number of dementia cases are reported in the age of around 80 .
For Female, dementia is prevalent in 70 years of Age.Most of the cases happens generally after 65 years of age

#### Balanced / Imbalanced

![image](https://user-images.githubusercontent.com/89502586/165405749-2b8ce36d-8acb-4a5a-b7d5-703334df50f6.png)

It is basically a balanced data which does not need re-sample


## Algorithem Comparison

### cross_val_score

```
models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('RFC', RandomForestClassifier()))
results = []
names = []
print('(', 'Model, ', 'Cross-Validation Accuracy: Mean, Stdev',')')
for name, model in models:
    kfold = KFold(n_splits=3, random_state = seed,shuffle=True)
    cv_results = cross_val_score(model, X_train_bi_sc, Y_train_bi, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = (name, format(cv_results.mean(), '.2f'))
    print(msg)
# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
plt.xlabel("different classifier")
plt.ylabel("Accuracy score")
ax.set_xticklabels(names)
plt.show();
```

![image](https://user-images.githubusercontent.com/89502586/165406555-bf6348d6-d2b0-490c-9455-9fd2de892e54.png)


## RandomForest 
### RandomForest Classifier Tuning
### Performance
## Logistic Regression
### Tune
### Performance

# Multi-labels Classification
## Algorithem Comparison 
## Performance
### Future  Study   

