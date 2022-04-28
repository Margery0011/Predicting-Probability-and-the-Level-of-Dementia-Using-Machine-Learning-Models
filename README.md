# Predicting-Probability-and-the-Level-of-Dementia-Using-Machine-Learning-Models
Dementia is a general term for a decline in mental ability severe enough to interfere with daily life. Clinical Dementia Rating Scale (CDR) is a global rating scale for staging patients diagnosed with dementia. In this project, I am using Cross-Sectional and Longitudinal OASIS MRI structural and demographic data to train machine learning models to predict if and at what level an individual has dementia. This problem is formulated as a binary classification problem (CDR = 0 and CDR > 0) and a multiclass problem (CDR = 0, CDR = 0.5, CDR = 1).  

# Background
### Dementia

Dementia is not a specific disease. It is an overall term for impaired ability to remember, think, or make decisions that interferes with doing everyday activities. Though dementia mostly affects older adults, it is not a part of normal aging.
### Clinical Dementia Rating(CDR):

CDR is a global rating scale for staging patients diagnosed with dementia. It evaluates cognitive, behavioral, and functional aspects of Alzheimer disease and other dementias. In this project  Clinical Dementia Rating (CDR) values provided in the data set will be used as "targets" for training the classification models. 

# Dataset

Combined Cross-Sectional and Longitudinal data from OASIS brain project (http://www.oasis-brains.org/) to train machine learning models

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

  - ![4541651116271_ pic](https://user-images.githubusercontent.com/89502586/165670241-da66277f-2263-48bf-964f-7e0f6c664435.jpg)

  When the data is skewed, it is good to consider using the median value for replacing the missing values. Note that imputing missing data with median value can only be done with numerical. Imputing missing data with most_frequent values can be done with numerical and categorical data.
  
  Because SES is discrete data, so use "most_frequent as strategy , and MMSE is not normal distribution， so use "median" as strategy to do imputer 
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
-  Change  "Gender" from categorical data to numerica data

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

### Age Group & Dementia

![image](https://user-images.githubusercontent.com/89502586/165406754-591e9bfa-c521-409f-8487-4201a948b3c0.png)

Majority of cases of Dementia are in the age group of 70-80 years (around 45%) while second most highest cases are in 80-90 years of age.

### Gender & Dementia

![image](https://user-images.githubusercontent.com/89502586/165406003-65f9fd84-37c1-434c-af10-6dad0a47a990.png)

For Male, most number of dementia cases are reported in the age of around 80 .
For Female, dementia is prevalent in 70 years of Age.Most of the cases happens generally after 65 years of age

### Data Balance / Imbalance

Data imbalance usually reflects an unequal distribution of classes within a dataset. If we train a binary classification model without fixing the imbalanced data , the model will be completely biased

![image](https://user-images.githubusercontent.com/89502586/165405749-2b8ce36d-8acb-4a5a-b7d5-703334df50f6.png)

Although it is not perfectly even, it is basically a balanced dataset which does not need resampling

## Split Training dataset/ Test dataset

Method: The original dataset is randomly divided into training set and validation set. This sample is divided into two parts according to the proportion of 80%~20%, 80% of the sample is used for training the model; 20% of the sample is used for model validation.

![4221651032844_ pic](https://user-images.githubusercontent.com/89502586/165439282-85054c2f-1d03-4ae9-8ebe-88a86cb569d1.jpg)

```
X_bi = np.asarray(data_bi[['Age','Educ','SES','MMSE','eTIV','nWBV','Gender']])
Y_bi = np.asarray(data_bi['CDR'])
validation_size = 0.20
seed = 42
X_train_bi, X_validation_bi, Y_train_bi, Y_validation_bi = train_test_split(X_bi, Y_bi, test_size=validation_size, random_state=seed)
```
## Algorithem Comparison

 **K-Fold Cross-Validation**

![image](https://user-images.githubusercontent.com/89502586/165439616-c0709acb-9c77-44ae-81fa-020159c17c9b.png)

- Divide the original data set into 3 parts
- Each group uses one of them as the test set, and the remaining 2 (K-1) as the training set
- The training set becomes K * D (D represents the number of data samples contained in each copy)
- Finally, the average value of the classification rate obtained for k times is calculated as the real classification rate of the model or hypothesis function

![image](https://user-images.githubusercontent.com/89502586/165440484-99f14b70-1be9-4d74-8663-7d67d559c35e.png)

Here I choosed K = 3 to do cross validation. 

### cross val score

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

By comparing cross-validation score, I found that Random Forest and Logistic Regression classfiers perform better than others, so I am going to tune these two classifier

## RandomForest Classifier Tuning

Random forest is an ensemble tool which takes a subset of observations and a subset of variables to build a decision trees. It builds multiple such decision tree and amalgamate them together to get a more accurate and stable prediction.

### Decision Tree

A decision tree is drawn upside down with its root at the top. The green circle represents a condition/internal node, based on which the tree splits into branches/ edges. The end of the branch that doesn’t split anymore is the decision/leaf.

![4551651119196_ pic](https://user-images.githubusercontent.com/89502586/165675751-91363c30-7004-4341-8f00-f747315281de.jpg)


### Find best n_estimator

- ***n_estimators-(integer)- Default=10***

The number of trees your want to build within a Random Forest before aggregating the predictions. 

```
scorel = []
for i in range(1,100):
    rfc = RandomForestClassifier(n_estimators=i,
                                 n_jobs=-1,
                                 random_state=42)
    score = cross_val_score(rfc,X_bi, Y_bi,cv=3).mean()
    scorel.append(score)

print(max(scorel),([*range(1,100)][scorel.index(max(scorel))]))
plt.figure(figsize=[20,5])
plt.plot(range(1,100),scorel)
plt.xlabel("n_estimator")
plt.ylabel("accuracy")
plt.show()
```
![image](https://user-images.githubusercontent.com/89502586/165406929-78765508-e89c-460d-8457-4c4ceb7e1e49.png)

The figure shows that when n_estimator = 69 , the model can get best accuracy.

### Hyperparameter Optimization for the RandomForest Model

Use Grid Search to search optimal values for hyperparameters. To tune hyperparameters, follow the steps below:

- Create a model instance of the RandomForest Model
- Specify hyperparameters with all possible values
- Define performance evaluation metrics
- Apply cross-validation
- Train the model using the training dataset
- Determine the best values for the hyperparameters given.

**GridSearcgCV()**

It is Exhaustive search method.After we created a list of hyperparameters dictionary , it will find the optimal paramters among all candidate parameter choices, by looping through, trying every possibility, then the best performing parameter is the final result. 

![4161651014505_ pic](https://user-images.githubusercontent.com/89502586/165407450-c3079dc0-ade8-4865-a30a-cd688c41bd0c.jpg)

- ***criterion-(string)-Default =”gini”***

Measures the quality of each split. It can either be “gini” or “entropy”. “gini” uses the Gini impurity while “entropy” makes the split based on the information gain.

- ***max_depth-(integer or none)- Default=None***

This selects how deep you want to make your decision trees, if set it as "None" means there is no limitation to  the depth of the subtree

- ***min_samples_leaf-(integer, float)-Default=1***

This parameter helps you determine minimum size of the end node of each decision tree. The end node is also known as a leaf.

Then build the classifier using these best parameters selected by grid search.

# Model Performance Evaluation

For Binary Classification, the output is either Non-Dementia (0) or Dementia (1)

**Confusion Matrix**

- True positive (TP): Prediction is 1 and X is Dementia, we want that
- True negative (TN): Prediction is 0 and X is Non-Dementia, we want that too
- False positive (FP): Prediction is 1 and X is Non-Dementia, false alarm, bad
- False negative (FN): Prediction is 0 and X is Dementia, the worst


![4211651032026_ pic_hd](https://user-images.githubusercontent.com/89502586/165437849-1afceb55-4b5c-4ae8-a145-fd8518802652.jpg)


**Precision**

The ratio of accurately predicted positive observations to the total predicted positive observations.

Precision = TP/TP+FP

**Recall**

The ratio of accurately predicted positive observations to all observations in actual class – yes.

Recall = TP/TP+FN

**F1 score** 

It is the harmonic mean of precision and recall. It takes both false positive and false negatives into account.F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account.


F1 Score = 2(Recall * Precision) / (Recall + Precision)

## Random Forest Classifier Performance Report


![image](https://user-images.githubusercontent.com/89502586/165438086-a89ea735-f864-4854-ac59-6a8363b416da.png)

![image](https://user-images.githubusercontent.com/89502586/165438099-f7174178-349c-412e-9368-31046f088ff2.png)

### Result explain 

- Тhe cell on row one, column one, contains the number of True Negatives (in our case, this cell contains the number of correctly predicted individual who does not have dementia ). The model truly predicts 59 of them.

- Тhe cell on row one, column two, contains the number of False Positives (in our case, this cell contains the number of predicted as Non-Dementia , but actually is Dementia ). The model falsely predicts 14 real patients.

-  Тhe cell on row two, column one, contains the number of False Negatives (in our case, this cell contains the number of predicted as Dementia , but actually is Non-Dementia ). The model falsely predicts 2 individuals.
    
- Тhe cell on row two, column two, contains the number of True Positives (in our case, this cell contains the number of correctly predicted Dementia that has Dementia ). The model truly predicts 47 samples



## Logistic Regression

Logistic Regression is used to solve classification problems. Models are trained on historical labelled datasets and aim to predict which category new observations will belong to. Logistic regression is well suited when we need to predict a binary answer (only 2 possible values like yes or no).

**Why Scale ?**

- Make sure features are on a similar scale

For example, if we were to make a regression on the population, the regression coefficients would differ considerably between the dimensions of "number" and "million"

![image](https://user-images.githubusercontent.com/89502586/165407836-f496773a-a0b5-4296-9f40-f9ebadf98c25.png)

#### Find the best Threshold

The output of a Logistic regression model is a probability. We can select a threshold value. If the probability is greater than this threshold value, the event is predicted positive otherwise it is predicted negative.

In binary classification, when a model gives us a score instead of the prediction itself, we usually need to convert this score into a prediction applying a threshold, the default threshold for sklearn is 0.5

![image](https://user-images.githubusercontent.com/89502586/165407880-368c50ef-86d3-4188-8be6-00f5faa0d445.png)
![4181651015004_ pic](https://user-images.githubusercontent.com/89502586/165407964-c34f3c63-ddb1-4897-ac6f-226c4b2693db.jpg)



## Logistic Regression Performance Report


### Logistic Regression Classifier using Default Threshold


![image](https://user-images.githubusercontent.com/89502586/165808912-20ac13c5-2281-4523-885a-99dd5443b0b3.png)
![image](https://user-images.githubusercontent.com/89502586/165808676-23475474-a5cc-4680-a66a-18c7e82f56d9.png)


### Logistic Regression Classifier using Chosen Threshold


![4611651166185_ pic](https://user-images.githubusercontent.com/89502586/165809633-96e62aa1-a110-4a69-abfe-b692db39a3d8.jpg)
![image](https://user-images.githubusercontent.com/89502586/165809276-4590b026-67b4-4ef9-91b5-1f7208734594.png)

Although there is not so much difference in the accuracy among 2 models, but the classifier after tuning threshold has better recall and less False Negative cases, so it is better when it comes to clinical diagnosis

### Choose Model : Recall or Precision ?

Which could be more tolerate ? 

- “false negative" : tell ill people they are healthy?
- “false positives”: tell healthy people they are ill?

For me , I think false negative is not tolerable here.

Recall tells us the prediction accuracy among only actual positives. It means how correct our prediction is among ill people. That matters in that case. 

That is why we have to minimize false negatives which means we are trying to maximize **recall**. It can cost us lower accuracies, which is still sufficient. 

So the classifier using chosen thershold would be better since the false negative cases have decreased. 

# Multi-labels Classification

## Imbalanced Data


![image](https://user-images.githubusercontent.com/89502586/165408453-16cd788c-6085-4e0b-93df-16d95fdf502e.png)

This dataset is highly imbalanced, most cases are 0.0, so I need to oversample in order to get an even dataset

### Oversample

![4591651127243_ pic](https://user-images.githubusercontent.com/89502586/165691787-1fdebabb-4cab-430c-8ada-94b7dfd2d165.jpg)

use Synthetic Minority Oversampling Technique(SMOTE) to accomplish oversample, the basic idea of the SMOTE algorithm is to analyze the minority class samples and artificially synthesize new samples based on the minority class samples to add to the dataset

```
from imblearn.over_sampling import SMOTE
oversample =SMOTE()
X_re,Y_re = oversample.fit_resample(X_train,Y_train)
Y_reD = pd.DataFrame(Y_re, columns = ['CDR'])
```
After re-sample, the distribution of data has become even

![image](https://user-images.githubusercontent.com/89502586/165420339-6523dc2e-2e87-496f-86b3-5e428cddb652.png)

## Algorithem Comparison 

### GridSearcv.best_score_ Comparison 

![image](https://user-images.githubusercontent.com/89502586/165810996-2acf5670-eb0a-4f94-9637-edaa9655415e.png)
### Cross_val_score Comparison Visualization

![image](https://user-images.githubusercontent.com/89502586/165810698-a768f782-3a8d-4e7c-b3a7-4ed8e74e0e7b.png)

**As the results say, random forest classifier performs best, so tune its hyparamters**

### Best Classifier 

After Tuning hyparameters by GridSearch, I have gotten classifier with best performance

![image](https://user-images.githubusercontent.com/89502586/165441361-52a10153-7c66-4207-ab7a-6d1daa36e734.png)

### Performance


![image](https://user-images.githubusercontent.com/89502586/165812611-55e4a640-21c6-4b98-8775-1fefe5c795b0.png)

![image](https://user-images.githubusercontent.com/89502586/165812756-3d9f28ad-2986-4ea0-a8cd-11d3e47c40d1.png)


- Class 0.0: High Precision, High recall
- Class 0.5: High Precision, Low Recall
- Class 1.0: Low Precision, High Recall

Class 0.0 has been separated well, Class 0.5 and Class 1.0 are not separated nicely, maybe need a new classifier to separated these two classes alone.

**Possible reason**

- Few samples in these 2 classes
- Invalid information introduced by oversample
- The differences in these 2 classes are not significant
# Summary 

In this project, I divided it into a binary classification and a multi-label classification problem. The dataset has 608 samples in total. I used eight features to train these models and used CDR values as the targets. 

For binary classification , after comparing different algorithms, I decided to tune the hypeparameters of random forest classifier and logistic regression classifier.For random forest classifier, the first step is to determine the "n_estimator", then tune other hyparamates by gridsearchcv(). As for Logistic Regression, decide which should be the best threshold.   

For multi-label classification, I choose to tune the RF classifier after comparing the algorithm. As a result, Class 0.0 has been separated well, and Class 0.5 and Class 1.0 are not separated nicely.

## Future  Study   

For Binary classification:
Increase the Recall and Decrease the False Negative prediction cases in Random Forest Classifier
For Multi-label classification:
Build a classifier target at separating 0.5 and 1.0  cases
Use different Oversample methods like Borderline-SMOTE
Overall:
Find more data to train model
Build other Classifiers by image data  
