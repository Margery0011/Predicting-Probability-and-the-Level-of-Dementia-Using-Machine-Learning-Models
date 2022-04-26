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
### Data Clean

- Unnecessary data removal:
  - Rows missing CDR value (608 subjects remained)
  - ASF column : Because it is almost correlated with factor eTIV (99%)	
  ![image](https://user-images.githubusercontent.com/89502586/165398227-378edec8-aa57-49a4-ab78-e1d4080ef23e.png)

  - Hand column: All right-handed, not representative
- Missing values imputer
  - SES 
  ![image](https://user-images.githubusercontent.com/89502586/165398389-62587e41-1e1e-4f41-ba7b-c6b9871b2b33.png)
  - MMSE
  ![image](https://user-images.githubusercontent.com/89502586/165398437-b79cda66-2532-4dcb-b5b0-650067707739.png)

## Data Explore

# Binary Classification
## Algorithem Comparison
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

