# Bootcamp: UCB-VIRT-DATA-PT-03-2020-U-B-TTH

### Bootcamp Challenge #17 - 7/5/2020
Bootcamp Challenge 17: Module supervised-machine-learning

### Links Used
- [LoanStats](https://courses.bootcampspot.com/courses/140/files/38961/download?wrap=1)

### Challenge Description
**Objectives**
The goals of this challenge are for you to:
- Implement machine learning models.
- Use resampling to attempt to address class imbalance.
- Evaluate the performance of machine learning models.

## Technologies Used
- Jupyter Notebook
- Python, Pandas, NumPy, SMOTE, SMOTEENN
- sklearn, imblearn, LogisticRegression, BalancedRandomForestClassifier, EasyEnsembleClassifier
- GIT

## Methodology, Summary, Purpose 
The goal of this challenge was to first import through Jupyter Notebook a CSV file that contains numerous rows pertaining to Loan Data. 
There was initial data transformation to set loan_status to either low_risk (if it is current) or high_risk (if it is Late, in Default, or in Grace Period).
In order to transform the data to be able to use train_test_split, I used a LabelEncoder in particular to encode the strings. 

I identified the y (dependent) variable to be the loan_status which had been enumerated to either low_risk or high_risk. I then identified the X dataset as all columns from the original CSV that were not loan_status and also excluding the columns that were entirely empty e.g. url. Segregating the data out to the two categories from loan_status showed a spread of 68,470 to 347. 

I then ran a train_test_split method on the X and y datasets provided to split the data between train and test datasets for both the X and y groups. y_train returned a split of {1: 51,366, 0: 246} - clearly indicating an unbalanced dataset. One method to balance the sets was to randomly oversample the set representing 0 or the high risk loans to oversample the data so it equals the number of good loans that were current. After both reached a value of 51,366, I ran a LogisticRegression on the resampled data. Once I had the model created, I was able to use it to predict the input of X_test stored as y_pred. With the predicted results, I was able to run a confusion matrix comparing it to y_test, this produced a balanced accuracy score of 72.31%. 

Running the SMOTE test also required an oversampling of the data to 51,366 points for both high_risk and low_risk for loan_status. It yieled a slightly higher balanced accuracy score of 72.64%.

Running the CluserCentroids algorithm does the opposite and undersamples the larger dataset until it equals to the smaller of the two. Thus running ClusterControids reduced both high_risk and low_risk loan_status to 246 entries. This dataset yielded a 70.67% balanced accuracy score. 

Running the SMOTEENN algorithm does a combination of both oversampling and undersampling the datasets to resolve the unbalanced datasets. The y_resampled variable showed a closely distribtuion spread of {0: 68470, 1: 64463}. This algorithm yieled nearly identical results with ClusterControids with a balanced accuracy score of 70.67%. 

Then using the same datset, I first over sampled the dataset until the high_risk and low_risk dataset numbers were equal, using the RandomOverSampler method. I then ran an Easy Ensemble AdaBoost classifier to train new models and calculate their respective accuracy scores. For BalancedRandomForestClassifier, I set the n_estimators to 100, max_depth=2, and random_state=1. This yielded an accuracy score of 73.06%, representing the highest seen thus far.

Finally I used the EasyEnsembleClassifier(n_estimators=100, random_state=1) algorithm. Based off the X_train and y_train data that was entered into it, it yieled by far the best accuracy score of 93.16%.

### Balanced Accuracy Scores
- RandomOverSampler: .7231
- SMOTE: .7264
- ClusterCentroids: .7067
- SMOTEENN: .7067

**Extension: Credit Risk Ensemble**
- BalancedRandomForestClassifier: 0.7306
- EasyEnsembleClassifier: .9316

### Precision & Recall Scores
RandomOverSampler </br>
  pre   rec   f1 </br>
0 0.02  0.72  .03 </br>
1 1.00  0.72  .84 </br>
  0.99  0.72  .83 </br>
</br>
  
SMOTE
  pre   rec   f1
0 0.02  0.72  .03
1 1.00  0.73  .84
  0.99  0.73  .84
  
ClusterCentroids
  pre   rec   f1
0 0.01  0.75  .03
1 1.00  0.66  .80
  0.99  0.66  .79

SMOTEENN
  pre   rec   f1
0 0.02  0.72  .03
1 1.00  0.72  .84
  0.99  0.72  .83
  
**Extension: Credit Risk Ensemble**
BalancedRandomForestClassifier
  pre   rec   f1
0 0.40  0.47  .43
1 1.00  1.00  1.00
  0.99  0.99  .99

RandomOverSampler
  pre   rec   f1
0 0.09  0.92  .16
1 1.00  0.94  .97
  0.99  0.94  .97

## Final Reccommendation
