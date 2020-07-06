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

The precision is defined as TP / TP+FP. was very consistent for the low risk loans represented by the 1 row, all siz algorithms had a value of 1.0. For the precision of the high_risk loans, the BalancedRandomForestClassifier algorithm performed best at .40 with all other algorithms posting values below .1. 

Recall or sensitivity is defined as TP/(TP+FN). For the high risk loans (represented with a 0), BalancedRandomForestClassifier had a low of 0.47 and EasyEnsembleClassifier had the best score at 0.92. For the low risk loans (represented with a 1), ClusterCentroids had the low of .66 and BalancedRandomForestClassifier had the greatest value at 1.0. The algorithm with the overall best Recall was BalancedRandomForestClassifier at .99 and ClusterCentroids was lowest at 0.66.

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
  
SMOTE </br>
  pre   rec   f1 </br>
0 0.02  0.72  .03 </br>
1 1.00  0.73  .84 </br>
  0.99  0.73  .84 </br>
  
ClusterCentroids </br>
  pre   rec   f1 </br>
0 0.01  0.75  .03 </br>
1 1.00  0.66  .80 </br>
  0.99  0.66  .79 </br>

SMOTEENN </br>
  pre   rec   f1 </br>
0 0.02  0.72  .03 </br>
1 1.00  0.72  .84 </br>
  0.99  0.72  .83 </br>
  
**Extension: Credit Risk Ensemble**
BalancedRandomForestClassifier </br>
  pre   rec   f1 </br>
0 0.40  0.47  .43 </br>
1 1.00  1.00  1.00 </br>
  0.99  0.99  .99 </br>

EasyEnsembleClassifier </br>
  pre   rec   f1 </br>
0 0.09  0.92  .16 </br>
1 1.00  0.94  .97 </br>
  0.99  0.94  .97 </br>

### F1 Score
This is defined as the harmonic mean between the Precision and Recall to give me a sense of the overall stability of the algorithm. ClusterCentroids yielded the lowest value of .79 with EasyEnsembleClassifier at .97 and BalancedRandomForestClassifier at .99. 

## Final Reccommendation

From the results provided, it can be seen that if you are going to choose between random sampling or SMOTE to do the oversampling, SMOTE will result in slightly better results and accuracy with respect to this dataset. ClusterCentroids algorithm showed the lowest overvall Precision at 66%, so if high precision is required this one should be avoided. SMOTEENN showed results very comparable to SMOTE and random sampler.

The two algorithms that stood out were BalancedRandomForestClassifier and EasyEnsembleClassifier. Using the F1 score, we can see that the highest F1 from the other four algorithm was only .84, both of these have values at or above .97. Looking a bit closer we can see that BalancedRandomForestClassifier provides an overall more balanced f1 score with values of .43 and 1.0 compared with .16 and .97 for EasyEnsembleClassifier. 

Based off of these results, it appears clear to me that BalancedRandomForestClassifier yielded the best overall results as indicated with its .99 F1 score. To get best prediction results with respect to each combintaion of precision, recall and high_risk and low_risk loans, this is the algorithm that showed overall best stability. Additional reccommendation would be to get a larger dataset and verify these prediction results to ensure the algorithm produced is not suffering from overfitting.
